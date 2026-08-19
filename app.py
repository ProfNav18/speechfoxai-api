# SpeechFoxAI — Hugging Face Spaces API backend
# Loads the v3 model + pre-trained probes from local Space files (uploaded alongside this app.py)
import os, io, base64
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import librosa
import joblib
import gradio as gr
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "model_finetuned_v3_noiserobust.pth"
ACOUSTIC_PROBE_PATH = "acoustic_probe.joblib"
PROSODIC_PROBE_PATH = "prosodic_probe.joblib"

SR = 16000
TARGET_LENGTH = SR * 4
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
DEMO_THRESHOLD = 0.5

# ---------- Preprocessing (final validated pipeline: SR fix + soft de-clip) ----------
def soft_declip(waveform, threshold=0.95, knee=0.05):
    out = waveform.copy()
    over = np.abs(out) > threshold
    if not np.any(over):
        return out
    sign = np.sign(out[over])
    excess = np.abs(out[over]) - threshold
    compressed_excess = knee * np.tanh(excess / knee)
    out[over] = sign * (threshold + compressed_excess)
    return out.astype(np.float32)

def load_and_window(filepath, target_length=TARGET_LENGTH, sr=SR):
    data, _ = librosa.load(filepath, sr=sr)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = soft_declip(data)
    current_length = len(data)
    if current_length < target_length:
        data = np.tile(data, int(np.ceil(target_length / current_length)))[:target_length]
    elif current_length > target_length:
        data = data[:target_length]
    return data.astype(np.float32)

def extract_mel_spectrogram(waveform, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

def extract_prosodic_features_fast(waveform, sr=SR, hop_length=HOP_LENGTH):
    f0 = librosa.yin(waveform, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
                      sr=sr, hop_length=hop_length).astype(np.float32)
    rms = librosa.feature.rms(y=waveform, hop_length=hop_length)[0].astype(np.float32)
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr, hop_length=hop_length)[0].astype(np.float32)
    min_len = min(len(f0), len(rms), len(centroid))
    return np.stack([f0[:min_len], rms[:min_len], centroid[:min_len]], axis=0)

# ---------- Model architecture ----------
class ResNet18GRUFusionTunable(nn.Module):
    def __init__(self, gru_hidden_size=128, gru_layers=2, dropout=0.3, hidden_width=128, prosodic_dim=3):
        super().__init__()
        resnet = tv_models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.gru = nn.GRU(input_size=prosodic_dim, hidden_size=gru_hidden_size,
                           num_layers=gru_layers, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(512 + gru_hidden_size, hidden_width),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_width, 1)
        )

    def forward(self, mel, prosodic):
        acoustic_feat = self.resnet_backbone(mel).flatten(1)
        _, h_n = self.gru(prosodic.permute(0, 2, 1))
        fused = torch.cat([acoustic_feat, h_n[-1]], dim=1)
        return self.fusion(fused).squeeze(-1)

# ---------- Load checkpoint + probes ----------
v3_ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
best_params = v3_ckpt["best_params"]
PROSODIC_STATS = v3_ckpt["prosodic_stats"]

def normalize_prosodic(prosodic, stats=PROSODIC_STATS):
    out = prosodic.copy().astype(np.float32)
    out[0] = (out[0] - stats["f0_mean"]) / stats["f0_std"]
    out[1] = (out[1] - stats["rms_mean"]) / stats["rms_std"]
    out[2] = (out[2] - stats["centroid_mean"]) / stats["centroid_std"]
    return out

v3_model = ResNet18GRUFusionTunable(dropout=best_params["dropout"], hidden_width=best_params["hidden_width"]).to(device)
v3_model.load_state_dict(v3_ckpt["model_state"])
v3_model.eval()

acoustic_probe = joblib.load(ACOUSTIC_PROBE_PATH)
prosodic_probe = joblib.load(PROSODIC_PROBE_PATH)

print("Model and probes loaded successfully.")

# ---------- Inference + explainability ----------
def score_audio(filepath):
    waveform = load_and_window(filepath)
    mel = extract_mel_spectrogram(waveform)
    prosodic = normalize_prosodic(extract_prosodic_features_fast(waveform))
    mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(device)
    pros_t = torch.from_numpy(prosodic).unsqueeze(0).float().to(device)
    with torch.no_grad():
        return torch.sigmoid(v3_model(mel_t, pros_t)).item()

def get_risk_level(prob_fake):
    if prob_fake < 0.3:
        return "Low Risk", "green", "Consistent with genuine human speech."
    elif prob_fake < 0.7:
        return "Medium Risk", "orange", "Some characteristics are ambiguous — treat with caution."
    else:
        return "High Risk", "red", "Consistent with AI-generated or synthetic speech."

_captured = {}
def _acoustic_hook(module, inp, out):
    _captured["acoustic"] = out.detach().flatten(1)
def _prosodic_hook(module, inp, out):
    _, h_n = out
    _captured["prosodic"] = h_n[-1].detach()

def get_score_breakdown(filepath):
    h1 = v3_model.resnet_backbone.register_forward_hook(_acoustic_hook)
    h2 = v3_model.gru.register_forward_hook(_prosodic_hook)
    try:
        prob_fake = score_audio(filepath)
        ac_emb = _captured["acoustic"].cpu().numpy()[0]
        pr_emb = _captured["prosodic"].cpu().numpy()[0]
    finally:
        h1.remove()
        h2.remove()
    acoustic_prob = acoustic_probe.predict_proba(ac_emb.reshape(1, -1))[0][1]
    prosodic_prob = prosodic_probe.predict_proba(pr_emb.reshape(1, -1))[0][1]
    return acoustic_prob, prosodic_prob, prob_fake

def compute_gradcam(filepath):
    v3_model.eval()
    activations, gradients = {}, {}
    def fwd_hook(module, inp, out):
        activations["value"] = out
    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]
    target_layer = v3_model.resnet_backbone[7]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    waveform = load_and_window(filepath)
    mel = extract_mel_spectrogram(waveform)
    prosodic = normalize_prosodic(extract_prosodic_features_fast(waveform))
    mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(device)
    pros_t = torch.from_numpy(prosodic).unsqueeze(0).float().to(device)

    v3_model.zero_grad()
    with torch.backends.cudnn.flags(enabled=False):
        output = v3_model(mel_t, pros_t)
        output.backward()
    h1.remove()
    h2.remove()

    acts = activations["value"][0]
    grads = gradients["value"][0]
    weights = grads.mean(dim=(1, 2))
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=device)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = torch.relu(cam).detach().cpu().numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, mel, torch.sigmoid(output).item()

def plot_gradcam(filepath):
    from scipy.ndimage import zoom, gaussian_filter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cam, mel, prob_fake = compute_gradcam(filepath)
    SUPERSAMPLE = 4
    target_h, target_w = mel.shape[0] * SUPERSAMPLE, mel.shape[1] * SUPERSAMPLE
    zoom_factors = (target_h / cam.shape[0], target_w / cam.shape[1])
    cam_hires = np.clip(zoom(cam, zoom_factors, order=3), 0, 1)
    cam_smooth = gaussian_filter(cam_hires, sigma=SUPERSAMPLE * 1.2)
    cam_smooth = cam_smooth / (cam_smooth.max() + 1e-8)
    alpha_map = np.clip(cam_smooth, 0, 1) * 0.75

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=150)
    axes[0].imshow(mel, aspect="auto", origin="lower", cmap="magma", interpolation="bicubic")
    axes[0].set_title("Mel-spectrogram")
    mel_extent = [0, mel.shape[1], 0, mel.shape[0]]
    axes[1].imshow(mel, aspect="auto", origin="lower", cmap="gray", interpolation="bicubic", extent=mel_extent)
    im = axes[1].imshow(cam_smooth, aspect="auto", origin="lower", cmap="jet",
                         alpha=alpha_map, vmin=0, vmax=1, interpolation="bilinear", extent=mel_extent)
    axes[1].set_title(f"Grad-CAM (prob_fake={prob_fake:.4f})")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def describe_deviation(name, z, pos_word, neg_word):
    if abs(z) < 0.5:
        return f"{name} is close to typical training values ({z:+.1f} SD)."
    direction = pos_word if z > 0 else neg_word
    return f"{name} is {direction} than typical training values ({z:+.1f} SD)."

def generate_explanation(filepath):
    acoustic_prob, prosodic_prob, fused_prob = get_score_breakdown(filepath)
    tier, color, risk_desc = get_risk_level(fused_prob)

    waveform = load_and_window(filepath)
    raw_prosodic = extract_prosodic_features_fast(waveform)
    f0_z = (raw_prosodic[0].mean() - PROSODIC_STATS["f0_mean"]) / PROSODIC_STATS["f0_std"]
    centroid_z = (raw_prosodic[2].mean() - PROSODIC_STATS["centroid_mean"]) / PROSODIC_STATS["centroid_std"]
    rms_z = (raw_prosodic[1].mean() - PROSODIC_STATS["rms_mean"]) / PROSODIC_STATS["rms_std"]

    cam, mel, _ = compute_gradcam(filepath)
    freq_bins, time_bins = cam.shape
    freq_idx, time_idx = np.indices(cam.shape)
    total_weight = cam.sum() + 1e-8
    freq_centroid_frac = (freq_idx * cam).sum() / total_weight / freq_bins
    time_centroid_frac = (time_idx * cam).sum() / total_weight / time_bins

    mel_freqs = librosa.mel_frequencies(n_mels=mel.shape[0], fmin=0, fmax=SR / 2)
    approx_freq_hz = mel_freqs[int(np.clip(freq_centroid_frac * (mel.shape[0] - 1), 0, mel.shape[0] - 1))]
    approx_time_sec = time_centroid_frac * (TARGET_LENGTH / SR)

    sorted_vals = np.sort(cam.flatten())[::-1]
    top20_count = max(1, int(0.2 * len(sorted_vals)))
    concentration = sorted_vals[:top20_count].sum() / (sorted_vals.sum() + 1e-8)
    concentration_desc = ("sharply localized" if concentration > 0.7
                           else "moderately localized" if concentration > 0.5
                           else "spread across much of the clip")

    lines = [f"Overall assessment: {tier} (spoof probability {fused_prob*100:.1f}%)."]
    if abs(acoustic_prob - prosodic_prob) > 0.15:
        dominant = "acoustic (spectral)" if acoustic_prob > prosodic_prob else "prosodic (pitch/rhythm/energy)"
        lines.append(f"This assessment is primarily driven by the {dominant} branch "
                     f"(acoustic: {acoustic_prob*100:.1f}%, prosodic: {prosodic_prob*100:.1f}%).")
    else:
        lines.append(f"Both branches broadly agree (acoustic: {acoustic_prob*100:.1f}%, "
                     f"prosodic: {prosodic_prob*100:.1f}%).")
    lines.append(describe_deviation("Spectral centroid", centroid_z, "elevated (brighter/noisier)", "reduced (duller)"))
    lines.append(describe_deviation("Average pitch", f0_z, "higher", "lower"))
    lines.append(describe_deviation("Overall energy/loudness", rms_z, "louder", "quieter"))
    lines.append(f"The model's attention was {concentration_desc} around {approx_freq_hz:.0f} Hz, "
                 f"roughly {approx_time_sec:.1f}s into the clip ({concentration*100:.0f}% of total "
                 f"attribution concentrated in the top 20% of the spectrogram).")
    return "\n".join(lines)

# ---------- API-facing function ----------
def analyze_audio_api(audio_filepath):
    """This is the function the Flutter app calls via the auto-generated Gradio API."""
    if audio_filepath is None:
        return {"error": "No audio provided"}, None

    acoustic_prob, prosodic_prob, fused_prob = get_score_breakdown(audio_filepath)
    tier, color, risk_desc = get_risk_level(fused_prob)
    explanation = generate_explanation(audio_filepath)
    heatmap_img = plot_gradcam(audio_filepath)

    result = {
        "verdict": "REAL" if fused_prob < DEMO_THRESHOLD else "FAKE",
        "prob_fake": round(float(fused_prob), 4),
        "risk_level": tier,
        "risk_color": color,
        "risk_description": risk_desc,
        "acoustic_prob_fake": round(float(acoustic_prob), 4),
        "prosodic_prob_fake": round(float(prosodic_prob), 4),
        "explanation": explanation,
    }
    return result, heatmap_img

demo = gr.Interface(
    fn=analyze_audio_api,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Voice sample"),
    outputs=[gr.JSON(label="Analysis"), gr.Image(label="Grad-CAM Heatmap")],
    title="SpeechFoxAI API",
    description="Explainable audio deepfake detection — ResNet18 + GRU fusion. Called by the SpeechFoxAI Flutter app.",
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
