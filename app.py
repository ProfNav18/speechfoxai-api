# SpeechFoxAI — Render API backend
# Loads the v3 model + pre-trained probes from files in this repo
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"  # cuts librosa's heavy numba/LLVM import footprint — matters a lot on 512MB

import io, base64
import numpy as np
import torch
torch.set_num_threads(1)  # free tier gives a fraction of a CPU core; multithreading just adds overhead here

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
try:
    v3_ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False, mmap=True)
except TypeError:
    # older torch versions don't support mmap= — fall back cleanly
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

# ---------- Inference + explainability (split into light + heavy calls) ----------
def get_risk_level(prob_fake):
    if prob_fake < 0.3:
        return "Low Risk", "green", "Consistent with genuine human speech."
    elif prob_fake < 0.7:
        return "Medium Risk", "orange", "Some characteristics are ambiguous — treat with caution."
    else:
        return "High Risk", "red", "Consistent with AI-generated or synthetic speech."

def quick_analysis(filepath):
    """LIGHT call: forward pass only, no backward pass, no Grad-CAM hooks.
    This is what the verdict/risk-level/breakdown/explanation depend on — no gradient
    tracking means no autograd graph to hold in memory, which is the main saving here."""
    print(">>> quick_analysis started", flush=True)
    captured = {}
    def _acoustic_hook(module, inp, out):
        captured["acoustic"] = out.detach().flatten(1)
    def _prosodic_hook(module, inp, out):
        _, h_n = out
        captured["prosodic"] = h_n[-1].detach()

    h1 = v3_model.resnet_backbone.register_forward_hook(_acoustic_hook)
    h2 = v3_model.gru.register_forward_hook(_prosodic_hook)

    try:
        waveform = load_and_window(filepath)
        raw_prosodic = extract_prosodic_features_fast(waveform)
        prosodic = normalize_prosodic(raw_prosodic)
        mel = extract_mel_spectrogram(waveform)
        mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(device)
        pros_t = torch.from_numpy(prosodic).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = v3_model(mel_t, pros_t)
        fused_prob = torch.sigmoid(output).item()
        ac_emb = captured["acoustic"].cpu().numpy()[0]
        pr_emb = captured["prosodic"].cpu().numpy()[0]
    finally:
        h1.remove(); h2.remove()

    acoustic_prob = acoustic_probe.predict_proba(ac_emb.reshape(1, -1))[0][1]
    prosodic_prob = prosodic_probe.predict_proba(pr_emb.reshape(1, -1))[0][1]
    print(">>> quick_analysis done", flush=True)

    return {
        "fused_prob": fused_prob,
        "acoustic_prob": float(acoustic_prob),
        "prosodic_prob": float(prosodic_prob),
        "raw_prosodic": raw_prosodic,
    }

def gradcam_analysis(filepath):
    """HEAVY call: forward+backward pass for Grad-CAM. Separate from quick_analysis
    so it runs in isolation, with its own memory budget, not competing with anything else."""
    print(">>> gradcam_analysis started", flush=True)
    captured = {}
    def _gradcam_fwd_hook(module, inp, out):
        captured["gradcam_acts"] = out
    def _gradcam_bwd_hook(module, grad_in, grad_out):
        captured["gradcam_grads"] = grad_out[0]

    target_layer = v3_model.resnet_backbone[7]
    h1 = target_layer.register_forward_hook(_gradcam_fwd_hook)
    h2 = target_layer.register_full_backward_hook(_gradcam_bwd_hook)

    try:
        waveform = load_and_window(filepath)
        mel = extract_mel_spectrogram(waveform)
        prosodic = normalize_prosodic(extract_prosodic_features_fast(waveform))
        mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(device)
        pros_t = torch.from_numpy(prosodic).unsqueeze(0).float().to(device)

        v3_model.zero_grad()
        with torch.backends.cudnn.flags(enabled=False):
            output = v3_model(mel_t, pros_t)
            output.backward()
        fused_prob = torch.sigmoid(output).item()
        acts = captured["gradcam_acts"][0]
        grads = captured["gradcam_grads"][0]
    finally:
        h1.remove(); h2.remove()

    weights = grads.mean(dim=(1, 2))
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=device)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = torch.relu(cam).detach().cpu().numpy()
    cam = cam / (cam.max() + 1e-8)
    print(">>> gradcam_analysis done", flush=True)

    return {"cam": cam, "mel": mel, "fused_prob": fused_prob}

def render_gradcam_image(cam, mel, prob_fake):
    """Pure rendering — no model call. Trimmed down (2x supersample instead of 4x,
    single panel instead of dual) to keep this isolated call as light as possible."""
    from scipy.ndimage import zoom, gaussian_filter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SUPERSAMPLE = 2
    target_h, target_w = mel.shape[0] * SUPERSAMPLE, mel.shape[1] * SUPERSAMPLE
    zoom_factors = (target_h / cam.shape[0], target_w / cam.shape[1])
    cam_hires = np.clip(zoom(cam, zoom_factors, order=1), 0, 1)
    cam_smooth = gaussian_filter(cam_hires, sigma=SUPERSAMPLE * 1.2)
    cam_smooth = cam_smooth / (cam_smooth.max() + 1e-8)
    alpha_map = np.clip(cam_smooth, 0, 1) * 0.75

    fig, ax = plt.subplots(figsize=(9, 4), dpi=100)
    mel_extent = [0, mel.shape[1], 0, mel.shape[0]]
    ax.imshow(mel, aspect="auto", origin="lower", cmap="gray", interpolation="bilinear", extent=mel_extent)
    im = ax.imshow(cam_smooth, aspect="auto", origin="lower", cmap="jet",
                    alpha=alpha_map, vmin=0, vmax=1, interpolation="bilinear", extent=mel_extent)
    ax.set_title(f"Grad-CAM (prob_fake={prob_fake:.4f})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img.load()  # force-decode now, so the BytesIO buffer can be freed
    return img

def describe_deviation(name, z, pos_word, neg_word):
    if abs(z) < 0.5:
        return f"{name} is close to typical training values ({z:+.1f} SD)."
    direction = pos_word if z > 0 else neg_word
    return f"{name} is {direction} than typical training values ({z:+.1f} SD)."

def generate_explanation(analysis):
    """Takes the dict from quick_analysis() — the feature-deviation lines don't need
    Grad-CAM at all. The spatial/attention line is added separately (see
    describe_gradcam_spatial below) once/if the heatmap call completes."""
    acoustic_prob, prosodic_prob, fused_prob = analysis["acoustic_prob"], analysis["prosodic_prob"], analysis["fused_prob"]
    tier, color, risk_desc = get_risk_level(fused_prob)

    raw_prosodic = analysis["raw_prosodic"]
    f0_z = (raw_prosodic[0].mean() - PROSODIC_STATS["f0_mean"]) / PROSODIC_STATS["f0_std"]
    centroid_z = (raw_prosodic[2].mean() - PROSODIC_STATS["centroid_mean"]) / PROSODIC_STATS["centroid_std"]
    rms_z = (raw_prosodic[1].mean() - PROSODIC_STATS["rms_mean"]) / PROSODIC_STATS["rms_std"]

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
    return "\n".join(lines)

def describe_gradcam_spatial(cam, mel):
    """Separate from generate_explanation — only called from the heavy Grad-CAM path,
    once that data actually exists."""
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

    return (f"The model's attention was {concentration_desc} around {approx_freq_hz:.0f} Hz, "
            f"roughly {approx_time_sec:.1f}s into the clip ({concentration*100:.0f}% of total "
            f"attribution concentrated in the top 20% of the spectrogram).")

# ---------- Visual GUI (matches the Flutter app's amber/crimson design system) ----------
RISK_COLORS = {"green": "#FFB020", "orange": "#FFB020", "red": "#E63950"}

def format_risk_badge(verdict, risk_level, risk_color, prob_fake):
    color = "#E63950" if verdict == "FAKE" else "#FFB020"
    text_color = "#1A1206" if verdict != "FAKE" else "#FFF3E0"
    verdict_label = "🧑 HUMAN VOICE" if verdict == "REAL" else "🤖 AI-GENERATED"
    return f"""
    <div style="background-color:{color}; color:{text_color}; padding:20px; border-radius:4px; text-align:center; font-family: 'Orbitron', sans-serif;">
        <div style="font-size:26px; font-weight:600; letter-spacing:0.05em;">{verdict_label}</div>
        <div style="font-size:13px; margin-top:8px; opacity:0.9; font-family: 'Sora', sans-serif; letter-spacing:0.04em;">{risk_level.upper()} — SPOOF PROBABILITY: {prob_fake*100:.1f}%</div>
    </div>
    """

def format_breakdown_bars(acoustic_prob, prosodic_prob):
    def bar(label, value):
        pct = value * 100
        color = "#E63950" if value > 0.5 else "#FFB020"
        return f"""
        <div style="margin-bottom:12px; font-family: 'Sora', sans-serif;">
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px; letter-spacing:0.04em; color:#D9CFC3;">
                <span>{label.upper()}</span><span>{pct:.1f}%</span>
            </div>
            <div style="background:#241E18; overflow:hidden; height:14px;">
                <div style="width:{pct}%; background:{color}; height:100%;"></div>
            </div>
        </div>
        """
    return f"""
    <div style="padding:14px; background:#171310; border-left: 2px solid #FFB020;">
        {bar("Acoustic (spectral) branch", acoustic_prob)}
        {bar("Prosodic (pitch / rhythm / energy) branch", prosodic_prob)}
    </div>
    """

def predict_quick(audio_filepath):
    """LIGHT path: verdict, risk level, breakdown, explanation. No Grad-CAM, no backward pass."""
    print(f">>> predict_quick called, audio_filepath={audio_filepath}", flush=True)
    if audio_filepath is None:
        empty_badge = "<div style='padding:18px; text-align:center; color:#6B6259; font-family: Sora, sans-serif;'>Record or upload a clip to begin.</div>"
        return empty_badge, "", "", "", 0.0, "", 0.0, 0.0, ""

    import gc

    analysis = quick_analysis(audio_filepath)
    fused_prob = analysis["fused_prob"]
    tier, color, risk_desc = get_risk_level(fused_prob)
    explanation = generate_explanation(analysis)
    verdict = "REAL" if fused_prob < DEMO_THRESHOLD else "FAKE"
    acoustic_prob, prosodic_prob = analysis["acoustic_prob"], analysis["prosodic_prob"]

    badge_html = format_risk_badge(verdict, tier, color, fused_prob)
    breakdown_html = format_breakdown_bars(acoustic_prob, prosodic_prob)
    explanation_md = "### Explanation\n\n" + explanation.replace("\n", "\n\n")

    del analysis
    gc.collect()

    # Visible outputs (badge_html, breakdown_html, explanation_md) for the browser GUI,
    # PLUS plain hidden values for Flutter to read from the same call.
    return (badge_html, breakdown_html, explanation_md,
            verdict, round(float(fused_prob), 4), tier, round(float(acoustic_prob), 4),
            round(float(prosodic_prob), 4), explanation)

def predict_heatmap(audio_filepath):
    """HEAVY path: Grad-CAM heatmap only. Called separately (button, or a later Flutter
    request) so it never competes with the light path for memory."""
    print(f">>> predict_heatmap called, audio_filepath={audio_filepath}", flush=True)
    if audio_filepath is None:
        return None, ""

    import gc

    gc_result = gradcam_analysis(audio_filepath)
    heatmap_img = render_gradcam_image(gc_result["cam"], gc_result["mel"], gc_result["fused_prob"])
    spatial_text = describe_gradcam_spatial(gc_result["cam"], gc_result["mel"])

    del gc_result
    gc.collect()

    return heatmap_img, spatial_text

with gr.Blocks(title="SpeechFoxAI — Explainable Audio Deepfake Detector") as demo:
    gr.HTML("<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Orbitron:wght@500;600&family=Sora:wght@400;500;600&display=swap'>")
    gr.Markdown("# 🦊 SpeechFox<span style='color:#FFB020'>AI</span>")
    gr.Markdown("### Explainable Audio Deepfake Detection\nRecord or upload a voice sample. Detection runs first (fast); the spectrogram heatmap loads separately afterward.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or upload a voice sample")
            analyze_btn = gr.Button("🔍 RUN DETECTION", variant="primary", size="lg")
            heatmap_btn = gr.Button("🌡️ SHOW HEATMAP (slower)", variant="secondary", size="lg")
        with gr.Column(scale=1):
            risk_badge = gr.HTML(label="Risk Level")
            breakdown_html = gr.HTML(label="Acoustic vs. Prosodic Breakdown")

    with gr.Row():
        heatmap_output = gr.Image(label="Grad-CAM Spectrogram Heatmap", type="pil")
        explanation_output = gr.Markdown(label="Explanation")

    # Hidden fields — not shown in the browser UI, but present in the API response for Flutter
    api_verdict = gr.Textbox(visible=False)
    api_prob_fake = gr.Number(visible=False)
    api_risk_level = gr.Textbox(visible=False)
    api_acoustic_prob = gr.Number(visible=False)
    api_prosodic_prob = gr.Number(visible=False)
    api_explanation_raw = gr.Textbox(visible=False)
    api_spatial_text = gr.Textbox(visible=False)

    analyze_btn.click(
        fn=predict_quick,
        inputs=audio_input,
        outputs=[risk_badge, breakdown_html, explanation_output,
                  api_verdict, api_prob_fake, api_risk_level, api_acoustic_prob,
                  api_prosodic_prob, api_explanation_raw],
        api_name="analyze_quick"
    )

    heatmap_btn.click(
        fn=predict_heatmap,
        inputs=audio_input,
        outputs=[heatmap_output, api_spatial_text],
        api_name="analyze_heatmap"
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
