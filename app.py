import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import altair as alt
import requests
from io import BytesIO
from transformers import AutoImageProcessor
from model import LungDiseaseModel
import os
import numpy as np

# --- GRAD-CAM IMPORTS ---
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LungScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS STYLING ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* GLOBAL STYLES */
.stApp {
    background: linear-gradient(145deg, hsl(222, 47%, 6%) 0%, hsl(217, 33%, 10%) 50%, hsl(222, 47%, 8%) 100%);
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: hsl(210, 40%, 98%) !important;
}

p, span, label, .stMarkdown, li {
    color: hsl(215, 20%, 80%) !important; 
}

/* HERO SECTION */
.hero-title {
    background: linear-gradient(135deg, hsl(217, 91%, 60%) 0%, hsl(199, 89%, 48%) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    text-align: center;
    color: hsl(215, 20%, 65%) !important;
    font-size: 1rem;
    margin-top: 8px;
    margin-bottom: 2rem;
}

/* CARDS */
.glass-card {
    background: linear-gradient(135deg, hsla(217, 33%, 17%, 0.6) 0%, hsla(222, 47%, 11%, 0.8) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid hsla(217, 33%, 30%, 0.3);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}

.glass-card-glow {
    box-shadow: 0 0 40px -10px hsla(217, 91%, 60%, 0.15);
}

.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.875rem;
    font-weight: 600;
    color: hsl(210, 40%, 98%);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid hsla(217, 33%, 30%, 0.3);
}

/* INPUTS */
[data-testid="stFileUploader"] {
    background: hsla(217, 33%, 17%, 0.3);
    border: 2px dashed hsla(217, 91%, 60%, 0.3);
    border-radius: 12px;
    padding: 20px;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: hsla(217, 91%, 60%, 0.6);
    background: hsla(217, 33%, 17%, 0.5);
}

.stTextInput > div > div > input {
    background: hsla(217, 33%, 17%, 0.5) !important;
    border: 1px solid hsla(217, 33%, 30%, 0.5) !important;
    border-radius: 10px !important;
    color: hsl(210, 40%, 98%) !important;
    padding: 12px 16px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* BUTTONS */
.stButton > button {
    width: 100%;
    border-radius: 10px !important;
    height: 3rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg, hsl(217, 91%, 60%) 0%, hsl(217, 91%, 50%) 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 14px -3px hsla(217, 91%, 60%, 0.4) !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px -3px hsla(217, 91%, 60%, 0.6) !important;
}

button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid hsla(217, 91%, 70%, 0.5) !important;
    color: hsl(210, 40%, 90%) !important;
}
button[kind="secondary"]:hover {
    background: hsla(217, 33%, 30%, 0.5) !important;
    border-color: hsl(217, 91%, 70%) !important;
    color: white !important;
}

/* METRICS */
[data-testid="stMetricValue"] {
    background: transparent;
    color: #ffffff !important;
    text-shadow: 0 0 10px hsla(217, 91%, 60%, 0.5);
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: hsl(215, 20%, 70%) !important;
    font-size: 0.8rem !important;
}

/* ALERTS */
.alert-box {
    padding: 16px 20px; border-radius: 12px; margin: 16px 0;
    display: flex; align-items: flex-start; gap: 12px;
}
.alert-success { background: hsla(142, 76%, 36%, 0.15); border: 1px solid hsla(142, 76%, 36%, 0.3); }
.alert-warning { background: hsla(38, 92%, 50%, 0.15); border: 1px solid hsla(38, 92%, 50%, 0.3); }
.alert-error { background: hsla(0, 84%, 60%, 0.15); border: 1px solid hsla(0, 84%, 60%, 0.3); }

.alert-title { font-weight: 600; font-size: 0.9rem; margin-bottom: 4px; color: white !important; }
.alert-message { font-size: 0.85rem; opacity: 0.9; color: #e2e8f0 !important; }

/* MISC */
.divider {
    display: flex; align-items: center; text-align: center; margin: 20px 0;
    color: hsl(215, 20%, 45%); font-size: 0.75rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.1em;
}
.divider::before, .divider::after { content: ''; flex: 1; border-bottom: 1px solid hsla(217, 33%, 30%, 0.3); }
.divider::before { margin-right: 16px; } .divider::after { margin-left: 16px; }

.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: hsla(142, 76%, 36%, 0.15); border: 1px solid hsla(142, 76%, 36%, 0.3);
    color: hsl(142, 76%, 60%); padding: 6px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 500;
}
.status-dot { width: 6px; height: 6px; background: hsl(142, 76%, 50%); border-radius: 50%; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

.image-container {
    background: hsla(217, 33%, 12%, 0.5); border-radius: 12px; padding: 16px;
    border: 1px solid hsla(217, 33%, 30%, 0.2);
    display: flex; justify-content: center;
}
.image-container img { border-radius: 8px; max-height: 400px; object-fit: contain; }

.chart-container {
    background: hsla(217, 33%, 15%, 0.3); border-radius: 12px; padding: 20px; margin-top: 16px;
}

/* EXPANDER */
.streamlit-expanderHeader {
    background-color: hsla(217, 33%, 17%, 0.5) !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background-color: hsla(217, 33%, 12%, 0.3) !important;
    border-radius: 0 0 8px 8px !important;
    border: 1px solid hsla(217, 33%, 30%, 0.2);
    padding: 16px !important;
}

.info-box {
    background: hsla(217, 91%, 60%, 0.05); border: 1px solid hsla(217, 91%, 60%, 0.2);
    border-radius: 12px; padding: 30px; text-align: center;
}
.info-box-icon { font-size: 2.5rem; margin-bottom: 16px; }
.info-box-text { color: hsl(215, 20%, 75%); font-size: 1rem; }

.footer {
    text-align: center; padding: 24px; margin-top: 40px;
    border-top: 1px solid hsla(217, 33%, 30%, 0.2); color: hsl(215, 20%, 50%); font-size: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
# Pastikan path ini benar (gunakan r"..." untuk Windows path)
CHECKPOINT_PATH = r"output\best-checkpoint.ckpt" 
MODEL_NAME = "facebook/convnextv2-tiny-22k-224"
CLASS_NAMES = ['Normal', 'Pneumonia', 'Tuberculosis', 'Unknown']

DISEASE_INFO = {
    "Normal": {
        "desc": "Kondisi paru-paru tampak sehat dan bersih. Tidak ditemukan adanya opasitas, infiltrat, atau lesi yang mengindikasikan kelainan patologis.",
        "symptoms": "Tidak ada gejala gangguan pernapasan, saturasi oksigen normal, dan fungsi paru-paru berjalan baik."
    },
    "Pneumonia": {
        "desc": "Infeksi yang menyebabkan peradangan pada kantung udara di satu atau kedua paru-paru. Kantung udara dapat berisi cairan atau nanah.",
        "symptoms": "Batuk berdahak atau bernanah, demam, menggigil, dan kesulitan bernapas (sesak napas)."
    },
    "Tuberculosis": {
        "desc": "Penyakit menular yang disebabkan oleh bakteri Mycobacterium tuberculosis. Biasanya menyerang paru-paru namun dapat menyebar ke bagian tubuh lain.",
        "symptoms": "Batuk berlangsung lama (3 minggu atau lebih), nyeri dada, batuk darah, kelelahan, berat badan turun drastis, dan berkeringat di malam hari."
    },
    "Unknown": {
        "desc": "Gambar tidak dapat dikenali sebagai citra X-ray dada yang valid atau kualitas gambar terlalu buruk untuk dianalisis.",
        "symptoms": "Pastikan gambar yang diunggah adalah foto Rontgen dada (Chest X-Ray) dengan orientasi yang benar."
    }
}

# --- STATE MANAGEMENT ---
if 'uploaded_image' not in st.session_state: st.session_state.uploaded_image = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'prediction' not in st.session_state: st.session_state.prediction = None
if 'confidence' not in st.session_state: st.session_state.confidence = 0.0
if 'probs_history' not in st.session_state: st.session_state.probs_history = None
if 'gradcam_image' not in st.session_state: st.session_state.gradcam_image = None

# --- FUNCTIONS ---
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_weights = torch.ones(len(CLASS_NAMES))
    try:
        model = LungDiseaseModel.load_from_checkpoint(
            CHECKPOINT_PATH,
            map_location=device,
            model_name=MODEL_NAME,
            num_classes=len(CLASS_NAMES),
            class_weights=dummy_weights
        )
        model.eval()
        model.to(device)
        return model, device
    except:
        return None, None

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except:
        return None

def process_image(image):
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    size = (processor.size["shortest_edge"], processor.size["shortest_edge"]) \
           if "shortest_edge" in processor.size else (224, 224)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    return transform(image).unsqueeze(0)

def generate_gradcam(model, image, target_class_index=None):
    if not GRAD_CAM_AVAILABLE:
        return None
    
    # 1. Target Layer (Stage terakhir encoder ConvNeXt V2)
    target_layers = [model.model.convnextv2.encoder.stages[-1].layers[-1]]
    
    # 2. Preprocess for Model (Tensor)
    img_tensor = process_image(image).to(model.device)
    
    # 3. Preprocess for Visualization (Numpy Float RGB 0-1)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    size = (processor.size["shortest_edge"], processor.size["shortest_edge"]) \
           if "shortest_edge" in processor.size else (224, 224)
    img_resized = image.resize(size)
    rgb_img = np.float32(img_resized) / 255
    
    # 4. Generate CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class_index)] if target_class_index is not None else None
    
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # 5. Overlay
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization

def reset_analysis():
    st.session_state.analysis_done = False
    st.session_state.prediction = None
    st.session_state.confidence = 0.0
    st.session_state.probs_history = None
    st.session_state.gradcam_image = None

# --- UI HEADER ---
st.markdown("""
<div style="text-align: center; padding: 20px 0 10px 0;">
    <h1 class="hero-title">🫁 LungScan AI</h1>
    <p class="hero-subtitle">Clinical Decision Support System for Chest Radiography</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display: flex; justify-content: center; margin-bottom: 30px;">
    <div class="status-badge">
        <span class="status-dot"></span>
        System Online
    </div>
</div>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
with st.spinner("Initializing AI Engine..."):
    model, device = load_model()

if model is None:
    st.markdown("""
    <div class="alert-box alert-warning">
        <div>
            <div class="alert-title">⚠️ DEMO MODE</div>
            <div class="alert-message">Model checkpoint not found. Running in demonstration mode.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    demo_mode = True
else:
    demo_mode = False

# --- MAIN LAYOUT ---
col_input, col_display = st.columns([1, 1.5], gap="large")

# --- LEFT COLUMN (INPUT) ---
with col_input:
    st.markdown("""
    <div class="glass-card">
        <div class="section-header">
            📤 Image Input
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload chest X-ray image",
        type=["jpg", "png", "jpeg"],
        help="Supported formats: JPG, PNG, JPEG",
        on_change=reset_analysis
    )
    
    st.markdown('<div class="divider">or paste URL</div>', unsafe_allow_html=True)
    
    url_input = st.text_input(
        "Image URL",
        placeholder="https://example.com/xray.jpg",
        label_visibility="collapsed",
        on_change=reset_analysis
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Image Loading Logic
    if uploaded_file:
        st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
    elif url_input.strip():
        if not st.session_state.uploaded_image or not uploaded_file:
            with st.spinner("Fetching image..."):
                new_image = load_image_from_url(url_input)
                if new_image:
                    st.session_state.uploaded_image = new_image
                else:
                    st.markdown("""
                    <div class="alert-box alert-error">
                        <div>
                            <div class="alert-title">❌ Error</div>
                            <div class="alert-message">Invalid URL or unsupported image format.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
    
    # SETTINGS FOR ANALYSIS (Simple Toggle for Grad-CAM)
    if st.session_state.uploaded_image:
        with st.expander("⚙️ Analysis Settings", expanded=True):
            show_gradcam = st.toggle("Explain AI Decision (Grad-CAM)", value=True, help="Visualisasikan area yang menjadi fokus model.")

    # BUTTONS SECTION
    col_btn1, col_btn2 = st.columns(2, gap="medium")
    
    with col_btn1:
        analyze_clicked = st.button("🔬 Analyze", type="primary", use_container_width=True)
    
    with col_btn2:
        reset_clicked = st.button("↺ Reset", type="secondary", use_container_width=True)

    if reset_clicked:
        st.session_state.uploaded_image = None
        reset_analysis()
        st.rerun()

# --- RIGHT COLUMN (DISPLAY) ---
with col_display:
    if st.session_state.uploaded_image:
        st.markdown("""
        <div class="glass-card glass-card-glow">
            <div class="section-header">
                📋 Analysis Report
            </div>
        """, unsafe_allow_html=True)
        
        # Placeholder for image
        img_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        # Inference Logic
        if analyze_clicked and not st.session_state.analysis_done:
            with st.spinner("Processing neural network inference..."):
                try:
                    if demo_mode:
                        import numpy as np
                        mock_probs = np.random.dirichlet(np.ones(len(CLASS_NAMES)))
                        top_idx = np.argmax(mock_probs)
                        st.session_state.prediction = CLASS_NAMES[top_idx]
                        st.session_state.confidence = mock_probs[top_idx]
                        st.session_state.probs_history = mock_probs
                    else:
                        # 1. Predict
                        img_tensor = process_image(st.session_state.uploaded_image).to(device)
                        with torch.no_grad():
                            logits = model(img_tensor)
                            probs = F.softmax(logits, dim=1)
                        
                        top_prob, top_idx = torch.max(probs, 1)
                        st.session_state.prediction = CLASS_NAMES[top_idx.item()]
                        st.session_state.confidence = top_prob.item()
                        st.session_state.probs_history = probs.cpu().numpy().flatten()
                        
                        # 2. Grad-CAM (If enabled)
                        if show_gradcam and GRAD_CAM_AVAILABLE:
                            heatmap = generate_gradcam(model, st.session_state.uploaded_image, target_class_index=top_idx.item())
                            st.session_state.gradcam_image = heatmap
                    
                    st.session_state.analysis_done = True
                except Exception as e:
                    st.markdown(f"""
                    <div class="alert-box alert-error">
                        <div>
                            <div class="alert-title">❌ Analysis Error</div>
                            <div class="alert-message">{str(e)}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # RENDER IMAGE (Switch between raw and heatmap)
        with img_placeholder.container():
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            if st.session_state.analysis_done and st.session_state.gradcam_image is not None and show_gradcam:
                st.image(st.session_state.gradcam_image, caption="AI Attention Heatmap (Grad-CAM)", width=450)
            else:
                st.image(st.session_state.uploaded_image, caption="Input Radiograph", width=450)
            st.markdown('</div>', unsafe_allow_html=True)

        # Display Results
        if st.session_state.analysis_done and st.session_state.probs_history is not None:
            st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
            
            # METRICS
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.metric("Diagnosis", st.session_state.prediction)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.metric("Confidence", f"{st.session_state.confidence:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            pred = st.session_state.prediction
            
            # ALERTS
            if pred == "Normal":
                st.markdown(f"""
                <div class="alert-box alert-success">
                    <div>
                        <div class="alert-title">✓ NEGATIVE (Normal)</div>
                        <div class="alert-message">No pathological findings detected. Lung fields appear clear.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif pred == "Unknown":
                st.markdown(f"""
                <div class="alert-box alert-warning">
                    <div>
                        <div class="alert-title">⚠ INVALID INPUT</div>
                        <div class="alert-message">Image pattern is inconsistent with standard chest radiography.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-box alert-error">
                    <div>
                        <div class="alert-title">✕ POSITIVE ({pred.upper()})</div>
                        <div class="alert-message">AI Model detected patterns consistent with <strong>{pred}</strong>. Clinical correlation recommended.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # DISEASE INFO
            info = DISEASE_INFO.get(pred, {})
            with st.expander(f"ℹ️ Informasi Detail: {pred}", expanded=False):
                st.markdown(f"**Deskripsi:**\n{info.get('desc', '')}")
                st.markdown(f"**Gejala Umum:**\n{info.get('symptoms', '')}")

            # CHART
            st.markdown("""
            <div class="chart-container">
                <div class="section-header" style="border-bottom: none; margin-bottom: 0px; padding-bottom: 0;">
                    📊 Confidence Distribution
                </div>
            """, unsafe_allow_html=True)
            
            df_chart = pd.DataFrame({
                'Condition': CLASS_NAMES,
                'Probability': st.session_state.probs_history
            })
            
            base = alt.Chart(df_chart).encode(
                y=alt.Y('Condition:N', 
                        sort='-x', 
                        axis=alt.Axis(title=None, labelColor='#e2e8f0', labelFontSize=13, labelLimit=200, ticks=False, domain=False)),
                x=alt.X('Probability:Q', axis=None) 
            )

            bars = base.mark_bar(cornerRadiusEnd=4, height=30).encode(
                color=alt.condition(
                    alt.datum.Condition == pred,
                    alt.value('#3b82f6'), 
                    alt.value('#1e293b') 
                )
            )

            text = base.mark_text(
                align='left',
                baseline='middle',
                dx=5, 
                color='#94a3b8',
                fontWeight=600
            ).encode(
                text=alt.Text('Probability:Q', format='.1%')
            )

            final_chart = (bars + text).properties(height=alt.Step(50)).configure_view(strokeWidth=0)
            
            st.altair_chart(final_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Empty State
        st.markdown("""
        <div class="glass-card" style="min-height: 450px; display: flex; align-items: center; justify-content: center;">
            <div class="info-box">
                <div class="info-box-icon">📸</div>
                <div class="info-box-text">
                    <strong>Ready to Scan</strong><br>
                    Upload a chest X-ray image (JPG/PNG)<br>to begin the AI analysis.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div class="footer">
    <p>⚕️ <strong>Disclaimer:</strong> This tool is for educational and experimental purposes only.<br>
    It is not a medical device and should not be used for primary diagnosis.</p>
    <p style="margin-top: 8px; opacity: 0.5;">LungScan AI v1.5 • Powered by ConvNeXt-V2</p>
</div>
""", unsafe_allow_html=True)