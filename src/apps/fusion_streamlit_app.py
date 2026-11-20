import os
import sys
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Setup Paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".cache" / "yolo"))

from extractors import (
    extract_face_features,
    extract_voice_features,
    extract_fingerprint_features,
    fuse_features,
)

MODEL_PATH = PROJECT_ROOT / "Models" / "Early_Fusion" / "early_fusion_model.joblib"
SAMPLE_DIR_FACE = PROJECT_ROOT / "Dataset" / "YOLO_Experiment" / "Face_Classification" / "train"
SAMPLE_DIR_VOICE = PROJECT_ROOT / "Dataset" / "Processed" / "Voice" / "Audio_Standardized"
SAMPLE_DIR_FP = PROJECT_ROOT / "Dataset" / "Processed" / "Fingerprint" / "Fingerprint_CLAHE"

# Page Configuration
st.set_page_config(
    page_title="Multimodal Biometric Early Fusion",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38BDF8;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #334155;
        text-align: center;
    }
    .success-box {
        background-color: rgba(34, 197, 94, 0.15);
        border: 2px solid #22C55E;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
    }
    .declined-box {
        background-color: rgba(239, 68, 68, 0.15);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_fusion_bundle():
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


def main():
    st.markdown('<div class="main-header">🛡️ Multimodal Biometric Early Fusion</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Unified Feature-Level Fusion combining <b>Face (256D)</b> + <b>Voice (30D)</b> + <b>Fingerprint (4366D)</b> &rarr; <b>4652D Fused Vector</b></div>',
        unsafe_allow_html=True
    )

    bundle = load_fusion_bundle()

    if bundle is None:
        st.error(f"❌ Early Fusion Model not found at `{MODEL_PATH}`. Please run `train_fusion.py` first.")
        st.stop()

    # Sidebar: System Info & Quick Demo Loader
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        st.success("✅ Early Fusion SVM Loaded")
        st.markdown(f"""
        - **Classifier:** Support Vector Machine (RBF)
        - **Total Modalities:** 3 (Face, Voice, Fingerprint)
        - **Fused Dimension:** `{bundle['feature_dim']}` features
        - **Authorized Classes:** `{', '.join(bundle['classes'])}`
        """)

        st.markdown("---")
        st.header("🧪 Quick Demo Loader")
        st.write("Load a pre-configured sample triplet directly from the dataset:")

        col_demo1, col_demo2 = st.columns(2)
        load_demo_user1 = col_demo1.button("Load User_01", use_container_width=True)
        load_demo_user2 = col_demo2.button("Load User_02", use_container_width=True)

    # Initialize session state for inputs
    if "face_bytes" not in st.session_state:
        st.session_state.face_bytes = None
        st.session_state.voice_bytes = None
        st.session_state.fp_bytes = None
        st.session_state.source_info = "Manual Upload"

    if load_demo_user1:
        face_sample = list((SAMPLE_DIR_FACE / "User_01").glob("*.png"))[0]
        voice_sample = list((SAMPLE_DIR_VOICE / "User_01").glob("*.wav"))[0]
        fp_sample = list((SAMPLE_DIR_FP / "User_01").glob("*.BMP"))[0]

        st.session_state.face_bytes = face_sample.read_bytes()
        st.session_state.voice_bytes = voice_sample.read_bytes()
        st.session_state.fp_bytes = fp_sample.read_bytes()
        st.session_state.source_info = "Loaded User_01 Demo Dataset"
        st.rerun()

    if load_demo_user2:
        face_sample = list((SAMPLE_DIR_FACE / "User_02").glob("*.png"))[0]
        voice_sample = list((SAMPLE_DIR_VOICE / "User_02").glob("*.wav"))[0]
        fp_sample = list((SAMPLE_DIR_FP / "User_02").glob("*.BMP"))[0]

        st.session_state.face_bytes = face_sample.read_bytes()
        st.session_state.voice_bytes = voice_sample.read_bytes()
        st.session_state.fp_bytes = fp_sample.read_bytes()
        st.session_state.source_info = "Loaded User_02 Demo Dataset"
        st.rerun()

    # Three Upload Columns
    st.subheader("📤 1. Upload Triplet Modalities")
    col_face, col_voice, col_fp = st.columns(3)

    # Column 1: Face
    with col_face:
        st.markdown("### 📸 Face Image")
        uploaded_face = st.file_uploader(
            "Upload Face (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            key="uploader_face"
        )
        if uploaded_face is not None:
            st.session_state.face_bytes = uploaded_face.read()
            st.session_state.source_info = "Manual Upload"

        if st.session_state.face_bytes:
            st.image(st.session_state.face_bytes, caption="Face Preview", use_container_width=True)
            st.caption("🔍 Extractor: YOLO11 Deep Embedding (256D)")

    # Column 2: Voice
    with col_voice:
        st.markdown("### 🎙️ Voice Sample")
        uploaded_voice = st.file_uploader(
            "Upload Audio (WAV, MP3)",
            type=["wav", "mp3"],
            key="uploader_voice"
        )
        if uploaded_voice is not None:
            st.session_state.voice_bytes = uploaded_voice.read()
            st.session_state.source_info = "Manual Upload"

        if st.session_state.voice_bytes:
            st.audio(st.session_state.voice_bytes)
            st.caption("🔍 Extractor: 13 MFCC + ZCR + RMS (30D)")

    # Column 3: Fingerprint
    with col_fp:
        st.markdown("### 🖐️ Fingerprint")
        uploaded_fp = st.file_uploader(
            "Upload Fingerprint (BMP, PNG, JPG)",
            type=["bmp", "png", "jpg"],
            key="uploader_fp"
        )
        if uploaded_fp is not None:
            st.session_state.fp_bytes = uploaded_fp.read()
            st.session_state.source_info = "Manual Upload"

        if st.session_state.fp_bytes:
            st.image(st.session_state.fp_bytes, caption="Fingerprint Preview", use_container_width=True)
            st.caption("🔍 Extractor: LBP (10D) + HOG (4356D) = 4366D")

    st.markdown("---")

    # Verification Trigger
    has_all_inputs = (
        st.session_state.face_bytes is not None
        and st.session_state.voice_bytes is not None
        and st.session_state.fp_bytes is not None
    )

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        verify_btn = st.button(
            "⚡ Run Early Fusion Verification",
            type="primary",
            disabled=not has_all_inputs,
            use_container_width=True
        )

    with col_info:
        if not has_all_inputs:
            st.warning("⚠️ Please provide all 3 modalities (Face + Voice + Fingerprint) to perform Early Fusion.")
        else:
            st.info(f"Ready! Source: `{st.session_state.source_info}`")

    if verify_btn and has_all_inputs:
        with st.spinner("Extracting features from Face, Voice, and Fingerprint..."):
            try:
                # 1. Feature Extraction
                f_vec = extract_face_features(st.session_state.face_bytes)
                v_vec = extract_voice_features(st.session_state.voice_bytes)
                fp_vec = extract_fingerprint_features(st.session_state.fp_bytes)

                # 2. Early Feature Concatenation
                fused_vec = fuse_features(f_vec, v_vec, fp_vec).reshape(1, -1)

                # 3. Normalization & Prediction
                scaled_vec = bundle["scaler"].transform(fused_vec)
                predicted_user = bundle["model"].predict(scaled_vec)[0]
                probabilities = bundle["model"].predict_proba(scaled_vec)[0]
                classes = bundle["classes"]

                prob_dict = dict(zip(classes, [float(p) for p in probabilities]))
                confidence = float(max(probabilities))

                # Display Results
                st.subheader("🎯 2. Early Fusion Results")

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Face Embedding", f"{len(f_vec)} dims")
                col_m2.metric("Voice Acoustic", f"{len(v_vec)} dims")
                col_m3.metric("Fingerprint Texture", f"{len(fp_vec)} dims")
                col_m4.metric("Fused Vector", f"{fused_vec.shape[1]} dims")

                # Decision Card
                if confidence >= 0.70:
                    st.markdown(f"""
                    <div class="success-box">
                        <h2 style="color: #22C55E; margin: 0;">ACCESS GRANTED ✅</h2>
                        <h3 style="color: white; margin-top: 10px;">Authenticated Identity: <b>{predicted_user}</b></h3>
                        <p style="color: #86EFAC; font-size: 1.1rem; margin: 5px 0;">Model Confidence: <b>{confidence:.2%}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="declined-box">
                        <h2 style="color: #EF4444; margin: 0;">ACCESS DECLINED ❌</h2>
                        <p style="color: #FCA5A5; font-size: 1.1rem; margin: 5px 0;">Low Confidence: <b>{confidence:.2%}</b> below security threshold.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Probabilities Breakdown
                st.markdown("### 📊 Class Probability Breakdown")
                prob_df = pd.DataFrame({
                    "Identity": list(prob_dict.keys()),
                    "Probability": [p * 100 for p in prob_dict.values()]
                })
                st.bar_chart(prob_df.set_index("Identity"), use_container_width=True)

                with st.expander("🔬 View Fused Representation Details"):
                    st.write(f"**Vector Shape:** `{fused_vec.shape}`")
                    st.write(f"**Mean Value:** `{float(np.mean(fused_vec)):.4f}` | **Std Dev:** `{float(np.std(fused_vec)):.4f}`")
                    st.line_chart(fused_vec.flatten()[:500], height=200)
                    st.caption("Preview of the first 500 features of the fused multimodal vector.")

            except Exception as e:
                st.error(f"❌ An error occurred during feature fusion: {e}")


if __name__ == "__main__":
    main()
