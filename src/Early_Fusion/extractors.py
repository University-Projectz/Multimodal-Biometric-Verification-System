import os
import sys
import tempfile
from pathlib import Path
from math import gcd

import cv2
import librosa
import numpy as np
import soundfile as sf
from PIL import Image
from scipy.signal import resample_poly
from skimage.feature import hog, local_binary_pattern
from ultralytics import YOLO

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACE_YOLO_PATH = PROJECT_ROOT / "Models" / "Face" / "YOLO_Experiment" / "best.pt"

# Cache ultralytics configuration
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".cache" / "yolo"))

_FACE_MODEL = None


def get_face_model():
    global _FACE_MODEL
    if _FACE_MODEL is None:
        if not FACE_YOLO_PATH.is_file():
            raise FileNotFoundError(f"YOLO Face model not found at {FACE_YOLO_PATH}")
        _FACE_MODEL = YOLO(str(FACE_YOLO_PATH))
    return _FACE_MODEL


# ============================================================
# 1. FACE FEATURE EXTRACTION (256-D Deep Embedding)
# ============================================================

def extract_face_features(image_input):
    """
    Extracts a 256-dimensional deep feature embedding from a face image.
    Accepts: file path (str or Path), PIL Image, or bytes.
    """
    model = get_face_model()
    temp_path = None

    try:
        if isinstance(image_input, (str, Path)):
            img_path = str(image_input)
        elif isinstance(image_input, bytes):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(image_input)
                temp_path = Path(f.name)
            img_path = str(temp_path)
        elif isinstance(image_input, Image.Image):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                image_input.save(f.name)
                temp_path = Path(f.name)
            img_path = str(temp_path)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        embeddings = model.embed(source=img_path, verbose=False)
        if not embeddings or len(embeddings) == 0:
            raise RuntimeError("YOLO model failed to extract face embedding.")

        embedding = embeddings[0].detach().cpu().numpy().flatten().astype(np.float32)

        if embedding.shape[0] != 256:
            raise ValueError(f"Expected 256 face features, got {embedding.shape[0]}")

        return embedding

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


# ============================================================
# 2. VOICE FEATURE EXTRACTION (30-D Acoustic Vector)
# ============================================================

def extract_voice_features(audio_input):
    """
    Extracts 30 acoustic features (13 MFCC Mean + 13 MFCC Std + 2 ZCR + 2 RMS).
    Accepts: file path (str or Path) or bytes.
    """
    temp_path = None
    try:
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
        elif isinstance(audio_input, bytes):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_input)
                temp_path = Path(f.name)
            audio_path = temp_path
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        signal, original_sample_rate = sf.read(str(audio_path))

        if signal.size == 0:
            raise ValueError("Audio input is empty.")

        # Convert stereo to mono
        if signal.ndim == 2:
            signal = np.mean(signal, axis=1)

        # Resample to 16 kHz
        target_sr = 16000
        if original_sample_rate != target_sr:
            c = gcd(int(original_sample_rate), target_sr)
            signal = resample_poly(signal, target_sr // c, int(original_sample_rate) // c)

        # Soft peak normalization to prevent volume bias
        peak = np.max(np.abs(signal))
        if peak > 0.38:
            signal = signal * (0.32 / peak)

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=target_sr,
            n_mfcc=13,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window="hann"
        )
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=400, hop_length=160)
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))

        # RMS
        rms = librosa.feature.rms(y=signal, frame_length=400, hop_length=160)
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        features = []
        for i in range(13):
            features.extend([float(mfcc_mean[i]), float(mfcc_std[i])])
        features.extend([zcr_mean, zcr_std, rms_mean, rms_std])

        return np.array(features, dtype=np.float32)

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


# ============================================================
# 3. FINGERPRINT FEATURE EXTRACTION (4366-D LBP + HOG)
# ============================================================

def extract_fingerprint_features(fingerprint_input):
    """
    Extracts 4366 features (10 LBP histogram + 4356 HOG descriptors).
    Accepts: file path (str or Path), PIL Image, or bytes.
    """
    if isinstance(fingerprint_input, (str, Path)):
        img = cv2.imread(str(fingerprint_input), cv2.IMREAD_GRAYSCALE)
    elif isinstance(fingerprint_input, bytes):
        arr = np.frombuffer(fingerprint_input, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    elif isinstance(fingerprint_input, Image.Image):
        img = np.array(fingerprint_input.convert("L"))
    else:
        raise TypeError(f"Unsupported fingerprint input type: {type(fingerprint_input)}")

    if img is None:
        raise ValueError("Failed to load or decode fingerprint image.")

    # Standardize size (matches project CLAHE dimension: width 96, height 103)
    if img.shape != (103, 96):
        img = cv2.resize(img, (96, 103))

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)

    # A. LBP (10 bins uniform)
    lbp = local_binary_pattern(enhanced_img, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # B. HOG (4356 features)
    hog_feats = hog(
        enhanced_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    ).astype(np.float32)

    return np.hstack([lbp_hist, hog_feats])


# ============================================================
# 4. EARLY FUSION CONCATENATION (4652-D Unified Representation)
# ============================================================

def fuse_features(face_vector, voice_vector, fingerprint_vector):
    """
    Concatenates feature representations from Face, Voice, and Fingerprint.
    Returns: fused 1D numpy array of shape (4652,)
    """
    if face_vector.ndim != 1 or face_vector.shape[0] != 256:
        raise ValueError(f"Face vector must have shape (256,), got {face_vector.shape}")
    if voice_vector.ndim != 1 or voice_vector.shape[0] != 30:
        raise ValueError(f"Voice vector must have shape (30,), got {voice_vector.shape}")
    if fingerprint_vector.ndim != 1 or fingerprint_vector.shape[0] != 4366:
        raise ValueError(f"Fingerprint vector must have shape (4366,), got {fingerprint_vector.shape}")

    fused = np.concatenate([face_vector, voice_vector, fingerprint_vector]).astype(np.float32)
    return fused
