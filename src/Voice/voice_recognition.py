
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from math import gcd
from scipy.signal import resample_poly


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SVM_MODEL_PATH = (
    PROJECT_ROOT
    / "Models"
    / "Voice"
    / "SVM_Voice_Recognition.joblib"
)

TARGET_SAMPLE_RATE = 16000

N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
WINDOW = "hann"

ZCR_FRAME_LENGTH = 400
RMS_FRAME_LENGTH = 400


# ============================================================
# 2. EXPECTED FEATURE NAMES
# ============================================================

EXPECTED_FEATURE_COLUMNS = []

for i in range(N_MFCC):
    EXPECTED_FEATURE_COLUMNS.extend([
        f"MFCC_{i}_Mean",
        f"MFCC_{i}_Std"
    ])

EXPECTED_FEATURE_COLUMNS.extend([
    "ZCR_Mean",
    "ZCR_Std",
    "RMS_Mean",
    "RMS_Std"
])


# ============================================================
# 3. LOAD SVM MODEL BUNDLE
# ============================================================

_SVM_BUNDLE = None


def load_svm_model(force_reload=False):

    global _SVM_BUNDLE

    if _SVM_BUNDLE is not None and not force_reload:
        return _SVM_BUNDLE

    if not SVM_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"SVM model not found: {SVM_MODEL_PATH}"
        )

    bundle = joblib.load(SVM_MODEL_PATH)

    required_keys = {
        "model",
        "scaler",
        "feature_columns"
    }

    if not isinstance(bundle, dict):
        raise TypeError("SVM model bundle must be a dictionary.")

    if not required_keys.issubset(bundle.keys()):
        raise KeyError("SVM bundle is missing required keys.")

    feature_columns = list(bundle["feature_columns"])

    if len(feature_columns) != 30:
        raise ValueError(
            f"Expected 30 features, found {len(feature_columns)}."
        )

    if feature_columns != EXPECTED_FEATURE_COLUMNS:
        raise ValueError(
            "Feature names or order do not match the expected "
            "Voice feature extraction pipeline."
        )

    if not hasattr(bundle["scaler"], "transform"):
        raise TypeError("Invalid StandardScaler in SVM bundle.")

    if not hasattr(bundle["model"], "predict"):
        raise TypeError("Invalid SVM model in bundle.")

    _SVM_BUNDLE = bundle

    return _SVM_BUNDLE


# ============================================================
# 4. AUDIO PREPROCESSING
# ============================================================

def preprocess_audio(audio_path):

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    audio_signal, original_sample_rate = sf.read(audio_path)

    if audio_signal.size == 0:
        raise ValueError("Audio file is empty.")

    if original_sample_rate <= 0:
        raise ValueError("Invalid sample rate.")

    # Convert stereo to mono
    if audio_signal.ndim == 2:
        audio_signal = np.mean(audio_signal, axis=1)

    if audio_signal.ndim != 1:
        raise ValueError("Audio must be mono after conversion.")

    if not np.all(np.isfinite(audio_signal)):
        raise ValueError("Audio contains invalid samples.")

    # Resample to 16 kHz
    if original_sample_rate != TARGET_SAMPLE_RATE:

        common_divisor = gcd(
            int(original_sample_rate),
            TARGET_SAMPLE_RATE
        )

        up = TARGET_SAMPLE_RATE // common_divisor
        down = int(original_sample_rate) // common_divisor

        audio_signal = resample_poly(
            audio_signal,
            up,
            down
        )

    return audio_signal, TARGET_SAMPLE_RATE


# ============================================================
# 5. VOICE FEATURE EXTRACTION
# ============================================================

def extract_voice_features(audio_signal, sample_rate):

    if audio_signal.ndim != 1:
        raise ValueError("Expected a mono audio signal.")

    if audio_signal.size == 0:
        raise ValueError("Cannot extract features from empty audio.")

    # --------------------------------------------------------
    # A. MFCC
    # --------------------------------------------------------

    mfcc_features = librosa.feature.mfcc(
        y=audio_signal,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW
    )

    mfcc_mean = np.mean(mfcc_features, axis=1)
    mfcc_std = np.std(mfcc_features, axis=1)

    # --------------------------------------------------------
    # B. ZERO CROSSING RATE
    # --------------------------------------------------------

    zcr_features = librosa.feature.zero_crossing_rate(
        y=audio_signal,
        frame_length=ZCR_FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )

    zcr_mean = np.mean(zcr_features)
    zcr_std = np.std(zcr_features)

    # --------------------------------------------------------
    # C. ROOT MEAN SQUARE
    # --------------------------------------------------------

    rms_features = librosa.feature.rms(
        y=audio_signal,
        frame_length=RMS_FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )

    rms_mean = np.mean(rms_features)
    rms_std = np.std(rms_features)

    # --------------------------------------------------------
    # D. BUILD FEATURE DICTIONARY
    # --------------------------------------------------------

    features = {}

    # Preserve the training feature order
    for i in range(N_MFCC):

        features[f"MFCC_{i}_Mean"] = float(mfcc_mean[i])
        features[f"MFCC_{i}_Std"] = float(mfcc_std[i])

    features["ZCR_Mean"] = float(zcr_mean)
    features["ZCR_Std"] = float(zcr_std)

    features["RMS_Mean"] = float(rms_mean)
    features["RMS_Std"] = float(rms_std)

    if len(features) != 30:
        raise ValueError(
            f"Expected 30 features, extracted {len(features)}."
        )

    if not np.all(np.isfinite(list(features.values()))):
        raise ValueError("Extracted features contain invalid values.")

    return features


# ============================================================
# 6. PREPARE FEATURES FOR SVM
# ============================================================

def prepare_features(features, feature_columns):

    # Convert dictionary to DataFrame
    # The columns must follow the exact training order.

    features_df = pd.DataFrame(
        [features],
        columns=feature_columns
    )

    if features_df.shape != (1, 30):
        raise ValueError(
            f"Unexpected feature shape: {features_df.shape}"
        )

    # Apply the previously fitted StandardScaler
    # Do NOT use fit_transform on new audio.

    svm_scaler = load_svm_model()["scaler"]

    scaled_features = svm_scaler.transform(features_df)

    return scaled_features


# ============================================================
# 7. VOICE PREDICTION
# ============================================================

def predict_voice(audio_path):

    # Load saved SVM bundle
    bundle = load_svm_model()

    svm_model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    # Step 1: Audio preprocessing
    audio_signal, sample_rate = preprocess_audio(audio_path)

    # Step 2: Feature extraction
    features = extract_voice_features(
        audio_signal,
        sample_rate
    )

    # Step 3: Feature preparation and scaling
    scaled_features = prepare_features(
        features,
        feature_columns
    )

    # Step 4: SVM prediction
    prediction = svm_model.predict(scaled_features)[0]

    return {
        "predicted_user": str(prediction),
        "model": "SVM",
        "features_count": len(features)
    }


# ============================================================
# 8. TESTING
# ============================================================

if __name__ == "__main__":

    test_audio = (
        PROJECT_ROOT
        / "Dataset"
        / "Raw"
        / "Voice"
        / "User_01"
        / "User_01_020.wav"
    )

    try:

        result = predict_voice(test_audio)

        print("\n========== VOICE RECOGNITION ==========")
        print(f"Model: {result['model']}")
        print(f"Predicted User: {result['predicted_user']}")
        print(f"Features Extracted: {result['features_count']}")
        print("=======================================\n")

    except Exception as error:

        print(f"\nVoice Recognition Error: {error}\n")
