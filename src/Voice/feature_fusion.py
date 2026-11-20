
from pathlib import Path
from itertools import product
from math import gcd
import tempfile

import joblib
import librosa
import numpy as np
import soundfile as sf
import torch

from scipy.signal import resample_poly
from ultralytics import YOLO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACE_MODEL_PATH = (
    PROJECT_ROOT
    / "Models"
    / "Face"
    / "YOLO_Experiment"
    / "best.pt"
)

FACE_DATASET_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "YOLO_Experiment"
    / "Face_Classification"
    / "train"
)

VOICE_DATASET_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "Processed"
    / "Voice"
    / "Audio_Standardized"
)

FUSION_MODEL_DIR = PROJECT_ROOT / "Models" / "Fusion"

FUSION_MODEL_PATH = (
    FUSION_MODEL_DIR / "Feature_Fusion_SVM.joblib"
)

USERS = ["User_01", "User_02"]

SAMPLE_RATE = 16000


# ============================================================
# 1. FACE FEATURE EXTRACTOR
# ============================================================

class FaceFeatureExtractor:

    def __init__(self, model_path):

        self.yolo = YOLO(str(model_path))
        self.embedding = None

        classifier = self.yolo.model.model[-1]

        if not hasattr(classifier, "linear"):
            raise RuntimeError(
                "Could not locate YOLO classification linear layer."
            )

        self.hook = classifier.linear.register_forward_pre_hook(
            self.capture_embedding
        )

    def capture_embedding(self, module, inputs):

        self.embedding = inputs[0].detach().cpu().numpy().flatten()

    def extract(self, image_path):

        self.embedding = None

        self.yolo.predict(
            source=str(image_path),
            verbose=False
        )

        if self.embedding is None:
            raise RuntimeError(
                f"Failed to extract face features: {image_path}"
            )

        return self.embedding.copy()

    def close(self):
        self.hook.remove()


# ============================================================
# 2. VOICE FEATURE EXTRACTION
# ============================================================

def extract_voice_features(audio_path):

    signal, original_rate = sf.read(str(audio_path))

    # Convert stereo to mono
    if signal.ndim == 2:
        signal = np.mean(signal, axis=1)

    # Resample to 16 kHz
    if original_rate != SAMPLE_RATE:

        divisor = gcd(original_rate, SAMPLE_RATE)

        signal = resample_poly(
            signal,
            SAMPLE_RATE // divisor,
            original_rate // divisor
        )

    signal = signal.astype(np.float32)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=13,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window="hann"
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(
        y=signal,
        frame_length=400,
        hop_length=160
    )

    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # RMS
    rms = librosa.feature.rms(
        y=signal,
        frame_length=400,
        hop_length=160
    )

    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # Preserve the original Notebook 02 feature order
    features = []

    for index in range(13):
        features.append(mfcc_mean[index])
        features.append(mfcc_std[index])

    features.extend([
        zcr_mean,
        zcr_std,
        rms_mean,
        rms_std
    ])

    return np.asarray(features, dtype=np.float32)


# ============================================================
# 3. EARLY FEATURE FUSION
# ============================================================

def fuse_features(face_features, voice_features):

    return np.concatenate([
        face_features,
        voice_features
    ])


# ============================================================
# 4. DATASET COLLECTION
# ============================================================

def get_files(directory, extensions):

    return sorted([
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
    ])


def extract_user_features(face_extractor, user_id):

    face_dir = FACE_DATASET_PATH / user_id
    voice_dir = VOICE_DATASET_PATH / user_id

    face_files = get_files(
        face_dir,
        {".jpg", ".jpeg", ".png", ".bmp"}
    )

    voice_files = get_files(
        voice_dir,
        {".wav"}
    )

    if len(face_files) < 2 or len(voice_files) < 2:
        raise ValueError(
            f"Not enough face or voice samples for {user_id}."
        )

    # Split each modality independently BEFORE pairing.
    # This prevents the same source files from appearing
    # in both training and testing pairs.

    face_train, face_test = train_test_split(
        face_files,
        test_size=0.20,
        random_state=42
    )

    voice_train, voice_test = train_test_split(
        voice_files,
        test_size=0.20,
        random_state=42
    )

    print(f"\nExtracting features for {user_id}...")

    face_features = {
        "train": [
            face_extractor.extract(path)
            for path in face_train
        ],
        "test": [
            face_extractor.extract(path)
            for path in face_test
        ]
    }

    voice_features = {
        "train": [
            extract_voice_features(path)
            for path in voice_train
        ],
        "test": [
            extract_voice_features(path)
            for path in voice_test
        ]
    }

    return face_features, voice_features


# ============================================================
# 5. BUILD FUSED TRAINING AND TEST DATA
# ============================================================

def build_fused_dataset(face_extractor):

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    for user_id in USERS:

        face_data, voice_data = extract_user_features(
            face_extractor,
            user_id
        )

        # Training pairs
        for face_vector, voice_vector in product(
            face_data["train"],
            voice_data["train"]
        ):

            fused = fuse_features(
                face_vector,
                voice_vector
            )

            X_train.append(fused)
            y_train.append(user_id)

        # Testing pairs
        for face_vector, voice_vector in product(
            face_data["test"],
            voice_data["test"]
        ):

            fused = fuse_features(
                face_vector,
                voice_vector
            )

            X_test.append(fused)
            y_test.append(user_id)

    return (
        np.asarray(X_train),
        np.asarray(y_train),
        np.asarray(X_test),
        np.asarray(y_test)
    )


# ============================================================
# 6. TRAIN THE FUSION MODEL
# ============================================================

def train_fusion_model():

    print("\n========== FEATURE FUSION TRAINING ==========")

    face_extractor = FaceFeatureExtractor(
        FACE_MODEL_PATH
    )

    try:
        X_train, y_train, X_test, y_test = (
            build_fused_dataset(face_extractor)
        )
    finally:
        face_extractor.close()

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Fused feature dimensions:", X_train.shape[1])

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    fusion_model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )

    fusion_model.fit(
        X_train_scaled,
        y_train
    )

    predictions = fusion_model.predict(X_test_scaled)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n========== EVALUATION ==========")
    print("Accuracy:", accuracy)
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    FUSION_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    bundle = {
        "model": fusion_model,
        "scaler": scaler,
        "face_model_path": str(FACE_MODEL_PATH),
        "face_feature_size": len(X_train[0]) - 30,
        "voice_feature_size": 30,
        "users": USERS
    }

    joblib.dump(
        bundle,
        FUSION_MODEL_PATH
    )

    print("\nFusion model saved successfully:")
    print(FUSION_MODEL_PATH)

    print("==============================================")


# ============================================================
# 7. LOAD AND USE THE FUSION MODEL
# ============================================================

def predict_fusion(face_image_path, audio_path):

    bundle = joblib.load(FUSION_MODEL_PATH)

    face_extractor = FaceFeatureExtractor(
        bundle["face_model_path"]
    )

    try:
        face_features = face_extractor.extract(
            face_image_path
        )
    finally:
        face_extractor.close()

    voice_features = extract_voice_features(
        audio_path
    )

    if len(face_features) != bundle["face_feature_size"]:
        raise ValueError("Unexpected face feature dimensions.")

    if len(voice_features) != bundle["voice_feature_size"]:
        raise ValueError("Unexpected voice feature dimensions.")

    fused_features = fuse_features(
        face_features,
        voice_features
    )

    fused_features = fused_features.reshape(1, -1)

    scaled_features = bundle["scaler"].transform(
        fused_features
    )

    prediction = bundle["model"].predict(
        scaled_features
    )[0]

    return {
        "predicted_user": str(prediction),
        "model": "Early Feature Fusion SVM",
        "features_count": len(fused_features[0])
    }


# ============================================================
# 8. RUN TRAINING
# ============================================================

if __name__ == "__main__":

    train_fusion_model()
