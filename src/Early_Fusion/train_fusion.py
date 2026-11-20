import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Ensure Early_Fusion is in path
import sys
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from extractors import (
    extract_face_features,
    extract_voice_features,
    extract_fingerprint_features,
    fuse_features,
)

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "Models" / "Early_Fusion"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODELS_DIR / "early_fusion_model.joblib"

DATASET_ROOT = PROJECT_ROOT / "Dataset"
FACE_DIR = DATASET_ROOT / "YOLO_Experiment" / "Face_Classification" / "train"
VOICE_DIR = DATASET_ROOT / "Processed" / "Voice" / "Audio_Standardized"
FP_DIR = DATASET_ROOT / "Processed" / "Fingerprint" / "Fingerprint_CLAHE"

USERS = ["User_01", "User_02"]


def collect_modality_files():
    data = {}
    for user in USERS:
        user_faces = sorted(list((FACE_DIR / user).glob("*.png")) + list((FACE_DIR / user).glob("*.jpg")))
        user_voices = sorted(list((VOICE_DIR / user).glob("*.wav")) + list((VOICE_DIR / user).glob("*.mp3")))
        user_fps = sorted(list((FP_DIR / user).glob("*.bmp")) + list((FP_DIR / user).glob("*.BMP")) + list((FP_DIR / user).glob("*.png")))

        print(f"[{user}] Found {len(user_faces)} faces, {len(user_voices)} voice files, {len(user_fps)} fingerprints.")
        data[user] = {
            "faces": user_faces,
            "voices": user_voices,
            "fps": user_fps,
        }
    return data


def build_fused_dataset(max_samples_per_user=150):
    print("\n[1/4] Collecting files and extracting features per modality...")
    files_per_user = collect_modality_files()

    # Pre-extract features for speed
    features_cache = {}

    for user in USERS:
        print(f"Extracting features for {user}...")
        face_feats = [extract_face_features(f) for f in files_per_user[user]["faces"][:20]]
        voice_feats = [extract_voice_features(v) for v in files_per_user[user]["voices"][:20]]
        fp_feats = [extract_fingerprint_features(fp) for fp in files_per_user[user]["fps"][:10]]

        features_cache[user] = {
            "face": face_feats,
            "voice": voice_feats,
            "fp": fp_feats,
        }

    print("\n[2/4] Generating Early Fusion multimodal triplets [Face + Voice + Fingerprint]...")
    X = []
    y = []

    random.seed(42)
    for user in USERS:
        face_list = features_cache[user]["face"]
        voice_list = features_cache[user]["voice"]
        fp_list = features_cache[user]["fp"]

        count = 0
        while count < max_samples_per_user:
            f = random.choice(face_list)
            v = random.choice(voice_list)
            fp = random.choice(fp_list)

            # Add subtle jitter to increase generalizability
            noise_v = v + np.random.normal(0, 0.005, size=v.shape).astype(np.float32)

            fused = fuse_features(f, noise_v, fp)
            X.append(fused)
            y.append(user)
            count += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"Total Fused Dataset Shape: X = {X.shape}, y = {y.shape}")
    return X, y


def train_early_fusion_model():
    X, y = build_fused_dataset(max_samples_per_user=120)

    print("\n[3/4] Splitting and Normalizing features...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[4/4] Training SVM Early Fusion Classifier (RBF kernel with probabilities)...")
    clf = SVC(kernel="rbf", C=1.5, probability=True, random_state=42)
    clf.fit(X_train_scaled, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    y_pred = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\n==========================================")
    print(f" Train Accuracy : {train_acc:.2%}")
    print(f" Test Accuracy  : {test_acc:.2%}")
    print(f"==========================================")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    bundle = {
        "model": clf,
        "scaler": scaler,
        "classes": list(clf.classes_),
        "feature_dim": 4652,
        "modality_dims": {
            "face": 256,
            "voice": 30,
            "fingerprint": 4366,
        },
    }

    joblib.dump(bundle, MODEL_SAVE_PATH)
    print(f"\n[SUCCESS] Early Fusion Model bundle saved successfully to:\n{MODEL_SAVE_PATH}\n")
    return bundle


def predict_early_fusion(face_input, voice_input, fingerprint_input, model_bundle=None):
    """
    Inference helper: takes raw inputs for face, voice, and fingerprint,
    extracts features, fuses them, and predicts the user with confidence scores.
    """
    if model_bundle is None:
        if not MODEL_SAVE_PATH.is_file():
            raise FileNotFoundError(f"Model not found at {MODEL_SAVE_PATH}. Run training first.")
        model_bundle = joblib.load(MODEL_SAVE_PATH)

    face_vec = extract_face_features(face_input)
    voice_vec = extract_voice_features(voice_input)
    fp_vec = extract_fingerprint_features(fingerprint_input)

    fused_vec = fuse_features(face_vec, voice_vec, fp_vec).reshape(1, -1)
    scaled_vec = model_bundle["scaler"].transform(fused_vec)

    pred_user = model_bundle["model"].predict(scaled_vec)[0]
    probs = model_bundle["model"].predict_proba(scaled_vec)[0]
    class_probs = dict(zip(model_bundle["classes"], [float(p) for p in probs]))
    confidence = float(max(probs))

    return {
        "predicted_user": str(pred_user),
        "confidence": confidence,
        "probabilities": class_probs,
        "feature_dimensions": {
            "face": len(face_vec),
            "voice": len(voice_vec),
            "fingerprint": len(fp_vec),
            "fused_total": fused_vec.shape[1],
        },
    }


if __name__ == "__main__":
    train_early_fusion_model()
