"""
command_recognition.py
======================
GMM + HMM based isolated-word command recognition.
Commands: "open" (Admin only) | "status" (All verified users)

Usage (training):
    python src/Command_Recognition/command_recognition.py --train

Usage (inference from Python):
    from src.Command_Recognition.command_recognition import predict_command, get_allowed_commands
    result = predict_command("path/to/audio.wav")
    # result = {"command": "open", "confidence": 0.87, "model": "GMM"}
"""

import os
import sys
import argparse
import warnings
import numpy as np
import librosa
import joblib
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR      = r"E:\Multimodal Biometric Verification System"
COMMANDS_DIR  = os.path.join(BASE_DIR, "Dataset", "Raw", "Command Recogntion")
MODELS_DIR    = os.path.join(BASE_DIR, "Models", "Command_Recognition")
os.makedirs(MODELS_DIR, exist_ok=True)
AUDIO_EXTENSIONS = (".wav",)

# Allow librosa/audioread to decode AAC with the environment-local FFmpeg.
FFMPEG_BIN = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
if os.path.isdir(FFMPEG_BIN):
    os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

# ─── Commands & Permissions ───────────────────────────────────────────────────
COMMANDS = ["open", "status"]

USER_PERMISSIONS = {
    "User_01": ["open", "status"],   # Admin → all commands
    "User_02": ["status"],           # Regular → status only
}

# ─── Feature Extraction Settings ─────────────────────────────────────────────
SAMPLE_RATE  = 16000
N_MFCC       = 13
HOP_LENGTH   = 160    # 10ms shift  @ 16kHz
WIN_LENGTH   = 400    # 25ms window @ 16kHz

# ─── GMM Settings ────────────────────────────────────────────────────────────
GMM_COMPONENTS   = 4
GMM_COV_TYPE     = "diag"
GMM_MAX_ITER     = 200
GMM_N_INIT       = 3

# ─── HMM Settings ────────────────────────────────────────────────────────────
HMM_N_STATES     = 5
HMM_N_MIX        = 2    # mixture components per state
HMM_COV_TYPE     = "diag"
HMM_N_ITER       = 100

# ─── Confidence threshold ─────────────────────────────────────────────────────
LOG_LIKELIHOOD_MARGIN = 10   # min gap between best and second-best score


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_mfcc_sequence(audio_path: str = None,
                           audio_array: np.ndarray = None,
                           sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract frame-level MFCCs (shape: [n_frames, N_MFCC*3]).
    Returns MFCC + Delta + Delta-Delta concatenated (39D per frame).
    """
    if audio_path is not None:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    elif audio_array is not None:
        y = audio_array.astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
    else:
        raise ValueError("Provide either audio_path or audio_array")

    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=20)

    if len(y) < WIN_LENGTH:
        # Pad if too short
        y = np.pad(y, (0, WIN_LENGTH - len(y)))

    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                    hop_length=HOP_LENGTH, n_fft=WIN_LENGTH)
    delta   = librosa.feature.delta(mfcc)
    delta2  = librosa.feature.delta(mfcc, order=2)

    # shape: (N_MFCC*3, n_frames) → transpose to (n_frames, N_MFCC*3)
    features = np.vstack([mfcc, delta, delta2]).T
    return features.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  GMM Training & Prediction
# ═══════════════════════════════════════════════════════════════════════════════

def train_gmm_models(audio_dir: str = COMMANDS_DIR,
                     save_dir: str = MODELS_DIR,
                     n_components: int = GMM_COMPONENTS) -> dict:
    """
    Train one GMM per command word.
    Returns dict of {command: GaussianMixture}
    """
    gmm_models = {}
    print("\n[GMM Training]")

    for cmd in COMMANDS:
        cmd_path = os.path.join(audio_dir, cmd)
        wav_files = [f for f in os.listdir(cmd_path) if f.lower().endswith(AUDIO_EXTENSIONS)]

        if not wav_files:
            raise FileNotFoundError(f"No supported audio files found in {cmd_path}")

        print(f"  [{cmd}] Loading {len(wav_files)} files ...")
        all_frames = []

        for wf in wav_files:
            fpath = os.path.join(cmd_path, wf)
            try:
                feats = extract_mfcc_sequence(audio_path=fpath)
                all_frames.append(feats)
            except Exception as e:
                print(f"    ⚠ Skipped {wf}: {e}")

        if not all_frames:
            raise RuntimeError(f"No valid features extracted for '{cmd}'")

        X = np.vstack(all_frames)
        print(f"  [{cmd}] Training GMM on {X.shape[0]} frames × {X.shape[1]} features ...")

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=GMM_COV_TYPE,
            max_iter=GMM_MAX_ITER,
            n_init=GMM_N_INIT,
            random_state=42,
        )
        gmm.fit(X)

        model_path = os.path.join(save_dir, f"GMM_{cmd}.joblib")
        joblib.dump(gmm, model_path)
        print(f"  [{cmd}] GMM saved → {model_path}")
        gmm_models[cmd] = gmm

    return gmm_models


# ═══════════════════════════════════════════════════════════════════════════════
#  HMM Training & Prediction
# ═══════════════════════════════════════════════════════════════════════════════

def train_hmm_models(audio_dir: str = COMMANDS_DIR,
                     save_dir: str = MODELS_DIR,
                     n_states: int = HMM_N_STATES) -> dict:
    """
    Train one left-to-right Gaussian HMM per command word.
    Returns dict of {command: GaussianHMM}
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        print("  ⚠ hmmlearn not available — skipping HMM training.")
        return {}

    hmm_models = {}
    print("\n[HMM Training]")

    for cmd in COMMANDS:
        cmd_path  = os.path.join(audio_dir, cmd)
        wav_files = [f for f in os.listdir(cmd_path) if f.lower().endswith(AUDIO_EXTENSIONS)]

        if not wav_files:
            raise FileNotFoundError(f"No supported audio files found in {cmd_path}")

        print(f"  [{cmd}] Loading {len(wav_files)} files ...")
        sequences  = []
        lengths    = []

        for wf in wav_files:
            fpath = os.path.join(cmd_path, wf)
            try:
                feats = extract_mfcc_sequence(audio_path=fpath)
                sequences.append(feats)
                lengths.append(len(feats))
            except Exception as e:
                print(f"    ⚠ Skipped {wf}: {e}")

        if not sequences:
            raise RuntimeError(f"No valid features extracted for '{cmd}'")

        X = np.vstack(sequences)
        print(f"  [{cmd}] Training HMM ({n_states} states) on {len(sequences)} utterances ...")

        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=HMM_COV_TYPE,
            n_iter=HMM_N_ITER,
            random_state=42,
        )

        # Left-to-right transition matrix initialization
        transmat = np.zeros((n_states, n_states))
        for i in range(n_states):
            if i < n_states - 1:
                transmat[i, i]   = 0.6
                transmat[i, i+1] = 0.4
            else:
                transmat[i, i] = 1.0
        model.transmat_ = transmat
        model.startprob_ = np.array([1.0] + [0.0] * (n_states - 1))

        try:
            model.fit(X, lengths)
        except Exception as e:
            print(f"  ⚠ HMM fit error for '{cmd}': {e} — using random init")
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type=HMM_COV_TYPE,
                n_iter=HMM_N_ITER,
                random_state=42,
            )
            model.fit(X, lengths)

        model_path = os.path.join(save_dir, f"HMM_{cmd}.joblib")
        joblib.dump(model, model_path)
        print(f"  [{cmd}] HMM saved → {model_path}")
        hmm_models[cmd] = model

    return hmm_models


# ═══════════════════════════════════════════════════════════════════════════════
#  Prediction
# ═══════════════════════════════════════════════════════════════════════════════

def _load_models(model_type: str = "GMM") -> dict:
    """Load saved GMM or HMM models from disk."""
    models = {}
    for cmd in COMMANDS:
        path = os.path.join(MODELS_DIR, f"{model_type}_{cmd}.joblib")
        if os.path.exists(path):
            models[cmd] = joblib.load(path)
    return models


def _score_gmm(model: GaussianMixture, features: np.ndarray) -> float:
    """Average log-likelihood per frame."""
    return model.score(features)   # already per-sample


def _score_hmm(model, features: np.ndarray) -> float:
    """Log-likelihood from HMM."""
    try:
        score, _ = model.decode(features, algorithm="viterbi")
        return score / max(len(features), 1)
    except Exception:
        try:
            return model.score(features) / max(len(features), 1)
        except Exception:
            return -np.inf


def predict_command(audio_path: str = None,
                    audio_array: np.ndarray = None,
                    prefer: str = "HMM",
                    margin: float = LOG_LIKELIHOOD_MARGIN) -> dict:
    """
    Predict the spoken command from an audio file or numpy array.

    Returns:
        {
          "command"    : str,     # "open" | "status" | "unknown"
          "confidence" : float,   # 0.0 – 1.0 normalized score
          "model"      : str,     # "GMM" | "HMM" | "none"
          "scores"     : dict,    # raw log-likelihood per command
        }
    """
    # Extract features
    try:
        features = extract_mfcc_sequence(audio_path=audio_path,
                                          audio_array=audio_array)
    except Exception as e:
        return {"command": "unknown", "confidence": 0.0, "model": "none",
                "scores": {}, "error": str(e)}

    # Try preferred model first, fall back to other
    result = _run_prediction(features, prefer, margin)
    if result["command"] == "unknown" and prefer == "HMM":
        result = _run_prediction(features, "GMM", margin)
        if result["command"] != "unknown":
            result["model"] = "GMM (fallback)"
    return result


def _run_prediction(features, model_type, margin):
    models = _load_models(model_type)
    if not models:
        return {"command": "unknown", "confidence": 0.0, "model": "none",
                "scores": {}}

    scores = {}
    for cmd, model in models.items():
        if model_type == "GMM":
            scores[cmd] = _score_gmm(model, features)
        else:
            scores[cmd] = _score_hmm(model, features)

    # Sort by score descending
    sorted_cmds = sorted(scores, key=lambda c: scores[c], reverse=True)
    best_cmd    = sorted_cmds[0]
    best_score  = scores[best_cmd]

    # Check margin (confidence gap)
    if len(sorted_cmds) > 1:
        second_score = scores[sorted_cmds[1]]
        gap = abs(best_score - second_score)
        if gap < margin:
            # Ambiguous — treat as unknown
            conf = 0.0
            detected = "unknown"
        else:
            conf     = min(1.0, gap / (margin * 3))   # normalize 0-1
            detected = best_cmd
    else:
        conf     = 1.0
        detected = best_cmd

    return {
        "command"    : detected,
        "confidence" : round(conf, 3),
        "model"      : model_type,
        "scores"     : {c: round(s, 2) for c, s in scores.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Permission Check
# ═══════════════════════════════════════════════════════════════════════════════

def get_allowed_commands(user_id: str) -> list:
    """
    Returns list of commands the user is allowed to execute.
    Unknown/unverified users → empty list.
    """
    return USER_PERMISSIONS.get(user_id, [])


def check_permission(user_id: str, command: str) -> dict:
    """
    Returns {allowed: bool, reason: str}
    """
    if command == "unknown":
        return {"allowed": False, "reason": "Command not recognized"}

    allowed = get_allowed_commands(user_id)
    if not allowed:
        return {"allowed": False, "reason": "User not verified"}

    if command in allowed:
        return {"allowed": True, "reason": f"'{command}' executed successfully"}
    else:
        return {"allowed": False,
                "reason": f"Permission Denied: '{command}' requires Admin (User_01)"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_wer(reference_list: list, hypothesis_list: list) -> dict:
    """
    Compute Word Error Rate (WER) for isolated-word recognition.

    For single-word commands:
        WER = Substitutions / N  (since D and I = 0 for isolated words)

    Args:
        reference_list  : list of true command labels
        hypothesis_list : list of predicted command labels

    Returns:
        dict with WER, accuracy, substitutions, total
    """
    assert len(reference_list) == len(hypothesis_list)
    n = len(reference_list)
    subs = sum(1 for r, h in zip(reference_list, hypothesis_list) if r != h)
    wer  = subs / n if n > 0 else 0.0
    return {
        "WER"          : round(wer, 4),
        "Accuracy"     : round(1 - wer, 4),
        "Substitutions": subs,
        "Deletions"    : 0,
        "Insertions"   : 0,
        "Total_Words"  : n,
    }


def evaluate_models(audio_dir: str = COMMANDS_DIR) -> dict:
    """
    Evaluate both GMM and HMM models on all available audio files.
    Returns evaluation results dict.
    """
    results = {}

    for model_type in ["GMM", "HMM"]:
        models = _load_models(model_type)
        if not models:
            print(f"  ⚠ No {model_type} models found — skipping.")
            continue

        y_true, y_pred = [], []

        for true_cmd in COMMANDS:
            cmd_path  = os.path.join(audio_dir, true_cmd)
            if not os.path.exists(cmd_path):
                continue
            wav_files = [f for f in os.listdir(cmd_path) if f.lower().endswith(AUDIO_EXTENSIONS)]

            for wf in wav_files:
                fpath = os.path.join(cmd_path, wf)
                try:
                    features = extract_mfcc_sequence(audio_path=fpath)
                    if model_type == "GMM":
                        scores = {
                            cmd: model.score(features)
                            for cmd, model in models.items()
                        }
                    else:
                        scores = {
                            cmd: model.score(features) / max(len(features), 1)
                            for cmd, model in models.items()
                        }
                    predicted = max(scores, key=scores.get)
                    y_true.append(true_cmd)
                    y_pred.append(predicted)
                except Exception as e:
                    print(f"    ⚠ Error on {wf}: {e}")

        wer_metrics = compute_wer(y_true, y_pred)
        results[model_type] = {
            "y_true"  : y_true,
            "y_pred"  : y_pred,
            "metrics" : wer_metrics,
        }

        print(f"\n[{model_type}] Evaluation Results:")
        print(f"  Accuracy : {wer_metrics['Accuracy']*100:.2f}%")
        print(f"  WER      : {wer_metrics['WER']*100:.2f}%")
        print(f"  Substitutions: {wer_metrics['Substitutions']} / {wer_metrics['Total_Words']}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Command Recognition — GMM + HMM")
    parser.add_argument("--train",    action="store_true", help="Train GMM and HMM models")
    parser.add_argument("--eval",     action="store_true", help="Evaluate trained models")
    parser.add_argument("--predict",  type=str,  default=None, help="Path to WAV file to predict")
    parser.add_argument("--user",     type=str,  default=None, help="User ID for permission check")
    args = parser.parse_args()

    if args.train:
        print("=" * 55)
        print("  Training Command Recognition Models")
        print("=" * 55)
        gmm_models = train_gmm_models()
        hmm_models = train_hmm_models()
        print("\n✅ Training complete.")
        print(f"   GMM models: {list(gmm_models.keys())}")
        print(f"   HMM models: {list(hmm_models.keys())}")

    if args.eval:
        print("\n[Evaluation]")
        evaluate_models()

    if args.predict:
        result = predict_command(audio_path=args.predict)
        print(f"\nPrediction Result:")
        print(f"  Command    : {result['command']}")
        print(f"  Confidence : {result['confidence']}")
        print(f"  Model      : {result['model']}")
        print(f"  Scores     : {result['scores']}")

        if args.user:
            perm = check_permission(args.user, result["command"])
            print(f"\nPermission Check ({args.user}):")
            print(f"  Allowed : {perm['allowed']}")
            print(f"  Reason  : {perm['reason']}")


if __name__ == "__main__":
    main()
