
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from ultralytics import YOLO

from voice_recognition import predict_voice


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACE_MODEL_PATH = (
    PROJECT_ROOT
    / "Models"
    / "Face"
    / "YOLO_Experiment"
    / "best.pt"
)

FACE_TEST_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "YOLO_Experiment"
    / "Face_Classification"
    / "test"
)

VOICE_TEST_USER = "User_01"

SUPPORTED_IMAGE_FORMATS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

SUPPORTED_AUDIO_FORMATS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a"
}


# ============================================================
# 2. LOAD FACE MODEL
# ============================================================

_face_model = None


def load_face_model():

    global _face_model

    if _face_model is None:

        if not FACE_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"YOLO model not found: {FACE_MODEL_PATH}"
            )

        _face_model = YOLO(str(FACE_MODEL_PATH))

    return _face_model


# ============================================================
# 3. FACE RECOGNITION
# ============================================================

def predict_face(face_image_path):

    face_model = load_face_model()

    face_image_path = Path(face_image_path)

    if not face_image_path.is_file():
        raise FileNotFoundError(
            f"Face image not found: {face_image_path}"
        )

    results = face_model.predict(
        source=str(face_image_path),
        imgsz=224,
        verbose=False
    )

    result = results[0]

    if result.probs is None:
        raise RuntimeError(
            "YOLO did not return classification probabilities."
        )

    predicted_index = int(result.probs.top1)

    predicted_user = result.names[predicted_index]

    face_confidence = float(result.probs.top1conf)

    return {
        "predicted_user": str(predicted_user),
        "confidence": face_confidence
    }


# ============================================================
# 4. AUTOMATIC TEST FILE SELECTION
# ============================================================

def get_test_face(user_id):

    user_face_path = FACE_TEST_PATH / user_id

    if not user_face_path.is_dir():
        raise FileNotFoundError(
            f"Face test directory not found: {user_face_path}"
        )

    image_files = sorted(
        file
        for file in user_face_path.iterdir()
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_IMAGE_FORMATS
    )

    if not image_files:
        raise FileNotFoundError(
            f"No test images found for {user_id}."
        )

    return image_files[0]


def get_test_audio(user_id):

    possible_audio_paths = [
        PROJECT_ROOT
        / "Dataset"
        / "Processed"
        / "Voice"
        / "Audio_Standardized"
        / user_id,

        PROJECT_ROOT
        / "Dataset"
        / "Audio_Standardized"
        / user_id,
    ]

    for audio_directory in possible_audio_paths:

        if not audio_directory.is_dir():
            continue

        audio_files = sorted(
            file
            for file in audio_directory.rglob("*")
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_AUDIO_FORMATS
        )

        if audio_files:
            return audio_files[0]

    raise FileNotFoundError(
        f"No standardized audio files found for {user_id}. "
        "Check the Audio_Standardized directory."
    )


# ============================================================
# 5. HIERARCHICAL VERIFICATION
# ============================================================

def hierarchical_verification(face_image_path, audio_path):

    # STEP 1: FACE RECOGNITION

    face_result = predict_face(face_image_path)

    face_identity = face_result["predicted_user"]

    # STEP 2: VOICE RECOGNITION

    voice_result = predict_voice(audio_path)

    voice_identity = voice_result["predicted_user"]

    # STEP 3: IDENTITY MATCHING

    is_verified = face_identity == voice_identity

    verification_status = (
        "Verified" if is_verified else "Rejected"
    )

    # STEP 4: RETURN RESULTS

    return {
        "face_identity": face_identity,
        "face_confidence": face_result["confidence"],
        "voice_identity": voice_identity,
        "verification_status": verification_status,
        "is_verified": is_verified
    }


# ============================================================
# 6. TESTING
# ============================================================

if __name__ == "__main__":

    try:

        # Select test files automatically

        test_face = get_test_face(VOICE_TEST_USER)

        test_audio = get_test_audio(VOICE_TEST_USER)

        print("\n========== TEST INPUTS ==========")

        print(f"Face Image: {test_face}")

        print(f"Audio File: {test_audio}")

        print("=================================\n")

        # Run hierarchical verification

        result = hierarchical_verification(
            face_image_path=test_face,
            audio_path=test_audio
        )

        # Display final result

        print("\n========== HIERARCHICAL VERIFICATION ==========")

        print(
            f"Face Identity: {result['face_identity']}"
        )

        print(
            f"Face Confidence: {result['face_confidence']:.4f}"
        )

        print(
            f"Voice Identity: {result['voice_identity']}"
        )

        print(
            f"Verification Status: {result['verification_status']}"
        )

        print(
            f"Verified: {result['is_verified']}"
        )

        print("================================================\n")

    except Exception as error:

        print(f"\nVerification Error: {error}\n")
