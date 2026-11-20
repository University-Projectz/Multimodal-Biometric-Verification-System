
from pathlib import Path
from io import BytesIO

import joblib
import numpy as np
import pandas as pd

from PIL import Image

# Face Processing Functions
from face_processing import (
    initialize_background_removal,
    remove_background,
    resize_face_image,
    detect_face,
    crop_face,
    convert_to_grayscale,
    standardize_grayscale_image
)

# Face Feature Extraction
from face_feature_extraction import extract_face_features


# ==========================================
# Project Paths
# ==========================================

PROJECT_PATH = Path(
    r"E:\Multimodal Biometric Verification System"
)

MODEL_PATH = (
    PROJECT_PATH
    / "Models"
    / "Face"
    / "SVM_Face_Recognition.joblib"
)


# ==========================================
# Load Face Model Bundle
# ==========================================

model_artifacts = joblib.load(MODEL_PATH)

svm_model = model_artifacts["model"]
svm_scaler = model_artifacts["scaler"]
feature_columns = model_artifacts["feature_columns"]

print("Face Model Loaded Successfully.")
print(f"Model Path: {MODEL_PATH}")
print(f"Number of Features: {len(feature_columns)}")


# ==========================================
# Convert Background Removal Output
# ==========================================

def ensure_pil_image(image_output):

    if isinstance(image_output, bytes):

        image_output = Image.open(
            BytesIO(image_output)
        )

    if not isinstance(image_output, Image.Image):

        raise TypeError(
            f"Expected PIL Image or bytes, got {type(image_output)}"
        )

    return image_output.convert("RGB")


# ==========================================
# Face Feature Preparation
# ==========================================

def prepare_face_features(face_crop):

    # 1. Grayscale Conversion
    grayscale_image = convert_to_grayscale(
        face_crop
    )

    # 2. Grayscale Standardization
    standardized_image = standardize_grayscale_image(
        grayscale_image
    )

    # 3. Feature Extraction
    combined_features = extract_face_features(
        standardized_image
    )

    # 4. Feature Validation
    if len(combined_features) != len(feature_columns):

        raise ValueError(
            "Feature count does not match the trained model."
        )

    # 5. Arrange Features in Training Order
    input_features = pd.DataFrame(
        [combined_features],
        columns=feature_columns
    )

    # 6. Feature Scaling
    scaled_features = svm_scaler.transform(
        input_features
    )

    return scaled_features


# ==========================================
# Face Recognition Confidence Configuration
# ==========================================

FACE_CONFIDENCE_THRESHOLD = 0.60


# ==========================================
# Prediction Utilities
# ==========================================

def predict_user_with_confidence(scaled_features):

    if not getattr(svm_model, "probability", False):

        raise RuntimeError(
            "SVM probability estimation is unavailable. "
            "Retrain the model with probability=True."
        )

    probabilities = svm_model.predict_proba(
        scaled_features
    )[0]

    class_index = int(np.argmax(probabilities))

    predicted_user = svm_model.classes_[class_index]

    confidence = float(probabilities[class_index])

    recognized_user = predicted_user

    if confidence < FACE_CONFIDENCE_THRESHOLD:

        recognized_user = "UNKNOWN"

    return recognized_user, confidence, predicted_user


# ==========================================
# Face Verification - Static Image
# ==========================================

def verify_face(image_path, background_session):

    image_path = Path(image_path)

    if not image_path.is_file():

        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    # 1. Load Original Input Image
    image = Image.open(image_path).convert("RGB")

    print("\n========== IMAGE DIAGNOSTICS ==========")
    print("Original Image Size:", image.size)

    # 2. Background Removal
    background_removed_output = remove_background(
        image,
        background_session
    )

    background_removed_image = ensure_pil_image(
        background_removed_output
    )

    print(
        "Background Removed Image Size:",
        background_removed_image.size
    )

    # 3. Image Resizing
    resized_image = resize_face_image(
        background_removed_image
    )

    print(
        "Resized Image Size:",
        resized_image.size
    )

    # 4. Face Detection
    print("\nTesting RetinaFace on RESIZED image...")

    detections = detect_face(resized_image)

    print("Resized Image Detections:", len(detections))

    if len(detections) == 0:

        raise ValueError(
            "No face detected in the input image."
        )

    if len(detections) > 1:

        raise ValueError(
            f"Multiple faces detected: {len(detections)}. "
            "Expected exactly one face."
        )

    # 5. Face Cropping
    face_data = next(iter(detections.values()))

    face_crop = crop_face(
        resized_image,
        face_data
    )

    # 6. Feature Preparation
    scaled_features = prepare_face_features(
        face_crop
    )

    # 7. Prediction + Confidence
    recognized_user, confidence, predicted_user = (
        predict_user_with_confidence(
            scaled_features
        )
    )

    print("\n========== PREDICTION RESULT ==========")
    print("Predicted Identity:", predicted_user)
    print("Recognized Identity:", recognized_user)
    print("Confidence:", confidence)

    return (
        recognized_user,
        confidence,
        predicted_user,
        "FACE_DETECTED"
    )


# ==========================================
# Live Face Recognition - Webcam Frame
# ==========================================

def recognize_face_frame(frame, background_session):

    # 1. Convert OpenCV BGR Frame to PIL RGB Image
    rgb_frame = np.ascontiguousarray(
        frame[:, :, ::-1]
    )

    image = Image.fromarray(rgb_frame)

    original_width, original_height = image.size

    # 2. Background Removal
    background_removed_output = remove_background(
        image,
        background_session
    )

    background_removed_image = ensure_pil_image(
        background_removed_output
    )

    # 3. Image Resizing
    resized_image = resize_face_image(
        background_removed_image
    )

    # 4. Face Detection
    detections = detect_face(resized_image)

    if len(detections) == 0:

        return None, None, None, "NO_FACE"

    if len(detections) > 1:

        return None, None, None, "MULTIPLE_FACES"

    # 5. Get Face Detection Data
    face_data = next(iter(detections.values()))

    x1, y1, x2, y2 = face_data["facial_area"]

    # 6. Map Bounding Box to Original Frame Coordinates
    scale_x = original_width / resized_image.width
    scale_y = original_height / resized_image.height

    bbox = (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    )

    # 7. Face Cropping
    face_crop = crop_face(
        resized_image,
        face_data
    )

    # 8. Feature Preparation
    scaled_features = prepare_face_features(
        face_crop
    )

    # 9. Prediction + Confidence
    recognized_user, confidence, _ = (
        predict_user_with_confidence(
            scaled_features
        )
    )

    return (
        recognized_user,
        confidence,
        bbox,
        "FACE_DETECTED"
    )


# ==========================================
# SVM Model Configuration
# ==========================================

def display_svm_configuration():

    print("\nSVM Configuration")
    print("------------------------------------------")

    print("Model Type:", type(svm_model).__name__)
    print("Kernel:", svm_model.kernel)
    print("Classes:", svm_model.classes_)

    print(
        "Decision Function Shape:",
        svm_model.decision_function_shape
    )

    print(
        "Number of Support Vectors:",
        svm_model.n_support_
    )


# ==========================================
# Static Face Verification Test
# ==========================================

if __name__ == "__main__":

    TEST_IMAGE_PATH = (
        r"E:\supervised-task2-main\supervised-task2-main"
        r"\data\raw\others\public_01\face\olivetti_01.png"
    )

    background_session = initialize_background_removal()

    recognized_user, confidence, predicted_user, status = verify_face(
        TEST_IMAGE_PATH,
        background_session
    )

    print("\n==========================================")
    print("FACE VERIFICATION RESULT")
    print("==========================================")
    print("Input Image:", Path(TEST_IMAGE_PATH).name)
    print("Status:", status)
    print("Predicted Identity:", predicted_user)
    print("Recognized Identity:", recognized_user)
    print("Confidence:", confidence)
    print("==========================================")

    display_svm_configuration()
