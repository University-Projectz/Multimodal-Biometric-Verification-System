
from pathlib import Path

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
# Face Verification - Static Image
# ==========================================

def verify_face(image_path, background_session):

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    # 1. Load Input Image
    image = Image.open(image_path).convert("RGB")

    # 2. Background Removal
    background_removed_image = remove_background(
        image,
        background_session
    )

    # 3. Image Resizing
    resized_image = resize_face_image(
        background_removed_image
    )

    # 4. Face Detection
    detections = detect_face(resized_image)

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

    # 7. SVM Prediction
    predicted_user = svm_model.predict(
        scaled_features
    )[0]

    # 8. SVM Decision Function
    decision_score = float(
        svm_model.decision_function(
            scaled_features
        )[0]
    )

    print("Predicted Identity:", predicted_user)
    print("Decision Score:", decision_score)

    return predicted_user


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
    background_removed_image = remove_background(
        image,
        background_session
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

    # 9. SVM Prediction
    predicted_user = svm_model.predict(
        scaled_features
    )[0]

    # 10. SVM Decision Function
    decision_score = float(
        svm_model.decision_function(
            scaled_features
        )[0]
    )

    return (
        predicted_user,
        decision_score,
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
    print("Number of Support Vectors:", svm_model.n_support_)


# ==========================================
# Static Face Verification Test
# ==========================================

if __name__ == "__main__":

    TEST_IMAGE_PATH = (
        PROJECT_PATH
        / "Dataset"
        / "Raw"
        / "Face"
        / "User_01"
        / "User_01_001.jpg"
    )

    background_session = initialize_background_removal()

    predicted_user = verify_face(
        TEST_IMAGE_PATH,
        background_session
    )

    print("\nFace Verification Completed.")
    print("------------------------------------------")
    print("Input Image:", TEST_IMAGE_PATH.name)
    print("Predicted Identity:", predicted_user)

    display_svm_configuration()
