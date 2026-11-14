
from pathlib import Path
import os

import numpy as np
import cv2
import joblib

from PIL import Image



# ==========================================
# Environment Configuration
# ==========================================

# Configure rembg model directory
REMBG_HOME = Path(
    r"E:\Tools\miniconda\envs\biometric\rembg_models"
)

os.environ["U2NET_HOME"] = str(REMBG_HOME)


# Configure RetinaFace model directory
DEEPFACE_HOME = Path(
    r"E:\Tools\miniconda\envs\biometric"
)

os.environ["DEEPFACE_HOME"] = str(DEEPFACE_HOME)

# Force RetinaFace to use ONNX backend
os.environ["DEEPFACE_BACKEND_ENGINE"] = "onnx"


# Verify RetinaFace ONNX model
RETINAFACE_MODEL_PATH = (
    DEEPFACE_HOME
    / ".deepface"
    / "weights"
    / "retinaface.onnx"
)

if not RETINAFACE_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"RetinaFace ONNX model not found: {RETINAFACE_MODEL_PATH}"
    )


# Import after configuring environment variables
from rembg import new_session, remove
from retinaface import RetinaFace


# ==========================================
# Project Configuration
# ==========================================

PROJECT_PATH = Path(
    r"E:\Multimodal Biometric Verification System"
)


# Face Model
MODEL_PATH = (
    PROJECT_PATH
    / "Models"
    / "Face"
    / "SVM_Face_Recognition.joblib"
)


# Face Processing Configuration
USERS = ["User_01", "User_02"]

PADDING_RATIO = 0.15


# RetinaFace Input Dimensions
TARGET_WIDTH = 1024
TARGET_HEIGHT = 1365


# Grayscale Standardization Dimensions
GRAY_WIDTH = 256
GRAY_HEIGHT = 352


# Feature Extraction Configuration
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"
LBP_BINS = LBP_POINTS + 2


# Background Removal Model
REMBG_MODEL = "u2net"


print("Face Processing Configuration Loaded Successfully.")
print("RetinaFace Backend: ONNX")
print("RetinaFace Model:", RETINAFACE_MODEL_PATH)


# ==========================================
# Background Removal Initialization
# ==========================================

def initialize_background_removal():

    """
    Initialize the U²-Net session for background removal.
    """

    session = new_session(REMBG_MODEL)

    return session






# ==========================================
# Background Removal
# ==========================================

def remove_background(image, session):

    """
    Remove the background from an input face image
    using the initialized U²-Net session.
    """

    input_image = image.convert("RGB")

    output_image = remove(
        input_image,
        session=session
    )

    return output_image



# ==========================================
# Image Resizing
# ==========================================

def calculate_resize_dimensions(image_width, image_height):
    """
    Calculate new image dimensions while preserving
    the aspect ratio and avoiding upscaling.
    """

    scale = min(
        TARGET_WIDTH / image_width,
        TARGET_HEIGHT / image_height,
        1.0
    )

    new_width = round(image_width * scale)
    new_height = round(image_height * scale)

    return new_width, new_height


def resize_face_image(image):
    """
    Resize the image to fit within the target dimensions
    while preserving its aspect ratio.
    """

    new_width, new_height = calculate_resize_dimensions(
        image.width,
        image.height
    )

    resized_image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    return resized_image


# ==========================================
# Face Detection
# ==========================================

def detect_face(image):
    """
    Detect faces in an image using RetinaFace.
    """

    rgb_image = image.convert("RGB")

    image_array = np.array(rgb_image)[:, :, ::-1].copy()

    detections = RetinaFace.detect_faces(image_array)

    if isinstance(detections, dict):
        return detections

    return {}





# ==========================================
# Face Cropping
# ==========================================

def crop_face(image, face_data):
    """
    Crop the detected face with additional padding.
    """

    x1, y1, x2, y2 = face_data["facial_area"]

    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = int(box_width * PADDING_RATIO)
    pad_y = int(box_height * PADDING_RATIO)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)

    crop_x2 = min(image.width, x2 + pad_x)
    crop_y2 = min(image.height, y2 + pad_y)

    face_crop = image.crop(
        (crop_x1, crop_y1, crop_x2, crop_y2)
    )

    return face_crop








# ==========================================
# Grayscale Conversion
# ==========================================

def convert_to_grayscale(face_crop):

    grayscale_image = face_crop.convert("L")

    return grayscale_image





# ==========================================
# Grayscale Standardization
# ==========================================

def standardize_grayscale_image(grayscale_image):

    original_width, original_height = grayscale_image.size

    scale = min(
        GRAY_WIDTH / original_width,
        GRAY_HEIGHT / original_height
    )

    new_width = round(original_width * scale)
    new_height = round(original_height * scale)

    resized_image = grayscale_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    image_array = np.array(resized_image)

    border_pixels = np.concatenate([
        image_array[0, :],
        image_array[-1, :],
        image_array[:, 0],
        image_array[:, -1]
    ])

    padding_value = int(np.median(border_pixels))

    standardized_image = Image.new(
        "L",
        (GRAY_WIDTH, GRAY_HEIGHT),
        color=padding_value
    )

    paste_x = (GRAY_WIDTH - new_width) // 2
    paste_y = (GRAY_HEIGHT - new_height) // 2

    standardized_image.paste(
        resized_image,
        (paste_x, paste_y)
    )

    return standardized_image














# ==========================================
# Face Processing Tests
# ==========================================

if __name__ == "__main__":

    # Initialize U²-Net
    rembg_session = initialize_background_removal()

    print("U²-Net Session Initialized Successfully.")

    # Select one raw face image
    RAW_FACE_PATH = (
        PROJECT_PATH
        / "Dataset"
        / "Raw"
        / "Face"
        / "User_01"
    )

    test_images = sorted([
        path for path in RAW_FACE_PATH.iterdir()
        if path.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    if not test_images:
        raise FileNotFoundError("No face images found.")

    test_image_path = test_images[0]

    # Apply Background Removal
    with Image.open(test_image_path) as image:

        processed_image = remove_background(
            image,
            rembg_session
        )

    print("\nBackground Removal Test Successful.")
    print("Input Image:", test_image_path.name)
    print("Input Mode:", image.mode)
    print("Output Mode:", processed_image.mode)
    print("Output Size:", processed_image.size)

    # Image Resizing Test
    resized_image = resize_face_image(processed_image)

    print("\nImage Resizing Test Successful.")
    print("Original Size:", processed_image.size)
    print("Resized Size:", resized_image.size)
    print("Image Mode:", resized_image.mode)

    # Face Detection Test
    detections = detect_face(resized_image)

    print("\nFace Detection Test")
    print("------------------------------")

    if detections:
        print("Face Detected Successfully.")
        print("Number of Faces:", len(detections))

        for face_id, face_data in detections.items():
            print(f"\nFace ID: {face_id}")
            print("Bounding Box:", face_data["facial_area"])
            print("Confidence:", face_data["score"])

    else:
        print("No Face Detected.")

    # Face Cropping Test
    if len(detections) == 1:

        face_data = next(iter(detections.values()))

        face_crop = crop_face(
            resized_image,
            face_data
        )

        print("\nFace Cropping Test Successful.")
        print("Cropped Face Size:", face_crop.size)
        print("Cropped Face Mode:", face_crop.mode)

    else:
        raise ValueError(
            f"Expected 1 face, but detected {len(detections)}."
        )

    # Grayscale Conversion Test
    grayscale_image = convert_to_grayscale(face_crop)

    print("\nGrayscale Conversion Test Successful.")
    print("Grayscale Image Size:", grayscale_image.size)
    print("Grayscale Image Mode:", grayscale_image.mode)

    # Grayscale Standardization Test
    standardized_image = standardize_grayscale_image(
        grayscale_image
    )

    print("\nGrayscale Standardization Test Successful.")
    print("Standardized Image Size:", standardized_image.size)
    print("Standardized Image Mode:", standardized_image.mode)