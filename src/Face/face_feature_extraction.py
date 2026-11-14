
import numpy as np

from skimage.feature import hog, local_binary_pattern


# ==========================================
# Feature Extraction Configuration
# ==========================================

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"
LBP_BINS = LBP_POINTS + 2

EXPECTED_HOG_FEATURES = 47988
EXPECTED_LBP_FEATURES = 10

EXPECTED_TOTAL_FEATURES = (
    EXPECTED_HOG_FEATURES + EXPECTED_LBP_FEATURES
)


# ==========================================
# LBP Feature Extraction
# ==========================================

def extract_lbp_features(standardized_image):

    image_array = np.array(standardized_image)

    lbp_image = local_binary_pattern(
        image_array,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD
    )

    lbp_histogram, _ = np.histogram(
        lbp_image.ravel(),
        bins=np.arange(0, LBP_BINS + 1),
        range=(0, LBP_BINS)
    )

    lbp_features = lbp_histogram.astype(float)

    lbp_features /= lbp_features.sum()

    return lbp_features


# ==========================================
# HOG Feature Extraction
# ==========================================

def extract_hog_features(standardized_image):

    image_array = np.array(standardized_image)

    hog_features = hog(
        image_array,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=False,
        feature_vector=True
    )

    return hog_features


# ==========================================
# Combined Face Feature Extraction
# ==========================================

def extract_face_features(standardized_image):

    lbp_features = extract_lbp_features(
        standardized_image
    )

    hog_features = extract_hog_features(
        standardized_image
    )

    if len(lbp_features) != EXPECTED_LBP_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_LBP_FEATURES} LBP features, "
            f"but received {len(lbp_features)}."
        )

    if len(hog_features) != EXPECTED_HOG_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_HOG_FEATURES} HOG features, "
            f"but received {len(hog_features)}."
        )

    # Important: LBP must come before HOG.
    combined_features = np.concatenate([
        lbp_features,
        hog_features
    ])

    if len(combined_features) != EXPECTED_TOTAL_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_FEATURES} total features, "
            f"but received {len(combined_features)}."
        )

    return combined_features





# ==========================================
# Feature Extraction Test
# ==========================================

if __name__ == "__main__":

    from PIL import Image

    TEST_IMAGE_PATH = (
        r"E:\Multimodal Biometric Verification System"
        r"\Dataset\Processed\Face"
        r"\Face_Grayscale_Standardization"
        r"\User_01"
        r"\User_01_001_Face_1.png"
    )

    standardized_image = Image.open(
        TEST_IMAGE_PATH
    ).convert("L")

    combined_features = extract_face_features(
        standardized_image
    )

    print("Face Feature Extraction Test Successful.")
    print("------------------------------------------")
    print("Image Size:", standardized_image.size)
    print("Image Mode:", standardized_image.mode)
    print("LBP Features:", EXPECTED_LBP_FEATURES)
    print("HOG Features:", EXPECTED_HOG_FEATURES)
    print("Combined Features:", combined_features.shape)
