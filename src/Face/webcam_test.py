
import cv2
from pathlib import Path
from face_verification import verify_face
from face_processing import initialize_background_removal


# ==========================================
# 1. Initialize
# ==========================================

OUTPUT_PATH = Path("webcam_test.jpg")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Webcam opened successfully.")
print("Press SPACE to capture an image.")
print("Press Q to quit.")


# ==========================================
# 2. Capture Image
# ==========================================

captured = None

while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to read webcam frame.")
        break

    cv2.imshow("Webcam Test - SPACE to Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        captured = frame.copy()
        break

    elif key == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()


# ==========================================
# 3. Save Image
# ==========================================

if captured is None:
    print("No image captured.")
    raise SystemExit

cv2.imwrite(str(OUTPUT_PATH), captured)

print(f"\nImage saved: {OUTPUT_PATH.resolve()}")


# ==========================================
# 4. Run Face Verification
# ==========================================

print("\nInitializing background removal...")

background_session = initialize_background_removal()

print("\nRunning Face Verification...")

result = verify_face(
    str(OUTPUT_PATH),
    background_session
)

print("\n==========================================")
print("WEBCAM FACE VERIFICATION RESULT")
print("==========================================")

recognized_user, confidence, predicted_user, status = result

print(f"Input Image: {OUTPUT_PATH}")
print(f"Status: {status}")
print(f"Predicted Identity: {predicted_user}")
print(f"Recognized Identity: {recognized_user}")
print(f"Confidence: {confidence:.4f}")

print("==========================================")
