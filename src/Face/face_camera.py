
import cv2
import threading
import time

from face_processing import initialize_background_removal
from face_verification import recognize_face_frame


# ==========================================
# Webcam Configuration
# ==========================================

CAMERA_INDEX = 0

WINDOW_NAME = "Live Face Recognition"

PROCESS_INTERVAL = 1.0


# ==========================================
# Shared Recognition State
# ==========================================

recognition_state = {
    "predicted_user": None,
    "confidence": None,
    "bbox": None,
    "status": "WAITING",
    "error": None
}

state_lock = threading.Lock()

processing_lock = threading.Lock()


# ==========================================
# Face Recognition Worker
# ==========================================

def process_frame(frame, background_session):

    try:

        result = recognize_face_frame(
            frame,
            background_session
        )

        predicted_user, confidence, bbox, status = result

        with state_lock:

            recognition_state["predicted_user"] = predicted_user
            recognition_state["confidence"] = confidence
            recognition_state["bbox"] = bbox
            recognition_state["status"] = status
            recognition_state["error"] = None

    except Exception as error:

        with state_lock:

            recognition_state["predicted_user"] = None
            recognition_state["confidence"] = None
            recognition_state["bbox"] = None
            recognition_state["status"] = "ERROR"
            recognition_state["error"] = str(error)

    finally:

        processing_lock.release()


# ==========================================
# Draw Recognition Results
# ==========================================

def draw_recognition_results(frame):

    with state_lock:

        predicted_user = recognition_state["predicted_user"]
        confidence = recognition_state["confidence"]
        bbox = recognition_state["bbox"]
        status = recognition_state["status"]
        error = recognition_state["error"]

    # Face Detected
    if status == "FACE_DETECTED" and bbox is not None:

        x1, y1, x2, y2 = bbox

        color = (0, 255, 0)

        if confidence is None:
            label = f"Recognized: {predicted_user}"
        else:
            label = (
                f"Recognized: {predicted_user} "
                f"({confidence:.2f})"
            )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    elif status == "NO_FACE":

        cv2.putText(
            frame,
            "No Face Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2
        )

    elif status == "MULTIPLE_FACES":

        cv2.putText(
            frame,
            "Multiple Faces Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    elif status == "ERROR":

        cv2.putText(
            frame,
            "Recognition Error",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "Waiting for Face Recognition...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    # Processing Indicator
    if processing_lock.locked():

        cv2.putText(
            frame,
            "Processing...",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    if status == "ERROR" and error:

        print("Recognition Error:", error)


# ==========================================
# Main Webcam Application
# ==========================================

if __name__ == "__main__":

    print("Initializing Face Recognition System...")

    background_session = initialize_background_removal()

    print("Background Removal Initialized Successfully.")

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        raise RuntimeError(
            "Could not open webcam. Check camera connection."
        )

    print("\nLive Face Recognition Started Successfully.")
    print("------------------------------------------")
    print("Press Q to quit.")
    print("Face recognition runs automatically.")

    last_process_time = 0

    try:

        while True:

            ret, frame = camera.read()

            if not ret:

                print("Failed to capture webcam frame.")
                break

            current_time = time.monotonic()

            # Start Recognition in Background
            if (
                not processing_lock.locked()
                and current_time - last_process_time >= PROCESS_INTERVAL
            ):

                if processing_lock.acquire(blocking=False):

                    frame_copy = frame.copy()

                    worker = threading.Thread(
                        target=process_frame,
                        args=(frame_copy, background_session),
                        daemon=True
                    )

                    worker.start()

                    last_process_time = current_time

            # Draw Latest Recognition Result
            display_frame = frame.copy()

            draw_recognition_results(
                display_frame
            )

            cv2.imshow(
                WINDOW_NAME,
                display_frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                print("\nWebcam Closed.")
                break

    finally:

        camera.release()

        cv2.destroyAllWindows()

        print("Camera Resources Released.")
