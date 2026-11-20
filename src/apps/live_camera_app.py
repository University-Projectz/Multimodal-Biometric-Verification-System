
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import cv2
import threading
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

# Ensure current directory is in sys.path for local module imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from voice_recognition import predict_voice
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Command_Recognition.command_recognition import (
    check_permission,
    get_allowed_commands,
    predict_command,
)


# =====================================================
# PATHS AND CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACE_MODEL_PATH = (
    PROJECT_ROOT
    / "Models"
    / "Face"
    / "YOLO_Experiment"
    / "best.pt"
)

CAMERA_INDEX = 0
SAMPLE_RATE = 16000
RECORD_DURATION = 4          # 4 seconds voice capture
COMMAND_RECORD_DURATION = 2  # Isolated command capture
FACE_CONFIDENCE_THRESHOLD = 0.80  # Minimum confidence (80%) to accept face, otherwise 'Unknown'


# =====================================================
# DESKTOP APPLICATION
# =====================================================

class BiometricApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Multimodal Biometric Authentication")
        self.root.geometry("1100x950")
        self.root.minsize(1000, 850)
        self.root.configure(bg="#151923")

        self.running = True
        self.camera = None
        self.frame_count = 0

        self.face_detected = False
        self.face_identity = None
        self.face_confidence = 0.0
        self.voice_identity = None
        self.authenticated_user = None
        self.system_state = "CLOSED"

        self.recording = False
        self.processing_voice = False
        self.command_recording = False
        self.command_processing = False

        # Load OpenCV Face Detector for spatial localization
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Load YOLO model once
        self.model = YOLO(str(FACE_MODEL_PATH))

        self.build_interface()

        self.camera = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

        if not self.camera.isOpened():
            raise RuntimeError("Cannot open camera.")

        self.update_camera()

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    # =================================================
    # USER INTERFACE
    # =================================================

    def build_interface(self):

        title = tk.Label(
            self.root,
            text="MULTIMODAL BIOMETRIC AUTHENTICATION",
            font=("Arial", 22, "bold"),
            fg="white",
            bg="#151923"
        )
        title.pack(pady=20)

        self.video_label = tk.Label(
            self.root,
            bg="black"
        )
        self.video_label.pack(pady=10)

        self.face_label = tk.Label(
            self.root,
            text="Face Recognition: Waiting...",
            font=("Arial", 14, "bold"),
            fg="white",
            bg="#151923"
        )
        self.face_label.pack(pady=8)

        self.voice_label = tk.Label(
            self.root,
            text="Voice Recognition: Waiting...",
            font=("Arial", 14, "bold"),
            fg="white",
            bg="#151923"
        )
        self.voice_label.pack(pady=8)

        self.continue_button = tk.Button(
            self.root,
            text="Continue to Voice Verification",
            font=("Arial", 12, "bold"),
            bg="#2563EB",
            fg="white",
            command=self.start_voice_verification,
            state="disabled",
            padx=15,
            pady=10
        )
        self.continue_button.place(
            relx=0.86,
            rely=0.34,
            anchor="center"
        )

        self.result_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 20, "bold"),
            fg="white",
            bg="#151923"
        )
        self.result_label.pack(pady=5)

        self.detail_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 11),
            fg="#94A3B8",
            bg="#151923"
        )
        self.detail_label.pack(pady=3)

        self.command_status_label = tk.Label(
            self.root,
            text="Command Recognition: Waiting for authentication...",
            font=("Arial", 12, "italic"),
            fg="#64748B",
            bg="#151923"
        )
        self.command_status_label.pack(pady=8)

        self.command_button = tk.Button(
            self.root,
            text="SAY COMMAND",
            font=("Arial", 12, "bold"),
            bg="#0F766E",
            fg="white",
            command=self.start_command_recognition,
            state="disabled",
            padx=15,
            pady=10
        )
        self.command_button.place(
            relx=0.86,
            rely=0.47,
            anchor="center"
        )

        self.reset_button = tk.Button(
            self.root,
            text="New Authentication",
            font=("Arial", 11),
            command=self.reset_authentication
        )
        self.reset_button.place(
            relx=0.86,
            rely=0.60,
            anchor="center"
        )

    # =================================================
    # LIVE CAMERA + FACE DETECTION + YOLO
    # =================================================

    def update_camera(self):

        if not self.running:
            return

        ret, frame = self.camera.read()

        if ret:

            self.frame_count += 1

            # Detect faces using OpenCV Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            if len(faces) == 0:
                self.face_detected = False
                self.face_identity = None
                self.face_confidence = 0.0
                self.root.after(0, self.update_face_status)
            else:
                self.face_detected = True

                # Select the largest face in the frame
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

                # Run YOLO periodically on the detected face
                if self.frame_count % 5 == 0:

                    try:
                        # Extract face crop with safety margin
                        margin = int(0.15 * w)
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(frame.shape[1], x + w + margin)
                        y2 = min(frame.shape[0], y + h + margin)
                        face_crop = frame[y1:y2, x1:x2]

                        yolo_input = face_crop if face_crop.size > 0 else frame

                        results = self.model.predict(
                            source=yolo_input,
                            verbose=False
                        )

                        prediction = results[0].probs
                        predicted_conf = float(prediction.top1conf)
                        predicted_name = self.model.names[prediction.top1]

                        # Security check: if confidence is below threshold, label as 'Unknown'
                        if predicted_conf >= FACE_CONFIDENCE_THRESHOLD:
                            self.face_identity = predicted_name
                            self.face_confidence = predicted_conf
                        else:
                            self.face_identity = "Unknown"
                            self.face_confidence = predicted_conf

                        self.root.after(
                            0,
                            self.update_face_status
                        )

                    except Exception as error:
                        print("YOLO Error:", error)

                # Draw bounding box and label on live camera
                if self.face_identity == "Unknown":
                    box_color = (0, 0, 255)  # Red for unknown
                    box_text = f"UNKNOWN ({self.face_confidence:.1%})"
                elif self.face_identity is not None:
                    box_color = (0, 255, 0)  # Green for recognized user
                    box_text = f"{self.face_identity} ({self.face_confidence:.1%})"
                else:
                    box_color = (0, 200, 255) # Orange/Yellow while detecting
                    box_text = "Analyzing face..."

                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(
                    frame,
                    box_text,
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    box_color,
                    2
                )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(frame)

            image = image.resize((800, 450))

            photo = ImageTk.PhotoImage(image=image)

            self.video_label.configure(image=photo)
            self.video_label.image = photo

        self.root.after(30, self.update_camera)

    # =================================================
    # FACE STATUS
    # =================================================

    def update_face_status(self):

        if not self.face_detected:
            self.face_label.configure(
                text="Face Recognition: Searching for face...",
                fg="#94A3B8"
            )
            self.continue_button.configure(state="disabled")
            return

        if self.face_identity == "Unknown":
            self.face_label.configure(
                text=f"Face Recognition: UNKNOWN PERSON ({self.face_confidence:.2%})",
                fg="#EF4444"
            )
            self.continue_button.configure(state="disabled")

        elif self.face_identity is not None:
            self.face_label.configure(
                text=(
                    f"Face Recognition: "
                    f"{self.face_identity} "
                    f"({self.face_confidence:.2%})"
                ),
                fg="#22C55E"
            )

            if not self.recording and not self.processing_voice:
                self.continue_button.configure(state="normal")

    # =================================================
    # VOICE RECORDING
    # =================================================

    def start_voice_verification(self):

        if not self.face_detected or self.face_identity == "Unknown" or self.face_identity is None:
            return

        if self.face_confidence < FACE_CONFIDENCE_THRESHOLD:
            return

        if self.recording or self.processing_voice:
            return

        self.recording = True

        self.continue_button.configure(
            state="disabled",
            text="Recording..."
        )

        self.voice_label.configure(
            text=f"Recording voice for {RECORD_DURATION} seconds... Speak now",
            fg="#FACC15"
        )

        threading.Thread(
            target=self.record_and_predict_voice,
            daemon=True
        ).start()

    def record_and_predict_voice(self):

        audio_path = None

        try:

            audio = sd.rec(
                int(RECORD_DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )

            sd.wait()

            # Inspect and calibrate audio level to prevent loudness bias
            audio_flat = audio.flatten()
            peak = float(np.max(np.abs(audio_flat)))
            rms = float(np.sqrt(np.mean(audio_flat ** 2)))
            print(f"\n[VOICE AUDIO] Raw Mic -> Peak: {peak:.4f}, RMS: {rms:.4f}")

            # If microphone volume is too loud, scale to training baseline (~0.30)
            if peak > 0.38:
                scale_factor = 0.30 / peak
                audio = audio * scale_factor
                print(f"[VOICE AUDIO] Calibrated Volume -> Scaled Peak: {float(np.max(np.abs(audio))):.4f}")

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as file:

                audio_path = Path(file.name)

            sf.write(
                str(audio_path),
                audio,
                SAMPLE_RATE,
                subtype="PCM_16"
            )

            self.root.after(
                0,
                lambda: self.voice_label.configure(
                    text="Processing voice...",
                    fg="#FACC15"
                )
            )

            self.processing_voice = True

            result = predict_voice(audio_path)

            self.voice_identity = result["predicted_user"]

            self.root.after(
                0,
                self.show_authentication_result
            )

        except Exception as error:

            self.root.after(
                0,
                lambda: self.voice_label.configure(
                    text=f"Voice Error: {error}",
                    fg="#EF4444"
                )
            )

        finally:

            if audio_path and audio_path.exists():
                audio_path.unlink()

            self.recording = False
            self.processing_voice = False

    # =================================================
    # FINAL AUTHENTICATION & COMMAND HOOKS
    # =================================================

    def show_authentication_result(self):

        self.voice_label.configure(
            text=f"Voice Recognition: {self.voice_identity}",
            fg="#22C55E"
        )

        if self.face_identity == self.voice_identity:
            self.authenticated_user = self.face_identity

            self.result_label.configure(
                text=f"ACCESS GRANTED: Welcome {self.face_identity}!",
                fg="#22C55E"
            )

            self.detail_label.configure(
                text=f"Verification Successful (Face: {self.face_confidence:.1%} | Voice: Matched)",
                fg="#86EFAC"
            )

            self.command_status_label.configure(
                text=f"🎙️ Command Recognition: ACTIVE - Press SAY COMMAND, then speak",
                fg="#38BDF8"
            )
            self.command_button.configure(state="normal", text="SAY COMMAND")

            self.on_access_granted(self.face_identity)

        else:

            self.result_label.configure(
                text="ACCESS DECLINED",
                fg="#EF4444"
            )

            self.detail_label.configure(
                text=f"Identity Mismatch: Face is '{self.face_identity}' but Voice is '{self.voice_identity}'.",
                fg="#FCA5A5"
            )

            self.command_status_label.configure(
                text="🔒 Command Recognition: LOCKED (Authentication Required)",
                fg="#94A3B8"
            )
            self.command_button.configure(state="disabled")

            self.on_access_declined(
                f"Mismatch: Face '{self.face_identity}' != Voice '{self.voice_identity}'"
            )

        self.continue_button.configure(
            text="Verification Completed",
            state="disabled"
        )

    def on_access_granted(self, authenticated_user):
        """
        HOOK: Triggered immediately when biometric authentication succeeds.
        This prepares and unlocks the system for the upcoming Command Recognition model!
        """
        print("\n" + "=" * 50)
        print(f" [AUTH SUCCESS] ACCESS GRANTED FOR: {authenticated_user}")
        print(f" [COMMAND MODULE] Unlocked & Ready to receive voice commands...")
        print("=" * 50 + "\n")

    def start_command_recognition(self):
        """Record and classify one command after multimodal authentication."""
        if not self.authenticated_user:
            return

        if self.command_recording or self.command_processing:
            return

        self.command_recording = True
        self.command_button.configure(
            state="disabled",
            text="LISTENING..."
        )
        self.command_status_label.configure(
            text=f"🎙️ Recording command for {COMMAND_RECORD_DURATION} seconds...",
            fg="#FACC15"
        )

        threading.Thread(
            target=self.record_and_predict_command,
            daemon=True
        ).start()

    def record_and_predict_command(self):
        audio_path = None

        try:
            audio = sd.rec(
                int(COMMAND_RECORD_DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
                audio_path = Path(file.name)

            sf.write(
                str(audio_path),
                audio,
                SAMPLE_RATE,
                subtype="PCM_16"
            )

            self.command_processing = True
            result = predict_command(
                audio_path=str(audio_path),
                prefer="GMM",
                margin=5
            )
            permission = check_permission(
                self.authenticated_user,
                result["command"]
            )

            self.root.after(
                0,
                lambda: self.show_command_result(result, permission)
            )

        except Exception as error:
            self.root.after(
                0,
                lambda: self.command_status_label.configure(
                    text=f"Command Error: {error}",
                    fg="#EF4444"
                )
            )
        finally:
            if audio_path and audio_path.exists():
                audio_path.unlink()

            self.command_recording = False
            self.command_processing = False
            self.root.after(
                0,
                lambda: self.command_button.configure(
                    state="normal" if self.authenticated_user else "disabled",
                    text="SAY COMMAND"
                )
            )

    def show_command_result(self, result, permission):
        command = result["command"]
        confidence = result.get("confidence", 0.0)

        if permission["allowed"]:
            if command == "open":
                self.system_state = "OPEN"
                self.result_label.configure(
                    text="SYSTEM STATE: OPEN",
                    fg="#22C55E"
                )
                self.detail_label.configure(
                    text=f"Command heard: OPEN | User: {self.authenticated_user}",
                    fg="#86EFAC"
                )
            elif command == "status":
                permissions = ", ".join(
                    get_allowed_commands(self.authenticated_user)
                ) or "none"
                self.result_label.configure(
                    text=f"STATUS: {self.authenticated_user}",
                    fg="#38BDF8"
                )
                self.detail_label.configure(
                    text=(
                        f"Command heard: STATUS | State: {self.system_state} | "
                        f"Face: {self.face_confidence:.1%} | Voice: Matched (SVM) | "
                        f"Permissions: {permissions}"
                    ),
                    fg="#BAE6FD"
                )

            self.command_status_label.configure(
                text=f"✅ Command '{command}' allowed ({confidence:.1%})",
                fg="#22C55E"
            )
            print(
                f"[COMMAND ALLOWED] User={self.authenticated_user} "
                f"Command={command} Model={result.get('model')}"
            )
        else:
            self.command_status_label.configure(
                text=f"❌ Command denied: {permission['reason']}",
                fg="#EF4444"
            )
            print(
                f"[COMMAND DENIED] User={self.authenticated_user} "
                f"Command={command} Reason={permission['reason']}"
            )

    def on_access_declined(self, reason):
        """
        HOOK: Triggered when authentication fails or is rejected.
        """
        print("\n" + "=" * 50)
        print(f" [AUTH FAILED] ACCESS DECLINED: {reason}")
        print("=" * 50 + "\n")

    # =================================================
    # RESET
    # =================================================

    def reset_authentication(self):

        if self.recording or self.processing_voice:
            return

        self.face_detected = False
        self.face_identity = None
        self.face_confidence = 0.0
        self.voice_identity = None
        self.authenticated_user = None
        self.system_state = "CLOSED"
        self.command_recording = False
        self.command_processing = False

        self.face_label.configure(
            text="Face Recognition: Waiting...",
            fg="white"
        )

        self.voice_label.configure(
            text="Voice Recognition: Waiting...",
            fg="white"
        )

        self.result_label.configure(text="")
        self.detail_label.configure(text="")

        self.command_status_label.configure(
            text="Command Recognition: Waiting for authentication...",
            fg="#64748B"
        )
        self.command_button.configure(
            text="SAY COMMAND",
            state="disabled"
        )

        self.continue_button.configure(
            text="Continue to Voice Verification",
            state="disabled"
        )

    # =================================================
    # CLOSE APPLICATION
    # =================================================

    def close_app(self):

        self.running = False

        if self.camera is not None:
            self.camera.release()

        self.root.destroy()


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = BiometricApp(root)

    root.mainloop()
