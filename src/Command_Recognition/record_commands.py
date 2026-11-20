"""
record_commands.py
==================
سكريبت لتسجيل أوامر صوتية (open / status) وحفظها في Dataset/Raw/Voice/Commands/
شغّل من Terminal:
    & "E:\Tools\miniconda\envs\biometric\python.exe" "src/Command_Recognition/record_commands.py"
"""

import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

# ─── Settings ───────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000       # 16 kHz
DURATION      = 2           # ثانيتين لكل تسجيل
COMMANDS      = ["open", "status"]
SAMPLES_EACH  = 15          # عدد العينات لكل كلمة
SAVE_BASE     = r"E:\Multimodal Biometric Verification System\Dataset\Raw\Voice\Commands"
# ─────────────────────────────────────────────────────────────────────────────


def record_audio(duration=DURATION, sr=SAMPLE_RATE) -> np.ndarray:
    """تسجّل صوت من الميكروفون وترجع numpy array"""
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def normalize(audio: np.ndarray, target_peak=0.9) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio


def main():
    print("=" * 55)
    print("  Command Recording Tool — Biometric Project")
    print("=" * 55)
    print(f"  Commands   : {COMMANDS}")
    print(f"  Samples    : {SAMPLES_EACH} per command")
    print(f"  Duration   : {DURATION}s each")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    print("=" * 55)

    for cmd in COMMANDS:
        cmd_dir = os.path.join(SAVE_BASE, cmd)
        os.makedirs(cmd_dir, exist_ok=True)

        # اعرف آخر رقم موجود
        existing = [f for f in os.listdir(cmd_dir) if f.endswith(".wav")]
        start_idx = len(existing) + 1

        print(f"\n📢  Ready to record command: '{cmd.upper()}'")
        print(f"    Will record {SAMPLES_EACH} samples starting from #{start_idx:03d}")
        input("    Press ENTER when ready...")

        for i in range(start_idx, start_idx + SAMPLES_EACH):
            filename = f"{cmd}_{i:03d}.wav"
            filepath = os.path.join(cmd_dir, filename)

            print(f"\n  [{i - start_idx + 1}/{SAMPLES_EACH}]  🎙️  Say '{cmd}' NOW ...")
            audio = record_audio()
            audio = normalize(audio)

            sf.write(filepath, audio, SAMPLE_RATE)
            print(f"           ✅ Saved → {filename}")
            time.sleep(0.5)   # استنى نص ثانية بين التسجيلات

        print(f"\n  ✔  Done recording '{cmd}' — {SAMPLES_EACH} files saved.")

    print("\n" + "=" * 55)
    print("  🎉 All commands recorded successfully!")
    print(f"  Files saved in: {SAVE_BASE}")
    print("=" * 55)


if __name__ == "__main__":
    main()
