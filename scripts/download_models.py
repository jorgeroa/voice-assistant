"""Download Kokoro ONNX model files and pre-cache WhisperX models."""

import shutil
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_FILES = {
    "kokoro-v1.0.onnx": f"{KOKORO_RELEASE}/kokoro-v1.0.onnx",
    "voices-v1.0.bin": f"{KOKORO_RELEASE}/voices-v1.0.bin",
}


def check_system_deps():
    """Check that required system dependencies are installed."""
    missing = []
    for cmd in ["ffmpeg", "espeak-ng"]:
        if shutil.which(cmd) is None:
            missing.append(cmd)
    if missing:
        print(f"Missing system dependencies: {', '.join(missing)}")
        print("Install with: brew install " + " ".join(missing))
        return False
    print("System dependencies OK (ffmpeg, espeak-ng)")
    return True


def download_kokoro():
    """Download Kokoro ONNX model and voices."""
    MODELS_DIR.mkdir(exist_ok=True)

    for filename, url in KOKORO_FILES.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"  {filename} already exists, skipping.")
            continue
        print(f"  Downloading {filename}...")
        subprocess.check_call(
            ["curl", "-L", "-o", str(dest), url],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print(f"  {filename} downloaded ({dest.stat().st_size / 1e6:.1f} MB)")


def precache_whisper(model_name: str = "base"):
    """Pre-download a faster-whisper model so first run is fast."""
    try:
        from faster_whisper import WhisperModel

        print(f"  Pre-caching faster-whisper model '{model_name}'...")
        WhisperModel(model_name, device="cpu", compute_type="float32")
        print(f"  Model '{model_name}' cached.")
    except ImportError:
        print("  faster-whisper not installed, skipping pre-cache.")
    except Exception as e:
        print(f"  Warning: could not pre-cache model: {e}")


def main():
    print("=== Voice Conversation - Model Setup ===\n")

    print("1. Checking system dependencies...")
    check_system_deps()

    print("\n2. Downloading Kokoro TTS models...")
    download_kokoro()

    print("\n3. Pre-caching Whisper model...")
    precache_whisper("base")

    print("\nDone! You can now run: python main.py")


if __name__ == "__main__":
    main()
