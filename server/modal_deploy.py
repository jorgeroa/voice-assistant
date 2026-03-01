"""Modal.com deployment for the voice conversation server.

Deploy with:
    modal deploy server/modal_deploy.py

Or run temporarily:
    modal serve server/modal_deploy.py
"""

import modal

app = modal.App("voice-conversation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "espeak-ng", "curl")
    .pip_install(
        "numpy",
        "soundfile",
        "torch",
        "faster-whisper>=1.0",
        "openai>=1.30",
        "kokoro-onnx>=0.4",
        "onnxruntime>=1.17",
        "pydantic-settings>=2.0",
        "python-dotenv>=1.0",
        "fastapi>=0.111",
        "uvicorn[standard]>=0.30",
        "websockets>=12.0",
    )
    .run_commands(
        # Pre-download faster-whisper base model
        'python -c "from faster_whisper import WhisperModel; WhisperModel(\'base\', device=\'cpu\')"',
        # Download Kokoro model files
        "mkdir -p /models",
        "curl -L -o /models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "curl -L -o /models/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    container_idle_timeout=120,
    secrets=[modal.Secret.from_name("voice-conversation-secrets")],
    mounts=[
        modal.Mount.from_local_dir("config", remote_path="/app/config"),
        modal.Mount.from_local_dir("core", remote_path="/app/core"),
        modal.Mount.from_local_dir("server", remote_path="/app/server"),
        modal.Mount.from_local_dir("static", remote_path="/app/static"),
    ],
)
@modal.web_server(port=8000)
def serve():
    import os
    import sys

    sys.path.insert(0, "/app")
    os.chdir("/app")

    # Override model paths to use pre-downloaded models
    os.environ.setdefault("KOKORO_MODEL_PATH", "/models/kokoro-v1.0.onnx")
    os.environ.setdefault("KOKORO_VOICES_PATH", "/models/voices-v1.0.bin")
    os.environ.setdefault("WHISPER_DEVICE", "cuda")
    os.environ.setdefault("WHISPER_COMPUTE_TYPE", "float16")

    import uvicorn

    from config.settings import AppSettings
    from server.app import app, configure

    settings = AppSettings()
    configure(settings)

    uvicorn.run(app, host="0.0.0.0", port=8000)
