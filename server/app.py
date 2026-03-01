from __future__ import annotations

import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config.settings import AppSettings
from server.session import ConversationSession

logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Conversation")

# Settings are loaded once at startup
_settings: AppSettings | None = None


def configure(settings: AppSettings):
    global _settings
    _settings = settings


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.websocket("/ws/conversation")
async def conversation_ws(websocket: WebSocket):
    await websocket.accept()
    settings = _settings or AppSettings()

    async def send_json(data: dict):
        await websocket.send_text(json.dumps(data))

    session = ConversationSession(settings=settings, send_fn=send_json)
    logger.info("WebSocket connection established")

    try:
        await send_json({"type": "status", "message": "Connected. Ready to talk."})
        await session._send_voices()

        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            await session.handle_message(data)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.close(code=1011, reason=str(e)[:120])
        except Exception:
            pass


# Mount static files after routes so / route takes priority
app.mount("/static", StaticFiles(directory="static"), name="static")
