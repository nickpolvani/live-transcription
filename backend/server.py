"""FastAPI application — serves the frontend and the WebSocket transcription endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from backend.audio_capture import AudioCapture
from backend.config import config
from backend.transcriber import Transcriber, TranscriptionSegment

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# ── Application & shared state ──────────────────────────────────────────────

app = FastAPI(title="Live Transcription")

transcriber = Transcriber()


class SessionState:
    """Per-connection mutable state (only one concurrent session supported)."""

    def __init__(self) -> None:
        self.recording: bool = False
        self.language: str = config.language
        self.transcript_segments: List[Dict] = []
        self.start_time: Optional[float] = None
        self.audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self.capture: Optional[AudioCapture] = None
        self._processing_task: Optional[asyncio.Task] = None


session = SessionState()

# ── Startup / shutdown hooks ────────────────────────────────────────────────


@app.on_event("startup")
async def startup() -> None:
    # Load model in a thread so we don't block the event loop
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, transcriber.load_model)
    logger.info("Application ready")


# ── REST endpoints ──────────────────────────────────────────────────────────


@app.get("/api/config")
async def get_config() -> JSONResponse:
    data = config.to_dict()
    data["model_info"]["loaded"] = transcriber.is_loaded
    return JSONResponse(data)


@app.get("/api/transcript")
async def get_transcript() -> PlainTextResponse:
    text = "\n".join(seg["text"] for seg in session.transcript_segments)
    return PlainTextResponse(text)


# ── WebSocket endpoint ──────────────────────────────────────────────────────


async def _send_json(ws: WebSocket, payload: dict) -> None:
    await ws.send_text(json.dumps(payload))


async def _send_status(ws: WebSocket) -> None:
    elapsed = 0.0
    if session.recording and session.start_time:
        elapsed = time.time() - session.start_time
    await _send_json(ws, {
        "type": "status",
        "recording": session.recording,
        "model_loaded": transcriber.is_loaded,
        "elapsed": round(elapsed, 1),
    })


def _transcribe_worker(audio_q: queue.Queue, result_q: asyncio.Queue, language: str) -> None:
    """Runs in a background *thread* — pulls audio chunks, transcribes, puts results."""
    while True:
        chunk = audio_q.get()
        if chunk is None:
            # Sentinel — capture stopped
            result_q.put_nowait(None)
            break
        try:
            segments = transcriber.transcribe_chunk(chunk, language=language)
            for seg in segments:
                result_q.put_nowait(seg)
        except Exception:
            logger.exception("Transcription error")


async def _processing_loop(ws: WebSocket, result_q: asyncio.Queue) -> None:
    """Async loop that reads transcription results and sends them over WS."""
    while True:
        seg: Optional[TranscriptionSegment] = await result_q.get()
        if seg is None:
            break
        payload = {
            "type": "transcript",
            "text": seg.text,
            "language": seg.language,
            "is_partial": seg.is_partial,
        }
        session.transcript_segments.append(payload)
        try:
            await _send_json(ws, payload)
        except Exception:
            break


async def _start_recording(ws: WebSocket) -> None:
    if session.recording:
        return

    session.recording = True
    session.start_time = time.time()
    session.audio_queue = queue.Queue(maxsize=100)
    session.capture = AudioCapture(session.audio_queue)
    session.capture.start()

    transcriber.reset_context()

    result_q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Spin up a thread-based transcription worker
    lang = session.language
    threading.Thread(
        target=_transcribe_worker,
        args=(session.audio_queue, result_q, lang),
        daemon=True,
    ).start()

    # Async task to relay results → WebSocket
    session._processing_task = asyncio.create_task(_processing_loop(ws, result_q))

    await _send_status(ws)
    logger.info("Recording started")


async def _stop_recording(ws: WebSocket) -> None:
    if not session.recording:
        return

    session.recording = False

    if session.capture and session.capture.is_running:
        session.capture.stop()

    # Wait for the processing loop to finish draining
    if session._processing_task:
        try:
            await asyncio.wait_for(session._processing_task, timeout=10.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            logger.warning("Processing loop timed out — cancelling")
            session._processing_task.cancel()
        except Exception:
            logger.exception("Error waiting for processing loop")
        session._processing_task = None

    try:
        await _send_status(ws)
    except Exception:
        logger.debug("Could not send status after stop (client may have disconnected)")
    logger.info("Recording stopped")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    logger.info("WebSocket client connected")

    # Send initial status
    await _send_status(ws)

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            action = msg.get("action")

            if action == "start":
                await _start_recording(ws)

            elif action == "stop":
                await _stop_recording(ws)

            elif action == "set_language":
                lang = msg.get("language", "auto")
                session.language = lang
                logger.info("Language set to %s", lang)
                await _send_status(ws)

            elif action == "save":
                text = "\n".join(s["text"] for s in session.transcript_segments)
                await _send_json(ws, {"type": "save", "text": text})

            elif action == "clear":
                session.transcript_segments.clear()
                transcriber.reset_context()
                await _send_json(ws, {"type": "cleared"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        if session.recording:
            await _stop_recording(ws)


# ── Mount static frontend (must be last so it doesn't shadow API routes) ────

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
