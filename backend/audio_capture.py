"""Microphone capture with ring-buffer and VAD pre-filtering."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from backend.config import AppConfig, config as default_config

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures microphone audio in fixed-length chunks via sounddevice.

    Chunks are placed on an :class:`queue.Queue` for async consumption.
    Silero VAD (via faster-whisper) is *not* run here — it is left to the
    transcriber's ``vad_filter=True`` parameter so that the model itself
    skips silence efficiently.  This class simply buffers raw audio.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        cfg: Optional[AppConfig] = None,
    ) -> None:
        self._cfg = cfg or default_config
        self._queue = audio_queue
        self._stream: Optional[sd.InputStream] = None
        self._buffer: np.ndarray = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._running = False
        self._paused = False
        self._chunk_samples = int(self._cfg.sample_rate * self._cfg.chunk_duration)

    # ---- sounddevice callback ----------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,  # noqa: ANN001
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("sounddevice status: %s", status)
        if self._paused:
            return

        # indata is (frames, channels) — flatten to 1-D mono
        mono = indata[:, 0].copy()

        with self._lock:
            self._buffer = np.concatenate([self._buffer, mono])

            # Emit full chunks
            while len(self._buffer) >= self._chunk_samples:
                chunk = self._buffer[: self._chunk_samples]
                self._buffer = self._buffer[self._chunk_samples:]
                try:
                    self._queue.put_nowait(chunk)
                except queue.Full:
                    logger.warning("Audio queue full — dropping chunk")

    # ---- public API --------------------------------------------------------

    def start(self) -> None:
        """Open the microphone stream and begin capturing."""
        if self._running:
            logger.warning("AudioCapture already running")
            return

        self._buffer = np.array([], dtype=np.float32)
        self._stream = sd.InputStream(
            samplerate=self._cfg.sample_rate,
            channels=self._cfg.channels,
            dtype=self._cfg.dtype,
            callback=self._audio_callback,
            blocksize=int(self._cfg.sample_rate * 0.1),  # 100 ms blocks
        )
        self._stream.start()
        self._running = True
        self._paused = False
        logger.info("Microphone capture started (rate=%d Hz)", self._cfg.sample_rate)

    def stop(self) -> None:
        """Stop microphone capture and flush any remaining audio."""
        if not self._running:
            return

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Flush leftover audio in the buffer
        with self._lock:
            if len(self._buffer) > 0:
                try:
                    self._queue.put_nowait(self._buffer.copy())
                except queue.Full:
                    pass
                self._buffer = np.array([], dtype=np.float32)

        # Sentinel to signal the consumer that capture is done
        self._queue.put(None)

        self._running = False
        self._paused = False
        logger.info("Microphone capture stopped")

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused
