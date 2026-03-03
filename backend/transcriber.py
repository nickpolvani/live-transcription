"""Wrapper around faster-whisper for chunk-based transcription."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel

from backend.config import AppConfig, config as default_config

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """A single recognised text segment."""
    text: str
    language: str
    start: float = 0.0
    end: float = 0.0
    is_partial: bool = False


class Transcriber:
    """Loads a faster-whisper model once and exposes chunk-level transcription."""

    def __init__(self, cfg: Optional[AppConfig] = None) -> None:
        self._cfg = cfg or default_config
        self._model: Optional[WhisperModel] = None
        self._loaded = False
        self._previous_text: str = ""  # for condition_on_previous_text continuity

    # ---- lifecycle ---------------------------------------------------------

    def load_model(self) -> None:
        """Download / load the Whisper model. Call once at startup."""
        logger.info(
            "Loading Whisper model '%s' on %s (compute_type=%s)…",
            self._cfg.model_size,
            self._cfg.device,
            self._cfg.compute_type,
        )
        self._model = WhisperModel(
            self._cfg.model_size,
            device=self._cfg.device,
            compute_type=self._cfg.compute_type,
        )
        self._loaded = True
        logger.info("Model loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ---- transcription -----------------------------------------------------

    def transcribe_chunk(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> List[TranscriptionSegment]:
        """Transcribe a numpy audio chunk (float32, 16 kHz, mono).

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 array of audio samples at 16 kHz.
        language : str | None
            ISO language code, or ``None`` / ``"auto"`` for auto-detection.

        Returns
        -------
        list[TranscriptionSegment]
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded — call load_model() first.")

        lang = language if (language and language != "auto") else None

        segments_iter, info = self._model.transcribe(
            audio,
            beam_size=5,
            language=lang,
            vad_filter=True,
            vad_parameters={"threshold": self._cfg.vad_threshold},
            condition_on_previous_text=True,
            initial_prompt=self._previous_text[-200:] if self._previous_text else None,
        )

        detected_lang = info.language or (lang or "unknown")

        results: List[TranscriptionSegment] = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue
            results.append(
                TranscriptionSegment(
                    text=text,
                    language=detected_lang,
                    start=seg.start,
                    end=seg.end,
                )
            )
            # keep a rolling context for next chunk
            self._previous_text += " " + text

        # Prevent unbounded growth
        if len(self._previous_text) > 2000:
            self._previous_text = self._previous_text[-1000:]

        return results

    def reset_context(self) -> None:
        """Clear the rolling context (e.g. when the user stops recording)."""
        self._previous_text = ""
