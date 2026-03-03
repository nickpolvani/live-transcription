"""Application configuration for live transcription."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "fr": "French",
    "it": "Italian",
}


def _default_model_size() -> str:
    """Allow override via WHISPER_MODEL env var (useful for testing)."""
    return os.environ.get("WHISPER_MODEL", "base")


def _detect_device() -> str:
    """Return 'cuda' if an NVIDIA GPU with CUDA is available, else 'cpu'."""
    try:
        import ctranslate2

        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            logger.info("CUDA device detected via ctranslate2")
            return "cuda"
    except Exception:
        pass

    # Fallback: try torch if available
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA device detected via torch: %s", torch.cuda.get_device_name(0))
            return "cuda"
    except ImportError:
        pass

    logger.info("No CUDA device found — falling back to CPU")
    return "cpu"


def _compute_type_for_device(device: str) -> str:
    """Return the optimal compute type for the given device."""
    return "float16" if device == "cuda" else "int8"


@dataclass
class AppConfig:
    """Central configuration for the live-transcription application."""

    # Model settings
    model_size: str = field(default_factory=_default_model_size)
    device: str = field(default_factory=_detect_device)
    compute_type: Optional[str] = None  # auto-resolved from device if None
    language: str = "auto"  # one of SUPPORTED_LANGUAGES keys

    # Audio settings
    sample_rate: int = 16_000  # 16 kHz
    channels: int = 1  # mono
    dtype: str = "float32"
    chunk_duration: float = 2.0  # seconds per chunk

    # Decoding settings
    beam_size: int = 1  # beam_size=1 (greedy) is ~2x faster than 5; quality tradeoff is small

    # VAD settings
    vad_threshold: float = 0.5  # Silero VAD speech probability threshold

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8080

    def __post_init__(self) -> None:
        if self.model_size not in AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model_size '{self.model_size}'. "
                f"Choose from: {AVAILABLE_MODELS}"
            )
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Invalid language '{self.language}'. "
                f"Choose from: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        if self.compute_type is None:
            self.compute_type = _compute_type_for_device(self.device)

    # ---- helpers exposed to the API ----

    @property
    def available_models(self) -> List[str]:
        return AVAILABLE_MODELS

    @property
    def supported_languages(self) -> dict:
        return SUPPORTED_LANGUAGES

    def to_dict(self) -> dict:
        """Serialise configuration for the /api/config endpoint."""
        return {
            "model_info": {
                "size": self.model_size,
                "device": self.device,
                "compute_type": self.compute_type,
                "loaded": False,  # updated by transcriber at runtime
            },
            "languages": self.supported_languages,
            "model_sizes": self.available_models,
            "audio": {
                "sample_rate": self.sample_rate,
                "chunk_duration": self.chunk_duration,
            },
        }


# Singleton config — importable everywhere
config = AppConfig()
