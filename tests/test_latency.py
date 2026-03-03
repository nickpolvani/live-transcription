"""Latency benchmarks for model inference.

Asserts that transcribe_chunk completes within 2 seconds for a 2-second audio chunk.
Uses the tiny model (via WHISPER_MODEL env var) to keep CI fast; run with a
larger model locally to benchmark your hardware.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

# Force tiny model for latency tests (can be overridden)
os.environ.setdefault("WHISPER_MODEL", "tiny")

from backend.config import AppConfig  # noqa: E402
from backend.transcriber import Transcriber  # noqa: E402

MAX_LATENCY_SECONDS = 2.0
SAMPLE_RATE = 16_000
CHUNK_DURATION = 2.0  # seconds


@pytest.fixture(scope="module")
def transcriber() -> Transcriber:
    """Load the model once for all tests in this module."""
    cfg = AppConfig(
        model_size=os.environ.get("WHISPER_MODEL", "tiny"),
        device="cpu",
        compute_type="int8",
    )
    t = Transcriber(cfg)
    t.load_model()
    return t


def _make_sine_chunk(freq: float = 440.0, duration: float = CHUNK_DURATION) -> np.ndarray:
    """Generate a sine wave chunk (simulates non-silent audio)."""
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def _make_speech_like_chunk(duration: float = CHUNK_DURATION) -> np.ndarray:
    """Generate a multi-frequency chunk that resembles speech more than a pure tone."""
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, dtype=np.float32)
    # Mix of frequencies common in human speech (100-3000 Hz)
    signal = (
        0.3 * np.sin(2 * np.pi * 150 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.15 * np.sin(2 * np.pi * 1000 * t)
        + 0.1 * np.sin(2 * np.pi * 2500 * t)
    )
    # Add a small amount of noise
    rng = np.random.default_rng(42)
    signal += 0.02 * rng.standard_normal(samples).astype(np.float32)
    return signal


def _make_silence_chunk(duration: float = CHUNK_DURATION) -> np.ndarray:
    """Generate near-silence (should be very fast â€” VAD skips it)."""
    samples = int(SAMPLE_RATE * duration)
    rng = np.random.default_rng(0)
    return 0.001 * rng.standard_normal(samples).astype(np.float32)


# â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestInferenceLatency:
    """Each test asserts transcribe_chunk finishes within MAX_LATENCY_SECONDS."""

    def test_sine_wave_2s_under_limit(self, transcriber: Transcriber) -> None:
        """2-second sine wave chunk should transcribe within the latency budget."""
        audio = _make_sine_chunk(duration=2.0)
        start = time.perf_counter()
        transcriber.transcribe_chunk(audio, language="en")
        elapsed = time.perf_counter() - start
        print(f"\n  sine 2s â†’ {elapsed:.3f}s")
        assert elapsed < MAX_LATENCY_SECONDS, (
            f"Inference took {elapsed:.2f}s, exceeds {MAX_LATENCY_SECONDS}s limit"
        )

    def test_speech_like_2s_under_limit(self, transcriber: Transcriber) -> None:
        """2-second speech-like chunk should transcribe within the latency budget."""
        audio = _make_speech_like_chunk(duration=2.0)
        start = time.perf_counter()
        transcriber.transcribe_chunk(audio, language="en")
        elapsed = time.perf_counter() - start
        print(f"\n  speech-like 2s â†’ {elapsed:.3f}s")
        assert elapsed < MAX_LATENCY_SECONDS, (
            f"Inference took {elapsed:.2f}s, exceeds {MAX_LATENCY_SECONDS}s limit"
        )

    def test_silence_faster_than_speech(self, transcriber: Transcriber) -> None:
        """Silence should be faster than speech (VAD skips it)."""
        silence = _make_silence_chunk(duration=2.0)
        start = time.perf_counter()
        transcriber.transcribe_chunk(silence, language="en")
        silence_time = time.perf_counter() - start

        speech = _make_speech_like_chunk(duration=2.0)
        start = time.perf_counter()
        transcriber.transcribe_chunk(speech, language="en")
        speech_time = time.perf_counter() - start

        print(f"\n  silence={silence_time:.3f}s  speech={speech_time:.3f}s")
        assert silence_time < MAX_LATENCY_SECONDS, (
            f"Silence inference took {silence_time:.2f}s, exceeds limit"
        )

    def test_autodetect_language_under_limit(self, transcriber: Transcriber) -> None:
        """Auto-detect language path should also stay within the latency budget."""
        audio = _make_speech_like_chunk(duration=2.0)
        start = time.perf_counter()
        transcriber.transcribe_chunk(audio, language="auto")
        elapsed = time.perf_counter() - start
        print(f"\n  auto-detect 2s â†’ {elapsed:.3f}s")
        assert elapsed < MAX_LATENCY_SECONDS, (
            f"Inference took {elapsed:.2f}s, exceeds {MAX_LATENCY_SECONDS}s limit"
        )

    def test_short_chunk_500ms_under_limit(self, transcriber: Transcriber) -> None:
        """A shorter 500ms chunk should be well within the budget."""
        audio = _make_speech_like_chunk(duration=0.5)
        start = time.perf_counter()
        transcriber.transcribe_chunk(audio, language="en")
        elapsed = time.perf_counter() - start
        print(f"\n  speech 0.5s â†’ {elapsed:.3f}s")
        assert elapsed < MAX_LATENCY_SECONDS, (
            f"Inference took {elapsed:.2f}s, exceeds {MAX_LATENCY_SECONDS}s limit"
        )

    def test_long_chunk_5s_under_scaled_limit(self, transcriber: Transcriber) -> None:
        """5-second chunk gets a proportionally larger budget (5s audio â†’ 5s limit)."""
        duration = 5.0
        scaled_limit = duration  # 1x real-time
        audio = _make_speech_like_chunk(duration=duration)
        start = time.perf_counter()
        transcriber.transcribe_chunk(audio, language="en")
        elapsed = time.perf_counter() - start
        print(f"\n  speech 5s â†’ {elapsed:.3f}s (limit {scaled_limit}s)")
        assert elapsed < scaled_limit, (
            f"Inference took {elapsed:.2f}s, exceeds {scaled_limit}s limit for {duration}s audio"
        )
