"""Benchmark all Whisper models on GPU and CPU.

Usage:
    python tests/benchmark_models.py

Outputs a table of latency results for each model x device combination.
"""

import gc
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, ".")

from backend.config import AVAILABLE_MODELS

SAMPLE_RATE = 16_000
CHUNK_DURATION = 2.0  # seconds
NUM_WARMUP = 1
NUM_RUNS = 3
MAX_LATENCY = 2.0  # seconds


def make_speech_chunk(duration: float = CHUNK_DURATION) -> np.ndarray:
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, dtype=np.float32)
    signal = (
        0.3 * np.sin(2 * np.pi * 150 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.15 * np.sin(2 * np.pi * 1000 * t)
        + 0.1 * np.sin(2 * np.pi * 2500 * t)
    )
    rng = np.random.default_rng(42)
    signal += 0.02 * rng.standard_normal(samples).astype(np.float32)
    return signal


def detect_devices() -> list:
    """Return list of (device, compute_type) tuples to test."""
    devices = [("cpu", "int8")]
    try:
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types("cuda")
        if "float16" in supported:
            devices.append(("cuda", "float16"))
        elif "int8" in supported:
            devices.append(("cuda", "int8"))
    except Exception:
        pass
    return devices


def benchmark_model(model_size: str, device: str, compute_type: str, audio: np.ndarray):
    """Load model, warm up, and measure inference latency. Returns (load_time, avg_latency, timings)."""
    from faster_whisper import WhisperModel

    # Load
    t0 = time.perf_counter()
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        return None, None, str(e)
    load_time = time.perf_counter() - t0

    # Warmup (vad_filter=False to force actual inference on synthetic audio)
    for _ in range(NUM_WARMUP):
        segments, _ = model.transcribe(audio, beam_size=5, language="en", vad_filter=False)
        for _ in segments:
            pass

    # Benchmark
    timings = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        segments, info = model.transcribe(audio, beam_size=5, language="en", vad_filter=False)
        for _ in segments:
            pass
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

    avg = sum(timings) / len(timings)

    # Cleanup
    del model
    gc.collect()

    return load_time, avg, timings


def main():
    devices = detect_devices()
    audio = make_speech_chunk(CHUNK_DURATION)

    print(f"\nBenchmark: {CHUNK_DURATION}s audio chunk, {NUM_RUNS} runs (after {NUM_WARMUP} warmup)")
    print(f"Max allowed latency: {MAX_LATENCY}s")
    print(f"Devices: {', '.join(f'{d}({ct})' for d, ct in devices)}")
    print()

    header = f"{'Model':<12} {'Device':<14} {'Load (s)':>10} {'Avg (s)':>10} {'Min (s)':>10} {'Max (s)':>10} {'Pass?':>6}"
    print(header)
    print("-" * len(header))

    results = []

    for model_size in AVAILABLE_MODELS:
        for device, compute_type in devices:
            label = f"{device}({compute_type})"
            sys.stdout.write(f"{model_size:<12} {label:<14} ")
            sys.stdout.flush()

            load_time, avg, timings = benchmark_model(model_size, device, compute_type, audio)

            if load_time is None:
                print(f"{'ERROR':>10} {str(timings)}")
                results.append((model_size, device, compute_type, None, None, False))
                continue

            mn = min(timings)
            mx = max(timings)
            passed = avg < MAX_LATENCY
            mark = "YES" if passed else "NO"

            print(f"{load_time:10.2f} {avg:10.3f} {mn:10.3f} {mx:10.3f} {mark:>6}")
            results.append((model_size, device, compute_type, avg, load_time, passed))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Models within 2s latency budget:")
    print("=" * 60)
    for model_size, device, ct, avg, load_t, passed in results:
        if passed:
            print(f"  OK  {model_size:<12} {device}({ct})  avg={avg:.3f}s  load={load_t:.1f}s")
    print()
    failed = [r for r in results if not r[5]]
    if failed:
        print("Models EXCEEDING 2s budget:")
        for model_size, device, ct, avg, load_t, passed in failed:
            if avg is not None:
                print(f"  FAIL  {model_size:<12} {device}({ct})  avg={avg:.3f}s")
            else:
                print(f"  ERR   {model_size:<12} {device}({ct})  could not load")
    print()


if __name__ == "__main__":
    main()
