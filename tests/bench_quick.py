"""Quick benchmark: all models x CPU, with vad_filter=False for real inference."""

import gc
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from faster_whisper import WhisperModel

SR = 16_000
DUR = 2.0
RUNS = 3

# Generate speech-like audio
samples = int(SR * DUR)
t = np.linspace(0, DUR, samples, dtype=np.float32)
audio = (
    0.3 * np.sin(2 * np.pi * 150 * t)
    + 0.2 * np.sin(2 * np.pi * 400 * t)
    + 0.15 * np.sin(2 * np.pi * 1000 * t)
    + 0.1 * np.sin(2 * np.pi * 2500 * t)
)
audio += 0.02 * np.random.default_rng(42).standard_normal(samples).astype(np.float32)

MODELS = ["tiny", "base", "small", "medium"]

DEVICES = [("cpu", "int8")]

# To add GPU benchmarks, uncomment:
# DEVICES.append(("cuda", "float16"))

print(f"\nBenchmark: {DUR}s audio chunk, {RUNS} runs (+ 1 warmup), vad_filter=False")
print(f"Devices: {DEVICES}\n")

header = f"{'Model':<12} {'Device':<16} {'Load(s)':>8} {'Run1':>8} {'Run2':>8} {'Run3':>8} {'Avg':>8} {'<2s?':>5}"
print(header)
print("-" * len(header))

for model_name in MODELS:
    for device, ct in DEVICES:
        tag = f"{device}({ct})"
        # Load
        t0 = time.perf_counter()
        try:
            m = WhisperModel(model_name, device=device, compute_type=ct)
        except Exception as e:
            print(f"{model_name:<12} {tag:<16} LOAD ERROR: {e}")
            continue
        load_t = time.perf_counter() - t0

        # Warmup
        segs, _ = m.transcribe(audio, beam_size=5, language="en", vad_filter=False)
        for _ in segs:
            pass

        # Timed runs
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            segs, _ = m.transcribe(audio, beam_size=5, language="en", vad_filter=False)
            for _ in segs:
                pass
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        ok = "YES" if avg < 2.0 else "NO"
        print(
            f"{model_name:<12} {tag:<16} {load_t:8.2f} "
            f"{times[0]:8.3f} {times[1]:8.3f} {times[2]:8.3f} "
            f"{avg:8.3f} {ok:>5}"
        )

        del m
        gc.collect()

print("\nDone.")
