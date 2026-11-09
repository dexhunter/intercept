#!/usr/bin/env python3
"""Baseline wall time for utils.wefax._goertzel_mag.

Workload: ten minutes of HF weather-fax scanning at the decoder's 22050 Hz
rate. The live scanner reads 100 ms chunks and, while looking for the
300 Hz start tone, asks _goertzel_mag for that tone plus the four reference
tones in _detect_tone. This harness repeats that call pattern 6000 times
(600 seconds of audio) on one realistic chunk: a strong 300 Hz tone with a
weak 1900 Hz carrier. The chunk is reused so the timed work is the
recurrence, not sample generation.

Prints median milliseconds and the full spread (max-min)/median over 5
timed runs, after two warmups. Garbage collection is off during the runs
so a collector pause is not part of the spread.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.wefax import _goertzel_mag  # noqa: E402

SAMPLE_RATE = 22050
CHUNK_SECONDS = 0.1
SCAN_SECONDS = 600
# Same reference set _detect_tone uses when the target is the 300 Hz start tone.
FREQUENCIES = (300.0, 1000.0, 1500.0, 1900.0, 2300.0)
RUNS = 5


def make_chunk() -> np.ndarray:
    n = int(SAMPLE_RATE * CHUNK_SECONDS)  # 2205 samples, the live read size
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    return 0.8 * np.sin(2.0 * np.pi * 300.0 * t) + 0.05 * np.sin(2.0 * np.pi * 1900.0 * t)


def scan_one_minute(chunk: np.ndarray) -> float:
    n_chunks = int(SCAN_SECONDS / CHUNK_SECONDS)
    acc = 0.0
    for _ in range(n_chunks):
        for freq in FREQUENCIES:
            acc += _goertzel_mag(chunk, freq, SAMPLE_RATE)
    return acc


def main() -> None:
    chunk = make_chunk()
    gc.collect()
    gc.disable()
    try:
        scan_one_minute(chunk)  # warmup (twice: first pass pays import/cache misses)
        scan_one_minute(chunk)
        times_ms: list[float] = []
        checksum = 0.0
        for _ in range(RUNS):
            started = time.perf_counter()
            checksum += scan_one_minute(chunk)
            times_ms.append((time.perf_counter() - started) * 1000.0)
    finally:
        gc.enable()
    ordered = sorted(times_ms)
    median = ordered[len(ordered) // 2]
    noise_pct = (max(times_ms) - min(times_ms)) / median * 100.0
    print(f"median_ms={median:.4f}")
    print(f"noise_pct={noise_pct:.4f}")
    print("runs_ms=" + ",".join(f"{t:.4f}" for t in times_ms))
    print(f"checksum={checksum:.6f}")


if __name__ == "__main__":
    main()
