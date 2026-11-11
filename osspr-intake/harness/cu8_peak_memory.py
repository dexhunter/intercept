"""Peak-memory baseline for utils.waterfall_fft.cu8_to_complex.

Workload: one listening-post waterfall frame of HackRF CU8 at 20 Msps and
25 fps. fft_reader sets timeslice_samples = sample_rate / fps, so this is
800_000 I/Q pairs (1_600_000 bytes) passed to cu8_to_complex in one call.
"""

from __future__ import annotations

import gc
import sys
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from utils.waterfall_fft import cu8_to_complex

RUNS = 7
N_SAMPLES = 20_000_000 // 25  # 800_000 complex samples


def _raw() -> bytes:
    rng = np.random.default_rng(20261111)
    return rng.integers(0, 256, size=N_SAMPLES * 2, dtype=np.uint8).tobytes()


def main() -> None:
    raw = _raw()
    peaks: list[int] = []
    for _ in range(RUNS):
        gc.collect()
        tracemalloc.start()
        samples = cu8_to_complex(raw)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if samples.shape != (N_SAMPLES,) or samples.dtype != np.complex64:
            raise SystemExit(f"unexpected result {samples.shape} {samples.dtype}")
        del samples
        peaks.append(int(peak))

    peaks.sort()
    median = float(peaks[len(peaks) // 2])
    spread = (max(peaks) - min(peaks)) / median * 100.0 if median else 0.0
    print(f"baseline_value={median:.0f}")
    print(f"baseline_noise_pct={spread:.4f}")
    print("runs_bytes=" + ",".join(str(p) for p in peaks))


if __name__ == "__main__":
    main()
