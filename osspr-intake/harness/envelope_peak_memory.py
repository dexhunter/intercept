"""Peak-memory baseline for utils.subghz_inspect.envelope_columns.

Workload: the largest inspect window the Sub-GHz inspector will plot.
inspect_window decimates a capture to about 1 MS/s and caps the window at
2 seconds, so envelope_columns sees 2_000_000 complex samples.
"""

from __future__ import annotations

import gc
import sys
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from utils.subghz_inspect import envelope_columns

RUNS = 7
N_SAMPLES = 2_000_000


def _iq() -> np.ndarray:
    rng = np.random.default_rng(20261111)
    real = rng.standard_normal(N_SAMPLES)
    imag = rng.standard_normal(N_SAMPLES)
    return (real + 1j * imag).astype(np.complex64)


def main() -> None:
    iq = _iq()
    peaks: list[int] = []
    for _ in range(RUNS):
        gc.collect()
        tracemalloc.start()
        columns, per_column = envelope_columns(iq)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if not columns or per_column < 1:
            raise SystemExit("envelope_columns returned an empty column list")
        del columns
        peaks.append(int(peak))

    peaks.sort()
    median = float(peaks[len(peaks) // 2])
    spread = (max(peaks) - min(peaks)) / median * 100.0 if median else 0.0
    print(f"baseline_value={median:.0f}")
    print(f"baseline_noise_pct={spread:.4f}")
    print("runs_bytes=" + ",".join(str(p) for p in peaks))


if __name__ == "__main__":
    main()
