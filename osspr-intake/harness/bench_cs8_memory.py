#!/usr/bin/env python3
"""Peak traced memory for cs8_to_complex on one HackRF waterfall frame.

hackrf_transfer is signed 8-bit I/Q. At 20 Msps and the waterfall default
of 25 fps, one frame is 800_000 complex samples (1_600_000 bytes).
tracemalloc is started after the input exists, so the peak is the call.
"""

from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.waterfall_fft import cs8_to_complex  # noqa: E402


def _peak(raw: bytes) -> int:
    tracemalloc.start()
    cs8_to_complex(raw)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main() -> None:
    raw = bytes(1_600_000)
    _peak(raw)  # warmup
    runs = [_peak(raw) for _ in range(5)]
    runs.sort()
    median = runs[len(runs) // 2]
    noise = (runs[-1] - runs[0]) / median * 100.0 if median else 0.0
    print(f"pairs={len(raw) // 2}")
    print(f"runs_bytes={','.join(str(v) for v in runs)}")
    print(f"baseline_value={median}")
    print(f"noise_pct={noise:.3f}")


if __name__ == "__main__":
    main()
