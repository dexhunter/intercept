#!/usr/bin/env python3
"""Baseline for SubGhzManager._fingerprint_burst_bytes.

The RX loop fingerprints each burst with the bytes it held, capped at
sample_rate * 2 bytes * 2 seconds. Default receive rate is 2 Msps, so the
cap is 8_000_000 bytes: a 2 s OOK/FSK capture. This times that one call.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.subghz import SubGhzManager  # noqa: E402


def _burst() -> bytes:
    sample_rate = 2_000_000
    n = sample_rate * 2
    rng = np.random.default_rng(0)
    i_vals = rng.normal(0, 3, n)
    q_vals = rng.normal(0, 3, n)
    # ~200 us on / 800 us off, a long keyed burst rather than quiet IQ.
    carrier_on = (np.arange(n) % 400) < 80
    i_vals = np.where(carrier_on, rng.normal(40, 5, n), i_vals)
    q_vals = np.where(carrier_on, rng.normal(0, 5, n), q_vals)
    raw = np.empty(n * 2, dtype=np.int8)
    raw[0::2] = i_vals.astype(np.int8)
    raw[1::2] = q_vals.astype(np.int8)
    return raw.tobytes()


def main() -> None:
    data = _burst()
    manager = SubGhzManager(data_dir=Path("/tmp/osspr-subghz-bench"))

    def once() -> str:
        return manager._fingerprint_burst_bytes(data, 2_000_000, 2.0)

    fingerprint = once()  # warmup
    runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        fingerprint = once()
        runs.append((time.perf_counter() - t0) * 1000.0)
    runs.sort()
    median = runs[len(runs) // 2]
    noise = (runs[-1] - runs[0]) / median * 100.0
    print(f"bytes={len(data)}")
    print(f"fingerprint={fingerprint}")
    print(f"runs_ms={','.join(f'{v:.3f}' for v in runs)}")
    print(f"median_ms={median:.3f}")
    print(f"noise_pct={noise:.3f}")


if __name__ == "__main__":
    main()
