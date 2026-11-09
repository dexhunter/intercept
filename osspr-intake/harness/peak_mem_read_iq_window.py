#!/usr/bin/env python3
"""Baseline peak RSS for utils.subghz_inspect.read_iq_window.

Workload: the longest window offline Sub-GHz inspection will load, 2.0
seconds of a HackRF cs8 capture at the mode's default 2_000_000 samples/s
(8 MB on disk, 4e6 complex samples). Each run is a fresh process so
ru_maxrss is the high-water mark of importing numpy and decoding that
window, not a leftover from an earlier call.

On Linux ru_maxrss is kilobytes. The printed value is bytes.
baseline noise is the full spread (max-min)/median over 5 processes.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SAMPLE_RATE = 2_000_000
DURATION_S = 2.0
RUNS = 5
CHILD_FLAG = "OSSPR_IQ_CHILD"


def write_capture(path: Path) -> None:
    import numpy as np

    n = int(SAMPLE_RATE * DURATION_S)
    t = np.arange(n, dtype=np.float64)
    i = np.rint(np.sin(2.0 * np.pi * t / 80.0) * 100.0).astype(np.int8)
    q = np.rint(np.cos(2.0 * np.pi * t / 80.0) * 100.0).astype(np.int8)
    raw = np.empty(n * 2, dtype=np.int8)
    raw[0::2] = i
    raw[1::2] = q
    path.write_bytes(raw.tobytes())


def child(path: Path) -> int:
    from utils.subghz_inspect import read_iq_window

    iq = read_iq_window(path, SAMPLE_RATE, 0.0, DURATION_S)
    # Touch both ends so the result stays live until the high-water mark is read.
    _ = float(iq.real[0] + iq.imag[-1])
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(peak_kb * 1024)
    return 0


def main() -> None:
    if os.environ.get(CHILD_FLAG) == "1":
        child(Path(os.environ["OSSPR_IQ_PATH"]))
        return

    fd, capture_name = tempfile.mkstemp(prefix="osspr-iq-", suffix=".cs8")
    os.close(fd)
    capture = Path(capture_name)
    try:
        write_capture(capture)
        peaks: list[int] = []
        env = os.environ.copy()
        env[CHILD_FLAG] = "1"
        env["OSSPR_IQ_PATH"] = str(capture)
        for _ in range(RUNS):
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve())],
                cwd=str(ROOT),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            peaks.append(int(result.stdout.strip().splitlines()[-1]))
    finally:
        capture.unlink(missing_ok=True)

    ordered = sorted(peaks)
    median = ordered[len(ordered) // 2]
    noise_pct = (max(peaks) - min(peaks)) / median * 100.0
    print(f"median_bytes={median}")
    print(f"noise_pct={noise_pct:.4f}")
    print("runs_bytes=" + ",".join(str(p) for p in peaks))


if __name__ == "__main__":
    main()
