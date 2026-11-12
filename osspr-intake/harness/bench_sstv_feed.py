#!/usr/bin/env python3
"""Baseline for SSTVImageDecoder.feed on one Scottie 1 image.

Scottie 1 is 256 lines, 428.22 ms each, at the decoder's 48 kHz rate.
feed() is what both the live audio thread and decode_file drive; here the
whole image is one call, which is the work done to finish one picture.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.sstv.image_decoder import SSTVImageDecoder  # noqa: E402
from utils.sstv.modes import SCOTTIE_1  # noqa: E402


def _audio() -> np.ndarray:
    spec = SSTVImageDecoder(SCOTTIE_1)
    n = spec._line_samples * spec._total_audio_lines
    # Pixel-band tone. The Scottie path still runs sync search and FM demod
    # on every line; the tone only makes the buffer a real SSTV-shaped signal.
    t = np.arange(n, dtype=np.float64)
    return np.sin(2.0 * np.pi * 1900.0 / spec._sample_rate * t)


def _time_feed(audio: np.ndarray) -> tuple[float, int]:
    decoder = SSTVImageDecoder(SCOTTIE_1)
    t0 = time.perf_counter()
    decoder.feed(audio)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, decoder.current_line


def main() -> None:
    audio = _audio()
    _time_feed(audio)  # warmup
    runs = []
    lines = None
    for _ in range(5):
        elapsed_ms, lines = _time_feed(audio)
        runs.append(elapsed_ms)
    runs.sort()
    median = runs[len(runs) // 2]
    noise = (runs[-1] - runs[0]) / median * 100.0
    print(f"lines={lines}")
    print(f"runs_ms={','.join(f'{v:.3f}' for v in runs)}")
    print(f"median_ms={median:.3f}")
    print(f"noise_pct={noise:.3f}")


if __name__ == "__main__":
    main()
