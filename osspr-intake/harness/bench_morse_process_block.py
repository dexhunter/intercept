#!/usr/bin/env python3
"""Baseline wall time for utils.morse.MorseDecoder.process_block.

Workload: 60 seconds of 8 kHz, 16-bit CW (700 Hz tone, keyed about 3 times
a second) passed to process_block in one call, repeated four times per
sample so the spread is not dominated by a single scheduling blip. Each
call is one recorded exchange, not the 2048-sample chunks the live reader
and decode_morse_wav_file use. Tone tracking is off and WPM is manual so
the timed work is the block loop itself.

process_block unpacks the entire buffer into a Python list, then on every
160-sample block copies a prefix out and deletes it. Deleting the front of
the list shifts the tail, so the cost grows with the square of the
recording. The harness times only process_block.

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

from utils.morse import MorseDecoder  # noqa: E402

SAMPLE_RATE = 8000
DURATION_S = 60
TONE_HZ = 700.0
RUNS = 5
REPEATS_PER_RUN = 4


def make_pcm() -> bytes:
    n = SAMPLE_RATE * DURATION_S
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    # On/off keying around a few elements per second, so the state machine
    # runs but the samples stay a plain recorded tone.
    envelope = np.sin(2.0 * np.pi * 3.0 * t) > 0.0
    audio = 0.6 * np.sin(2.0 * np.pi * TONE_HZ * t) * envelope
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def new_decoder() -> MorseDecoder:
    return MorseDecoder(
        sample_rate=SAMPLE_RATE,
        tone_freq=TONE_HZ,
        wpm=15,
        auto_tone_track=False,
        tone_lock=True,
        wpm_mode="manual",
        wpm_lock=True,
    )


def run_once(pcm: bytes) -> int:
    events = new_decoder().process_block(pcm)
    if not events:
        return 0
    amplitudes = events[-1].get("amplitudes") or []
    return len(events) + len(amplitudes)


def main() -> None:
    pcm = make_pcm()
    gc.collect()
    gc.disable()
    try:
        run_once(pcm)  # warmup
        run_once(pcm)
        times_ms: list[float] = []
        checksum = 0
        for _ in range(RUNS):
            started = time.perf_counter()
            for _repeat in range(REPEATS_PER_RUN):
                events = new_decoder().process_block(pcm)
            times_ms.append((time.perf_counter() - started) * 1000.0)
            amplitudes = events[-1].get("amplitudes") or [] if events else []
            checksum += len(events) + len(amplitudes)
    finally:
        gc.enable()
    ordered = sorted(times_ms)
    median = ordered[len(ordered) // 2]
    noise_pct = (max(times_ms) - min(times_ms)) / median * 100.0
    print(f"median_ms={median:.4f}")
    print(f"noise_pct={noise_pct:.4f}")
    print("runs_ms=" + ",".join(f"{t:.4f}" for t in times_ms))
    print(f"checksum={checksum}")


if __name__ == "__main__":
    main()
