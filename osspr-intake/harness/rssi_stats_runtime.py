"""Runtime baseline for DeviceAggregator._update_rssi_stats.

Workload: the steady-state RSSI buffer. After a few minutes of scanning,
each device holds MAX_RSSI_SAMPLES (300) observations and every new
advertisement recomputes min/max/median/variance/confidence over that
whole buffer. One run is 800 such updates, about one second of advertisements
from ~80 nearby devices at 10 Hz, or one busy device's updates over 80 s.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from utils.bluetooth.aggregator import DeviceAggregator
from utils.bluetooth.constants import MAX_RSSI_SAMPLES
from utils.bluetooth.models import BTDeviceAggregate

RUNS = 7
UPDATES = 800


def _device() -> tuple[DeviceAggregator, BTDeviceAggregate]:
    rng = np.random.default_rng(20261111)
    values = rng.integers(-90, -30, size=MAX_RSSI_SAMPLES)
    start = datetime(2026, 6, 1, 12, 0, 0)
    device = BTDeviceAggregate(
        device_id="AA:BB:CC:DD:EE:FF:public",
        address="AA:BB:CC:DD:EE:FF",
        address_type="public",
    )
    device.rssi_samples = [
        (start + timedelta(milliseconds=200 * i), int(values[i])) for i in range(MAX_RSSI_SAMPLES)
    ]
    return DeviceAggregator(), device


def main() -> None:
    aggregator, device = _device()
    # Warm the statistics path once so the timed runs are the steady state.
    aggregator._update_rssi_stats(device)

    elapsed_ms: list[float] = []
    for _ in range(RUNS):
        started = time.perf_counter()
        for _update in range(UPDATES):
            aggregator._update_rssi_stats(device)
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)

    ordered = sorted(elapsed_ms)
    median = ordered[len(ordered) // 2]
    spread = (max(elapsed_ms) - min(elapsed_ms)) / median * 100.0 if median else 0.0
    print(f"baseline_median_ms={median:.4f}")
    print(f"baseline_noise_pct={spread:.4f}")
    print("runs_ms=" + ",".join(f"{v:.4f}" for v in elapsed_ms))


if __name__ == "__main__":
    main()
