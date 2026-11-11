"""Runtime baseline for DeviceCorrelator.correlate.

Workload: one TSCM refresh over 200 Wi-Fi and 200 Bluetooth devices, the
size of a busy office or conference scan. The two populations are hours
apart, so nothing is persisted, but correlate still parses and scores
every pair. That is the cost of a scan that does not find a match.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.correlation import DeviceCorrelator

RUNS = 7
N_DEVICES = 200


def _devices() -> tuple[dict, dict]:
    wifi: dict = {}
    bluetooth: dict = {}
    wifi_base = datetime(2026, 1, 1, 9, 0, 0)
    bt_base = datetime(2026, 1, 1, 15, 0, 0)
    for i in range(N_DEVICES):
        wifi[f"02:00:{i:02x}:10:20:30"] = {
            "first_seen": (wifi_base + timedelta(seconds=i)).isoformat(),
            "last_seen": (wifi_base + timedelta(seconds=i, milliseconds=800)).isoformat(),
            "rssi": -40 - (i % 40),
            "name": f"ap-{i}",
            "manufacturer": "WifiVendor",
        }
        bluetooth[f"02:AA:{i:02x}:10:20:30"] = {
            "first_seen": (bt_base + timedelta(seconds=i)).isoformat(),
            "last_seen": (bt_base + timedelta(seconds=i, milliseconds=800)).isoformat(),
            "rssi": -48 - (i % 35),
            "name": f"phone-{i}",
            "manufacturer": "BtVendor",
        }
    return wifi, bluetooth


def main() -> None:
    wifi, bluetooth = _devices()
    correlator = DeviceCorrelator()
    warmup = correlator.correlate(wifi, bluetooth)
    if any(row["confidence"] >= 0.7 for row in warmup):
        raise SystemExit("workload persisted correlations; timings would include sqlite")

    elapsed_ms: list[float] = []
    for _ in range(RUNS):
        started = time.perf_counter()
        rows = correlator.correlate(wifi, bluetooth)
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
        if len(rows) != len(warmup):
            raise SystemExit("correlate result count changed between runs")

    ordered = sorted(elapsed_ms)
    median = ordered[len(ordered) // 2]
    spread = (max(elapsed_ms) - min(elapsed_ms)) / median * 100.0 if median else 0.0
    print(f"baseline_median_ms={median:.4f}")
    print(f"baseline_noise_pct={spread:.4f}")
    print(f"result_count={len(warmup)}")
    print("runs_ms=" + ",".join(f"{v:.4f}" for v in elapsed_ms))


if __name__ == "__main__":
    main()
