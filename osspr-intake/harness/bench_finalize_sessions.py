#!/usr/bin/env python3
"""Baseline for DeviceIdentityEngine.finalize_all_sessions.

Called when a TSCM sweep ends (routes/tscm). 200 BLE devices, six
advertisements a second apart, is a crowded room rather than a quiet
office. Ingest is not included in the timed section.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.tscm.device_identity import BLEObservation, DeviceIdentityEngine  # noqa: E402


def _observations() -> list[BLEObservation]:
    base = datetime(2026, 11, 12, 8, 0, 0)
    observations = []
    for i in range(200):
        mac = f"{(i // 65536) % 256:02X}:{(i // 256) % 256:02X}:{i % 256:02X}:10:20:30"
        payload = bytes([i % 256, (i // 3) % 256, 7, 8, 9, 10, 11, 12])
        for k in range(6):
            observations.append(
                BLEObservation(
                    timestamp=base + timedelta(seconds=k),
                    addr=mac,
                    addr_type="public",
                    rssi=-60 - (i % 15),
                    manufacturer_id=0x0006,
                    manufacturer_data=payload,
                    service_uuids=[f"{(0x1800 + (i % 40)):04x}"],
                    local_name=f"Dev{i % 30}",
                )
            )
    return observations


def _ready(observations: list[BLEObservation]) -> DeviceIdentityEngine:
    engine = DeviceIdentityEngine()
    for obs in observations:
        engine.ingest_ble_observation(obs)
    return engine


def main() -> None:
    observations = _observations()
    warmup = _ready(observations)
    t0 = time.perf_counter()
    warmup.finalize_all_sessions()
    _ = (time.perf_counter() - t0) * 1000.0

    runs = []
    clusters = None
    for _ in range(5):
        engine = _ready(observations)
        t0 = time.perf_counter()
        engine.finalize_all_sessions()
        runs.append((time.perf_counter() - t0) * 1000.0)
        clusters = len(engine.clusters)
    runs.sort()
    median = runs[len(runs) // 2]
    noise = (runs[-1] - runs[0]) / median * 100.0
    print(f"clusters={clusters}")
    print(f"runs_ms={','.join(f'{v:.3f}' for v in runs)}")
    print(f"median_ms={median:.3f}")
    print(f"noise_pct={noise:.3f}")


if __name__ == "__main__":
    main()
