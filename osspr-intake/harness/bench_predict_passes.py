#!/usr/bin/env python3
"""Baseline for utils.satellite_predict.predict_passes.

One 24 h ISS prediction from the bundled TLE, London observer, default
trajectory (30 points) and ground track (60 points). That is what
/satellite/predict and /satellite/iss/live ask for. The window is anchored
at the TLE epoch so the run does not touch the network and stays repeatable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.satellites import TLE_SATELLITES  # noqa: E402
from skyfield.api import EarthSatellite, load, wgs84  # noqa: E402

from utils.satellite_predict import predict_passes  # noqa: E402


def _setup():
    name, line1, line2 = TLE_SATELLITES["ISS"]
    ts = load.timescale()
    satellite = EarthSatellite(line1, line2, name, ts)
    t0 = satellite.epoch
    t1 = ts.tt_jd(t0.tt + 1.0)
    observer = wgs84.latlon(51.5074, -0.1278)
    tle = (name, line1, line2)
    return tle, observer, ts, t0, t1


def main() -> None:
    tle, observer, ts, t0, t1 = _setup()

    def once():
        return predict_passes(tle, observer, ts, t0, t1, min_el=10.0)

    passes = once()  # warmup
    runs = []
    count = len(passes)
    for _ in range(5):
        t_start = time.perf_counter()
        passes = once()
        runs.append((time.perf_counter() - t_start) * 1000.0)
        count = len(passes)
    runs.sort()
    median = runs[len(runs) // 2]
    noise = (runs[-1] - runs[0]) / median * 100.0
    print(f"passes={count}")
    print(f"runs_ms={','.join(f'{v:.3f}' for v in runs)}")
    print(f"median_ms={median:.3f}")
    print(f"noise_pct={noise:.3f}")


if __name__ == "__main__":
    main()
