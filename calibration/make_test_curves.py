"""Test curves of known fractal dimension, for calibrating the pipeline.

Writes two open curves as ``latlon`` text files, placed at the latitude of the
Buzios example and matched to it in size (about 7 km between the end points and
about 38-39 km along the curve):

    koch.txt  Koch curve, level 6 (4097 vertices, 9.6 m segments), D = log 4 / log 3
    sine.txt  three periods of a sinusoid, 4001 vertices, D = 1 (a smooth curve)

Usage:  python calibration/make_test_curves.py [output_dir]   (default: calibration)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EARTH_RADIUS_KM = 6371.0
LAT0_DEG, LON0_DEG = -22.77, -41.90     # near the Buzios example
BASE_KM = 7.0                           # distance between the end points
TARGET_LENGTH_KM = 38.2                 # length of the sinusoid, to match Buzios


def koch(level: int, base: float) -> np.ndarray:
    """Planar Koch curve from (0, 0) to (base, 0), shape (4**level + 1, 2), km."""
    p = np.array([[0.0, 0.0], [base, 0.0]])
    c, s = 0.5, np.sqrt(3.0) / 2.0
    rot = np.array([[c, -s], [s, c]])                  # +60 degrees
    for _ in range(level):
        a, b = p[:-1], p[1:]
        third = (b - a) / 3.0
        p1, p3 = a + third, a + 2.0 * third
        p2 = p1 + third @ rot.T
        p = np.vstack([np.stack([a, p1, p2, p3], axis=1).reshape(-1, 2), p[-1:]])
    return p


def sine(base: float, periods: int, target_length: float, n: int = 4001) -> np.ndarray:
    """Planar sinusoid from (0, 0) to (base, 0) whose length reaches target_length."""
    t = np.linspace(0.0, 1.0, n)
    for amp in np.linspace(0.1, 10.0, 991):
        p = np.column_stack([base * t, amp * np.sin(2.0 * np.pi * periods * t)])
        if np.hypot(*np.diff(p, axis=0).T).sum() >= target_length:
            return p
    raise RuntimeError("target length not reached")


def save_latlon(path: Path, p: np.ndarray) -> None:
    """Map planar km to (lat, lon) degrees around (LAT0, LON0) and write them."""
    lat0 = np.radians(LAT0_DEG)
    lat = LAT0_DEG + np.degrees(p[:, 1] / EARTH_RADIUS_KM)
    lon = LON0_DEG + np.degrees(p[:, 0] / (EARTH_RADIUS_KM * np.cos(lat0)))
    np.savetxt(path, np.column_stack([lat, lon]), fmt="%.10f",
               header=f"{path.stem}: lat lon (degrees)")
    seg = np.hypot(*np.diff(p, axis=0).T)
    print(f"wrote {path}: {len(p)} vertices, length {seg.sum():.2f} km, "
          f"median segment {1000 * np.median(seg):.1f} m")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("calibration")
    out.mkdir(parents=True, exist_ok=True)
    save_latlon(out / "koch.txt", koch(6, BASE_KM))
    save_latlon(out / "sine.txt", sine(BASE_KM, 3, TARGET_LENGTH_KM))


if __name__ == "__main__":
    main()
