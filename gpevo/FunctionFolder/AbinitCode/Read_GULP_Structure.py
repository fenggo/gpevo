"""Python equivalent of Read_GULP_Structure.m."""
from __future__ import annotations

import numpy as np

from gpevo.matrix import latConverter


def Read_GULP_Structure():
    if not __import__("os").path.exists("optimized.structure"):
        raise FileNotFoundError("optimized.structure")
    coords = []
    dolat = 0
    docoor = 0
    lat = np.eye(3)
    with open("optimized.structure", errors="replace") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            lower = line.lower()
            if "cell" in lower:
                dolat = 1
                continue
            if "fractional" in lower or "cartesian" in lower:
                docoor = 1
                continue
            if dolat == 1:
                parts = [x for x in line.replace(",", " ").split() if _num(x)]
                if len(parts) >= 3:
                    nums = [float(x) for x in parts]
                    if len(nums) >= 6:
                        six = np.array(nums[:6], dtype=float)
                        six[3:6] = np.radians(six[3:6])
                        lat = latConverter(six)
                    dolat = 0
                continue
            if docoor == 1:
                if "core" in lower:
                    parts = [x for x in line.replace(",", " ").split() if _num(x)]
                    if len(parts) >= 3:
                        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                else:
                    if coords:
                        docoor = 0
                        break
    coords = np.asarray(coords, dtype=float)
    coords = coords - np.floor(coords)
    return coords, lat


def _num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
