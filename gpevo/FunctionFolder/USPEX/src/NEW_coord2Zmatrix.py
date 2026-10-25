"""Python equivalent of NEW_coord2Zmatrix.m."""
from __future__ import annotations

import math
import numpy as np


def NEW_coord2Zmatrix(coords, fmt):
    coords = np.asarray(coords, dtype=float).copy()
    n = coords.shape[0]
    if n == 1:
        return coords.copy()
    z = np.zeros((n, 3))
    z[0] = coords[0]
    coords = coords - coords[0]
    d = float(np.linalg.norm(coords[1]))
    z[1, 0] = d
    z[1, 1] = math.pi/2 if coords[1, 2] == 0 else math.acos(np.clip(coords[1, 2] / d, -1, 1))
    z[1, 2] = 0 if coords[1, 1] == 0 else math.atan2(coords[1, 1], coords[1, 0])

    fmt = np.asarray(fmt, dtype=int)
    for ind in range(2, n):
        i1 = max(0, int(fmt[ind, 0]) - 1)
        i2 = max(0, int(fmt[ind, 1]) - 1)
        d = float(np.linalg.norm(coords[ind] - coords[i1]))
        z[ind, 0] = d
        vec1 = coords[ind] - coords[i1]
        vec2 = coords[i1] - coords[i2]
        denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denom < 1e-12:
            z[ind, 1] = 0.0
        else:
            z[ind, 1] = math.acos(np.clip(np.dot(vec1, -vec2) / denom, -1, 1))
        if ind == 2:
            v3 = np.array([0, 0, -1.0])
            i3 = -1
        else:
            i3 = max(0, int(fmt[ind, 2]) - 1)
            v3 = coords[i2] - coords[i3]
        normal1 = np.cross(vec2, v3)
        normal2 = np.cross(vec1, vec2)
        denom2 = np.linalg.norm(normal1) * np.linalg.norm(normal2)
        numer = np.linalg.norm(vec2) * np.dot(vec1, normal1)
        if denom2 < 1e-12:
            tors = 0.0
        else:
            tors = math.atan2(numer, np.dot(normal2, normal1))
        z[ind, 2] = -tors
        if abs(tors + math.pi) < 0.0001:
            z[ind, 2] = math.pi
    return z
