"""Python equivalent of Rotate_rigid_body.m (rigid body rotation)."""
from __future__ import annotations

import numpy as np
import math


def Rotate_rigid_body(point, R_axis, inputs, angle):
    point = np.asarray(point, dtype=float)
    axis = np.asarray(R_axis, dtype=float)
    coords = np.asarray(inputs, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0:
        n = 1.0
    mu = axis / n
    x, y, z = mu
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c
    M = np.array([
        [c + x*x*t, x*y*t - z*s, x*z*t + y*s],
        [x*y*t + z*s, c + y*y*t, y*z*t - x*s],
        [x*z*t - y*s, y*z*t + x*s, c + z*z*t],
    ])
    r = coords - point
    return r @ M + point
