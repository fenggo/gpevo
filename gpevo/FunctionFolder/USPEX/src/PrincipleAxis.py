"""Python equivalent of PrincipleAxis.m."""
from __future__ import annotations

import numpy as np


def PrincipleAxis(absolute_coord):
    coords = np.asarray(absolute_coord, dtype=float)
    coords = coords - np.mean(coords, axis=0)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    I = np.zeros((3, 3))
    I[0, 0] = np.sum(y*y + z*z)
    I[1, 1] = np.sum(x*x + z*z)
    I[2, 2] = np.sum(x*x + y*y)
    I[0, 1] = I[1, 0] = -np.sum(x*y)
    I[1, 2] = I[2, 1] = -np.sum(y*z)
    I[0, 2] = I[2, 0] = -np.sum(z*x)
    w, v = np.linalg.eig(I)
    return v, w
