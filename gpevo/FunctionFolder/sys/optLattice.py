"""Python port of ``FunctionFolder/sys/optLattice.m``."""
from __future__ import annotations

import numpy as np


def _reoptimize_vector(v1, v2):
    """Return ``(new_v1, changed)`` after a Delaunay-style reduction step."""
    if abs(float(np.dot(v1, v2))) > np.linalg.norm(v2) / 2:
        try:
            factor = np.ceil(abs(np.dot(v1, v2)) / (np.linalg.norm(v2) ** 2))
            trial = v1 - factor * np.sign(np.dot(v1, v2)) * v2
        except Exception:
            trial = v1
        if np.linalg.norm(trial) < np.linalg.norm(v1):
            return trial, True
    return v1, False


def optLattice(coor, lattice):
    """Reduce the lattice vectors and re-wrap ``coor`` into the reduced cell."""
    lat = np.asarray(lattice, dtype=float).reshape(3, 3).copy()
    v = [lat[0].copy(), lat[1].copy(), lat[2].copy()]
    flag = True
    step = 0
    while flag:
        flag = False
        for i, j in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
            v[i], changed = _reoptimize_vector(v[i], v[j])
            flag = flag or changed
        if step > 100:
            break
        for i, (j, k) in enumerate(((1, 2), (0, 2), (0, 1))):
            v[i], changed = _reoptimize_vector(v[i], v[j] + v[k])
            flag = flag or changed
        if flag:
            step += 1

    opt_lat = np.array(v, dtype=float)
    coords = np.asarray(coor, dtype=float)

    if np.linalg.det(opt_lat) < 1e-7 and np.sum(coords) == 0:
        opt_lat = np.eye(3)
        temp = coords
    else:
        try:
            temp = coords @ np.linalg.inv(opt_lat)
        except np.linalg.LinAlgError:
            temp = coords
            opt_lat = np.asarray(lattice, dtype=float).reshape(3, 3)
    opt_coor = temp - np.floor(temp)
    opt_coor = opt_coor @ opt_lat
    return opt_coor, opt_lat
