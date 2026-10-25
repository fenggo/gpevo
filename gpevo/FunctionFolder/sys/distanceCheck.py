"""Python equivalent of distanceCheck.m."""
from __future__ import annotations

import itertools
import numpy as np

from gpevo.matrix import latConverter


def distanceCheck(coordinates, lattice, composition, min_dist_matrix):
    coordinates = np.asarray(coordinates, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    composition = list(map(int, composition))
    min_dist_matrix = np.asarray(min_dist_matrix, dtype=float)

    if not np.any(min_dist_matrix > 0.0001):
        return 1
    if coordinates.shape[0] != int(sum(composition)):
        return 0
    if np.iscomplexobj(coordinates) or np.iscomplexobj(lattice):
        return 0
    if np.any(np.isnan(coordinates)) or np.any(np.isnan(lattice)):
        return 0

    if lattice.size == 6:
        lattice = latConverter(lattice)
    elif lattice.size == 9:
        lattice = lattice.reshape(3, 3)

    if len(coordinates.shape) == 1:
        coordinates = coordinates.reshape(-1, 3)

    offsets = np.array([p for p in itertools.product(range(-1, 2), repeat=3)]) @ lattice
    ion_change = np.cumsum(composition)
    for loop in range(coordinates.shape[0] - 1):
        row = int(np.searchsorted(ion_change, loop + 0.5, side="right"))
        c1 = coordinates[loop] @ lattice
        for who in range(loop + 1, coordinates.shape[0]):
            col = int(np.searchsorted(ion_change, who + 0.5, side="right"))
            images = offsets + coordinates[who] @ lattice
            dmin = float(np.min(np.linalg.norm(images - c1, axis=1)))
            if dmin < min_dist_matrix[row, col]:
                return 0
    return 1
