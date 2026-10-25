"""Python port of ``FunctionFolder/sys/Transform_Convex_Hull.m``.

Converts a convex-hull table between "block" space (multiples of the
composition blocks given by ``numIons``) and "atom" space (absolute atom
counts), rescaling the enthalpy column accordingly.

``numIons`` is the block matrix: one row per block, one column per atom type.
``flag == 0`` converts block -> atom, otherwise atom -> block.
"""
from __future__ import annotations

import numpy as np


def Transform_Convex_Hull(convex_hull, numIons, flag=0):
    if convex_hull is None or np.size(convex_hull) == 0:
        return np.zeros((0, 0))

    hull = np.asarray(convex_hull, dtype=float)
    if hull.ndim == 1:
        hull = hull.reshape(1, -1)
    n_point = hull.shape[0]

    matrix = np.asarray(numIons, dtype=float)
    if matrix.ndim == 1:
        # a single block (or a single atom type) - treat as one row
        matrix = matrix.reshape(1, -1)

    n_block, n_atom = matrix.shape

    if int(flag) == 0:
        out = np.zeros((n_point, n_atom + 2), dtype=float)
        for i in range(n_point):
            block = hull[i, :n_block]
            atom = block @ matrix
            out[i, :n_atom] = atom
            sum_block = float(np.sum(block))
            sum_atom = float(np.sum(atom))
            out[i, n_atom] = hull[i, n_block] * (sum_block / sum_atom) if sum_atom else hull[i, n_block]
            out[i, n_atom + 1] = hull[i, n_block + 1]
        return out

    out = np.zeros((n_point, n_block + 2), dtype=float)
    for i in range(n_point):
        atom = hull[i, :n_atom]
        # mrdivide: block @ matrix == atom
        block = np.linalg.lstsq(matrix.T, atom, rcond=None)[0]
        out[i, :n_block] = block
        sum_atom = float(np.sum(atom))
        sum_block = float(np.sum(block))
        out[i, n_block] = hull[i, n_atom] * (sum_atom / sum_block) if sum_block else hull[i, n_atom]
        out[i, n_block + 1] = hull[i, n_atom + 1]
    return out
