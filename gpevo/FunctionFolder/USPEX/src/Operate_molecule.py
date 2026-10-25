"""Python equivalent of Operate_molecule.m."""
from __future__ import annotations

import numpy as np

from gpevo.matrix import Cart2Frac, Frac2Cart


def Operate_molecule(R0, Opt, lat, P, PB):
    R0 = np.asarray(R0, dtype=float)
    Opt = np.asarray(Opt, dtype=float)
    P = [int(p) - 1 for p in P]
    PB = [int(pb) - 1 for pb in PB]
    R0_frac = Cart2Frac(R0, lat)
    R_frac_p = R0_frac[:, P] @ Opt[0:3, 0:3] + Opt[3, :]
    # Undo lattice axis permutation
    R_frac = np.zeros_like(R_frac_p)
    for axis, src in enumerate(PB):
        R_frac[:, axis] = R_frac_p[:, src]
    return Frac2Cart(R_frac, lat)
