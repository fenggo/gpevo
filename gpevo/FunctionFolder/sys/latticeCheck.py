"""Python port of ``FunctionFolder/sys/latticeCheck.m``."""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G
from gpevo.matrix import latConverter


def latticeCheck(Lattice):
    """Return 1 when the lattice passes USPEX's angle/length sanity checks."""
    ORG = G["ORG_STRUC"]
    lat6 = np.asarray(latConverter(Lattice), dtype=float).reshape(-1)[:6]
    lat = np.asarray(Lattice, dtype=float).reshape(3, 3)

    if np.any(np.isnan(lat)):
        return 0

    ok = True
    min_angle = float(getattr(ORG, "minAngle", 55))
    angles = np.degrees(lat6[3:6])
    if np.any(angles < min_angle) or np.any(angles > 180.0 - min_angle):
        ok = False

    min_vector = float(getattr(ORG, "minVectorLength", 0.0))
    if np.any(lat6[:3] < min_vector):
        ok = False

    if ok:
        a, b, c = lat6[:3]
        alpha, beta, gamma = lat6[3:6]
        a_bc = float(np.dot(lat[0], lat[1] + lat[2]))
        ab_c = float(np.dot(lat[2], lat[1] + lat[0]))
        b_ca = float(np.dot(lat[1], lat[0] + lat[2]))

        def angle(num, den):
            den = float(den)
            if abs(den) < 1e-12:
                return 90.0
            return float(np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0))))

        angles_diag = np.array([
            angle(a_bc, a * np.sqrt(b**2 + c**2 + 2 * b * c * np.cos(alpha))),
            angle(b_ca, b * np.sqrt(a**2 + c**2 + 2 * a * c * np.cos(beta))),
            angle(ab_c, c * np.sqrt(b**2 + a**2 + 2 * b * a * np.cos(gamma))),
        ])
        min_diag = float(getattr(ORG, "minDiagAngle", 30))
        if np.any(angles_diag < min_diag) or np.any(angles_diag > 180.0 - min_diag):
            ok = False

    if int(getattr(ORG, "dimension", 3)) == 2:
        ok = True
    if int(getattr(ORG, "constLattice", 0)):
        ok = True
    return 1 if ok else 0
