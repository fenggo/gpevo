"""Python equivalent of NEW_coord2Zmatrix.m.

The per-atom scalar loop is compiled in ``sys/rotation_kernels_cy.pyx``.  The
original numpy routine is kept below as ``_python_coord2zmatrix`` (reference
oracle + fallback when the extension is not built); the parity tests in
``tests/test_rotation_kernels_cy.py`` assert the two agree bit-for-bit.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from gpevo.FunctionFolder.sys.rotation_kernels_cy import (
        coord2zmatrix as _cython_coord2zmatrix,
    )
except Exception:                       # extension not built / incompatible
    _cython_coord2zmatrix = None


def _python_coord2zmatrix(coords, fmt):
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


def NEW_coord2Zmatrix(coords, fmt):
    """Cartesian coordinates -> Z-matrix (bonds / angles / torsions).

    ``fmt`` holds 1-based reference-atom indices; it is cast to int32 for the
    Cython memoryview.  Falls back to the pure-numpy oracle when the compiled
    kernel is unavailable.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] == 1:
        return coords.copy()
    fmt_i = np.ascontiguousarray(fmt, dtype=np.intc)
    if _cython_coord2zmatrix is not None:
        return _cython_coord2zmatrix(
            np.ascontiguousarray(coords, dtype=float), fmt_i)
    return _python_coord2zmatrix(coords.copy(), np.asarray(fmt_i, dtype=int))


def _rot_mat(angle, axis):
    """Rodrigues rotation matrix (matches NEW_ZMATRIXCOORD.m)."""
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return np.array([
        [c + x * x * t,     x * y * t - z * s, x * z * t + y * s],
        [y * x * t + z * s, c + y * y * t,     y * z * t - x * s],
        [z * x * t - y * s, z * y * t + x * s, c + z * z * t],
    ])


def _con_struct_coo(related, r, theta, phi):
    """Build one atom from its 3 reference atoms (related: 3x3, rows = i, j, k)."""
    xi, yi, zi = related[0]
    xj, yj, zj = related[1]
    xk, yk, zk = related[2]
    xji = xi - xj; yji = yi - yj; zji = zi - zj
    rji = math.sqrt(xji * xji + yji * yji + zji * zji)
    xji = xji / rji; yji = yji / rji; zji = zji / rji
    xki = xk - xj; yki = yk - yj; zki = zk - zj
    xn = yji * zki - yki * zji
    yn = zji * xki - zki * xji
    zn = xji * yki - xki * yji
    rn = math.sqrt(xn * xn + yn * yn + zn * zn)
    xn = xn / rn; yn = yn / rn; zn = zn / rn
    xp = yn * zji - yji * zn
    yp = zn * xji - zji * xn
    zp = xn * yji - xji * yn
    rji = r * math.cos(theta)
    rn = r * math.sin(theta) * math.sin(phi)
    rp = r * math.sin(theta) * math.cos(phi)
    x = xi - rji * xji + rn * xn + rp * xp
    y = yi - rji * yji + rn * yn + rp * yp
    z = zi - rji * zji + rn * zn + rp * zp
    return np.array([x, y, z])


def NEW_ZMATRIXCOORD(zmatrix, fmt):
    """Python port of NEW_ZMATRIXCOORD.m (Z-matrix -> Cartesian coordinates)."""
    z = np.asarray(zmatrix, dtype=float)
    fmt = np.asarray(fmt, dtype=int)
    n = z.shape[0]
    if n == 1:
        return z.copy()
    teempi = z[0].copy()
    z = z.copy()
    z[0] = 0.0
    coords = np.zeros((n, 3))
    coords[0] = [0.0, 0.0, 0.0]
    coords[1] = [z[1, 0], 0.0, 0.0]
    if n > 2.5:
        for ind in range(2, n):
            if ind == 2:
                related = np.array([
                    [z[max(0, int(fmt[ind, 0]) - 1), 0], 0.0, 0.0],
                    [z[max(0, int(fmt[ind, 1]) - 1), 0], 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ])
            else:
                form = [max(0, int(fmt[ind, 0]) - 1),
                        max(0, int(fmt[ind, 1]) - 1),
                        max(0, int(fmt[ind, 2]) - 1)]
                related = coords[form]
            r = z[ind, 0]
            theta = z[ind, 1]
            phi = z[ind, 2]
            coords[ind] = _con_struct_coo(related, r, theta, phi)
            if ind == 2:
                for outer in range(2):
                    if outer == 0:
                        rot_vec = np.array([0.0, 1.0, 0.0])
                        torsang = z[1, 1] - math.pi / 2.0
                    else:
                        rot_vec = np.array([0.0, 0.0, 1.0])
                        torsang = z[1, 2]
                    for inder in range(3):
                        coords[inder] = _rot_mat(torsang, rot_vec) @ coords[inder]
    elif n == 2:
        for outer in range(2):
            if outer == 0:
                rot_vec = np.array([0.0, 1.0, 0.0])
                torsang = z[1, 1] - math.pi / 2.0
            else:
                rot_vec = np.array([0.0, 0.0, 1.0])
                torsang = z[1, 2]
            for inder in range(2):
                coords[inder] = _rot_mat(torsang, rot_vec) @ coords[inder]
    coords[:, 0] += teempi[0]
    coords[:, 1] += teempi[1]
    coords[:, 2] += teempi[2]
    return np.real(coords)
