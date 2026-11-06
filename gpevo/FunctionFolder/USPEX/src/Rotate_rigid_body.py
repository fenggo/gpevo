"""Python equivalent of Rotate_rigid_body.m (rigid body rotation).

The scalar Rodrigues loop is compiled in ``sys/rotation_kernels_cy.pyx``.  The
original numpy routine is kept below as a reference oracle (and a fallback for
source checkouts where the extension has not been built); the parity tests in
``tests/test_rotation_kernels_cy.py`` assert the two agree bit-for-bit.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from gpevo.FunctionFolder.sys.rotation_kernels_cy import rotate as _cython_rotate
except Exception:                       # extension not built / incompatible
    _cython_rotate = None


def _python_rotate(point, axis, coords, angle):
    """Original numpy implementation, kept as the reference oracle."""
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


def Rotate_rigid_body(point, R_axis, inputs, angle):
    point = np.ascontiguousarray(point, dtype=float)
    axis = np.ascontiguousarray(R_axis, dtype=float)
    coords = np.ascontiguousarray(inputs, dtype=float)
    if _cython_rotate is not None:
        return _cython_rotate(point, axis, coords, float(angle))
    return _python_rotate(point, axis, coords, float(angle))
