# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython kernels for the two shared numeric helpers of the 310 operators.

``NEW_coord2Zmatrix`` (src/NEW_coord2Zmatrix.py) built a small numpy array and
called np.cross / np.linalg.norm / np.dot once per atom, inside a Python loop.
``Rotate_rigid_body`` (src/Rotate_rigid_body.py) built a 3x3 Rodrigues matrix
and did one Python-level ``r @ M`` per call -- it is invoked thousands of times
per rotation offspring.  Both are pure scalar arithmetic, so they belong in C.

Semantics kept bit-for-bit with the numpy versions (see the parity tests):

* zmatrix: the reference-atom indices in ``fmt`` are 1-based; 0 (or negative)
  clamps to atom 0, exactly like ``max(0, int(fmt[ind, k]) - 1)``.
  ``ind == 2`` uses the fixed ``v3 = (0, 0, -1)`` normal.  The clip to [-1, 1]
  before acos, the 1e-12 degenerate-denominator guards, and the
  ``abs(tors + pi) < 1e-4 -> pi`` special case are all reproduced.
* rotate: ``r @ M + point`` with ROW vectors (r is (n, 3)); the Rodrigues
  matrix is the transpose of the column-vector form.  A zero-length axis is
  replaced by 1.0 before normalising, matching ``n = 1.0 if n == 0``.

Only the buffer protocol is used (no cimport numpy, no numpy headers at build
time).  The output arrays are allocated by the Python caller (GIL held), then
filled inside ``nogil`` blocks.
"""

import numpy as np

from libc.math cimport sqrt, acos, atan2, cos, sin, fabs, M_PI


cdef inline double _clip(double x) noexcept nogil:
    if x < -1.0:
        return -1.0
    if x > 1.0:
        return 1.0
    return x


def coord2zmatrix(double[:, ::1] coords, int[:, ::1] fmt):
    """Return the (n, 3) Z-matrix for one molecule."""
    cdef int n = coords.shape[0]
    cdef double[:, ::1] z = np.zeros((n, 3))
    if n == 1:
        z[0, 0] = coords[0, 0]
        z[0, 1] = coords[0, 1]
        z[0, 2] = coords[0, 2]
        return np.asarray(z)

    cdef int ind, i1, i2, i3
    cdef double d, denom, denom2, numer, tors, dot
    cdef double v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z
    cdef double n1x, n1y, n1z, n2x, n2y, n2z
    cdef double cx = coords[0, 0], cy = coords[0, 1], cz = coords[0, 2]
    cdef double ax = coords[1, 0] - cx, ay = coords[1, 1] - cy
    cdef double az = coords[1, 2] - cz

    with nogil:
        z[0, 0] = cx; z[0, 1] = cy; z[0, 2] = cz
        d = sqrt(ax*ax + ay*ay + az*az)
        z[1, 0] = d
        z[1, 1] = M_PI / 2 if az == 0 else acos(_clip(az / d))
        z[1, 2] = 0.0 if ay == 0 else atan2(ay, ax)

        for ind in range(2, n):
            i1 = fmt[ind, 0] - 1 if fmt[ind, 0] > 0 else 0
            i2 = fmt[ind, 1] - 1 if fmt[ind, 1] > 0 else 0
            v1x = coords[ind, 0] - coords[i1, 0]
            v1y = coords[ind, 1] - coords[i1, 1]
            v1z = coords[ind, 2] - coords[i1, 2]
            v2x = coords[i1, 0] - coords[i2, 0]
            v2y = coords[i1, 1] - coords[i2, 1]
            v2z = coords[i1, 2] - coords[i2, 2]
            d = sqrt(v1x*v1x + v1y*v1y + v1z*v1z)
            z[ind, 0] = d
            denom = d * sqrt(v2x*v2x + v2y*v2y + v2z*v2z)
            if denom < 1e-12:
                z[ind, 1] = 0.0
            else:
                dot = -(v1x*v2x + v1y*v2y + v1z*v2z) / denom
                z[ind, 1] = acos(_clip(dot))
            if ind == 2:
                v3x = 0.0; v3y = 0.0; v3z = -1.0
            else:
                i3 = fmt[ind, 2] - 1 if fmt[ind, 2] > 0 else 0
                v3x = coords[i2, 0] - coords[i3, 0]
                v3y = coords[i2, 1] - coords[i3, 1]
                v3z = coords[i2, 2] - coords[i3, 2]
            n1x = v2y*v3z - v2z*v3y
            n1y = v2z*v3x - v2x*v3z
            n1z = v2x*v3y - v2y*v3x
            n2x = v1y*v2z - v1z*v2y
            n2y = v1z*v2x - v1x*v2z
            n2z = v1x*v2y - v1y*v2x
            denom2 = (sqrt(n1x*n1x + n1y*n1y + n1z*n1z)
                      * sqrt(n2x*n2x + n2y*n2y + n2z*n2z))
            numer = (sqrt(v2x*v2x + v2y*v2y + v2z*v2z)
                     * (v1x*n1x + v1y*n1y + v1z*n1z))
            if denom2 < 1e-12:
                tors = 0.0
            else:
                tors = atan2(numer, n2x*n1x + n2y*n1y + n2z*n1z)
            z[ind, 2] = -tors
            if fabs(tors + M_PI) < 0.0001:
                z[ind, 2] = M_PI
    return np.asarray(z)


def rotate(double[::1] point, double[::1] axis, double[:, ::1] inputs,
           double angle):
    """Rigid-body rotation of (n, 3) coordinates about ``axis`` through ``point``."""
    cdef int n = inputs.shape[0], i
    cdef double nx = axis[0], ny = axis[1], nz = axis[2]
    cdef double ln = sqrt(nx*nx + ny*ny + nz*nz)
    if ln == 0:
        ln = 1.0
    nx /= ln; ny /= ln; nz /= ln
    cdef double c = cos(angle), s = sin(angle), t = 1.0 - c
    cdef double m00 = c + nx*nx*t, m01 = nx*ny*t - nz*s, m02 = nx*nz*t + ny*s
    cdef double m10 = nx*ny*t + nz*s, m11 = c + ny*ny*t, m12 = ny*nz*t - nx*s
    cdef double m20 = nx*nz*t - ny*s, m21 = ny*nz*t + nx*s, m22 = c + nz*nz*t
    cdef double[:, ::1] out = np.zeros((n, 3))
    cdef double dx, dy, dz
    with nogil:
        for i in range(n):
            dx = inputs[i, 0] - point[0]
            dy = inputs[i, 1] - point[1]
            dz = inputs[i, 2] - point[2]
            # row-vector r @ M (matches r @ M in Rotate_rigid_body.py)
            out[i, 0] = dx*m00 + dy*m10 + dz*m20 + point[0]
            out[i, 1] = dx*m01 + dy*m11 + dz*m21 + point[1]
            out[i, 2] = dx*m02 + dy*m12 + dz*m22 + point[2]
    return np.asarray(out)
