# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython kernel for intramol_bonds (molecular bond-table construction).

The pure-numpy port (mol_bond_table.py) walked an O(n^2) double Python loop,
calling np.linalg.norm and _rcov() (a G/dict lookup) once per atom pair.  This
kernel walks the same pairs in plain C with typed memoryviews; the covalent
radii are precomputed once by the caller and passed in.

Semantics kept bit-for-bit with the numpy version:

* a pair (i, j), i < j, is a bond when sqrt(dx^2+dy^2+dz^2) is strictly less
  than BOND_SCALE*(r[i]+r[j]) -- same strict ``<`` as np.linalg.norm(v) < thr;
* radii are the already-resolved covalent radii for each atom (see _rcov);
* returns a Python list of (i, j) tuples, i.e. exactly the set contents.

Only the buffer protocol is used (no cimport numpy), so no numpy headers are
needed at build time.
"""

from libc.math cimport sqrt


def bonds(double[:, ::1] coords, double[::1] radii, double scale, int n):
    """Return the list of bonded (i, j) pairs (i < j) for one molecule."""
    cdef int i, j
    cdef double dx, dy, dz, d, thr
    cdef list out = []
    cdef double* c = &coords[0, 0]
    cdef double* r = &radii[0]
    if n < 2:
        return out
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = c[3*i] - c[3*j]
            dy = c[3*i+1] - c[3*j+1]
            dz = c[3*i+2] - c[3*j+2]
            d = sqrt(dx*dx + dy*dy + dz*dz)
            thr = scale * (r[i] + r[j])
            if d < thr:
                out.append((i, j))
    return out
