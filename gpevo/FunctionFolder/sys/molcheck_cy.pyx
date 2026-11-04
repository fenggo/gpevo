# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython kernel for newMolCheck (molecular distance validity checks).

The Python port (newMolCheck.py) used to build one numpy array per atom
pair: ~25k Python-level numpy calls for an 8-molecule TNT/CL20 cell.  This
kernel walks the same loops in plain C with typed memoryviews and no GIL.

Semantics kept bit-for-bit with the numpy version:

* minimum-image convention uses np.round (round half to EVEN), not C
  round (half away from zero) -- _rint below reproduces np.rint;
* a cartesian displacement is dfrac @ lat, i.e. columns of lat;
* distances are sqrt(sum of squares) compared against the threshold,
  exactly like np.linalg.norm(vectors, axis=1) <= flatthr;
* the first violation returns 0 immediately (early rejection is what makes
  bad offspring cheap).

Only the buffer protocol is used (no cimport numpy), so no numpy headers
are needed at build time.
"""

from libc.math cimport floor, fmod, sqrt


cdef inline double _rint(double x) noexcept nogil:
    """numpy np.round / rint: round half to the nearest EVEN integer."""
    cdef double f = floor(x)
    cdef double r = x - f
    if r > 0.5:
        return f + 1.0
    if r < 0.5:
        return f
    if fmod(f, 2.0) == 0.0:
        return f
    return f + 1.0


cdef int _molcheck(
        double[:, ::1] frac,        # (N, 3) fractional coords, concatenated
        double[:, ::1] coords,      # (N, 3) cartesian coords, concatenated
        int[::1] starts,            # (n_mols + 1,) first atom of each molecule
        int[::1] types,             # (N,) 0-based atom type
        double[:, ::1] lat,         # (3, 3) lattice
        double[:, ::1] min_matrix,  # (n_types, n_types) distance thresholds
        double[:, ::1] offsets,     # (124, 3) periodic image shifts
) noexcept nogil:
    cdef Py_ssize_t n_mols = starts.shape[0] - 1
    cdef Py_ssize_t n_off = offsets.shape[0]
    cdef Py_ssize_t i, k, a, b, p, m, n, ind
    cdef Py_ssize_t si, ei, sk, ek, s, na
    cdef double d0, d1, d2, x, y, z, dist2, thr, best
    cdef int ta, tb

    # ---- intermolecular: every atom pair of every molecule pair ----
    for i in range(n_mols - 1):
        si = starts[i]
        ei = starts[i + 1]
        if ei <= si:
            continue
        for k in range(i + 1, n_mols):
            sk = starts[k]
            ek = starts[k + 1]
            if ek <= sk:
                continue
            for a in range(si, ei):
                ta = types[a]
                for b in range(sk, ek):
                    tb = types[b]
                    d0 = frac[a, 0] - frac[b, 0]
                    d0 -= _rint(d0)
                    d1 = frac[a, 1] - frac[b, 1]
                    d1 -= _rint(d1)
                    d2 = frac[a, 2] - frac[b, 2]
                    d2 -= _rint(d2)
                    x = d0 * lat[0, 0] + d1 * lat[1, 0] + d2 * lat[2, 0]
                    y = d0 * lat[0, 1] + d1 * lat[1, 1] + d2 * lat[2, 1]
                    z = d0 * lat[0, 2] + d1 * lat[1, 2] + d2 * lat[2, 2]
                    dist2 = x * x + y * y + z * z
                    thr = min_matrix[ta, tb]
                    if thr >= 0.0 and sqrt(dist2) <= thr:
                        return 0

    # ---- intramolecular: each atom against all 124 periodic images ----
    for ind in range(n_mols):
        s = starts[ind]
        na = starts[ind + 1] - s
        if na < 2:
            continue
        for n in range(1, na):
            for m in range(n):
                thr = min_matrix[types[s + m], types[s + n]]
                if thr < 0.0:
                    continue
                best = 1.0e300
                for p in range(n_off):
                    x = coords[s + n, 0] + offsets[p, 0] - coords[s + m, 0]
                    y = coords[s + n, 1] + offsets[p, 1] - coords[s + m, 1]
                    z = coords[s + n, 2] + offsets[p, 2] - coords[s + m, 2]
                    dist2 = x * x + y * y + z * z
                    if dist2 < best:
                        best = dist2
                if sqrt(best) <= thr:
                    return 0
    return 1


def molcheck(frac, coords, starts, types, lat, min_matrix, offsets):
    """Return 1 when the structure passes, 0 as soon as a contact is too close.

    All arguments are numpy arrays; the buffers are acquired while the GIL is
    held, then the whole distance scan runs with the GIL released.
    """
    cdef double[:, ::1] f = frac
    cdef double[:, ::1] c = coords
    cdef int[::1] s = starts
    cdef int[::1] t = types
    cdef double[:, ::1] l = lat
    cdef double[:, ::1] mm = min_matrix
    cdef double[:, ::1] o = offsets
    cdef int result
    with nogil:
        result = _molcheck(f, c, s, t, l, mm, o)
    return result
