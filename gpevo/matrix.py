"""Small NumPy-backed matrix helpers mirroring USPEX sys routines."""
from __future__ import annotations

import math
import numpy as np


def ensure_np(value, dtype=float):
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False) if value.dtype != dtype else value
    return np.asarray(value, dtype=dtype)


def det(value) -> float:
    return float(np.linalg.det(ensure_np(value)))


def inv(matrix):
    return np.linalg.inv(ensure_np(matrix))


def eig(matrix):
    w, v = np.linalg.eig(ensure_np(matrix))
    return v, w


def zeros(*shape):
    if len(shape) == 1:
        return np.zeros(shape[0], dtype=float)
    return np.zeros(tuple(shape), dtype=float)


def ones(*shape):
    if len(shape) == 1:
        return np.ones(shape[0], dtype=float)
    return np.ones(tuple(shape), dtype=float)


def rand(*shape):
    if not shape:
        return np.random.random()
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return np.random.random(tuple(shape))


def randperm(n):
    return np.random.permutation(int(n)) + 1


def sum_all(a):
    return np.sum(a)


def mean(a, axis=None):
    return np.mean(a, axis=axis)


def norm(a):
    return float(np.linalg.norm(a))


def latConverter(lat):
    """Mirror FunctionFolder/sys/latConverter.m.

    A 3x3 lattice is converted to [a b c alpha beta gamma] (angles in
    radians). A 6-parameter lattice is converted to a 3x3 Cartesian matrix;
    angles are assumed to be in radians, matching the MATLAB implementation.

    The convention throughout gpevo is that the rows of the 3x3 matrix are
    the lattice vectors: ``latConverter`` reads them off the rows, and
    ``Frac2Cart``/``Cart2Frac`` multiply on the right (``frac @ M``,
    ``cart @ inv(M)``).  The 6-parameter branch must therefore build a
    matrix whose ROWS are a, b, c -- the lower-triangular form of
    latConverter.m.  Building the transpose instead (the previous
    behaviour) leaves the vectors in the columns, so every non-orthogonal
    cell came out as a different lattice: it does not even round-trip
    through ``latConverter``, and the symmetry operations of the generated
    space group are then not isometries of it, which shears rigid
    molecules into distorted copies.
    """
    arr = np.asarray(lat, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] == 3:
        output = np.zeros((6, 1), dtype=float)
        va = arr[0, :]
        vb = arr[1, :]
        vc = arr[2, :]
        output[0, 0] = np.linalg.norm(va)
        output[1, 0] = np.linalg.norm(vb)
        output[2, 0] = np.linalg.norm(vc)
        output[3, 0] = math.acos(np.clip(np.dot(vb, vc) / (output[1, 0] * output[2, 0]), -1.0, 1.0))
        output[4, 0] = math.acos(np.clip(np.dot(va, vc) / (output[0, 0] * output[2, 0]), -1.0, 1.0))
        output[5, 0] = math.acos(np.clip(np.dot(va, vb) / (output[0, 0] * output[1, 0]), -1.0, 1.0))
        return output

    a, b, c, alpha, beta, gamma = arr.reshape(-1)[:6]
    cos_a, cos_b, cos_g = math.cos(alpha), math.cos(beta), math.cos(gamma)
    M = np.zeros((3, 3), dtype=float)
    M[0, 0] = a
    M[1, 0] = b * cos_g
    M[1, 1] = b * math.sin(gamma)
    M[2, 0] = c * cos_b
    M[2, 1] = (c * cos_a * math.sin(gamma)
               - (c * cos_b - c * cos_a * cos_g) / math.tan(gamma))
    M[2, 2] = math.sqrt(max(0.0, c * c - M[2, 0] ** 2 - M[2, 1] ** 2))
    return M


def Frac2Cart(frac, lat):
    """Fractional coordinates -> Cartesian coordinates.

    Accepts either 6 lattice parameters or an actual 3x3 lattice.
    """
    fr = np.asarray(frac, dtype=float)
    l = np.asarray(lat, dtype=float).reshape(-1)
    if l.size == 6:
        M = latConverter(l)
    elif l.size == 9:
        M = l.reshape(3, 3)
    else:
        raise ValueError("lattice must be 6 parameters or 3x3")
    return fr @ M


def Cart2Frac(cart, lat):
    c = np.asarray(cart, dtype=float)
    l = np.asarray(lat, dtype=float).reshape(-1)
    if l.size == 6:
        M = latConverter(l)
    elif l.size == 9:
        M = l.reshape(3, 3)
    else:
        raise ValueError("lattice must be 6 parameters or 3x3")
    return c @ np.linalg.inv(M)


def matmul(a, b):
    return np.asarray(a) @ np.asarray(b)


def pdist2(a, b):
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    return np.sqrt(np.sum((a[:, None, :] - b[None, :, :])**2, axis=-1))
