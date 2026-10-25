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
    v = math.sqrt(1 - math.cos(alpha)**2 - math.cos(beta)**2 - math.cos(gamma)**2
                  + 2*math.cos(alpha)*math.cos(beta)*math.cos(gamma))
    M = np.array([
        [a, b*math.cos(gamma), c*math.cos(beta)],
        [0, b*math.sin(gamma), c*(math.cos(alpha)-math.cos(beta)*math.cos(gamma))/math.sin(gamma)],
        [0, 0, c*v/math.sin(gamma)]
    ], dtype=float)
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
