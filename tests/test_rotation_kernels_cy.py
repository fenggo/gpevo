"""Parity tests for the two shared 310 numeric kernels.

The original numpy routines (``_python_coord2zmatrix`` / ``_python_rotate``)
are the oracle.  A disagreement would silently change Z-matrices (read by the
soft-mode / coordinate-mutation fallbacks) or rigid-body rotations (every 310
operator), corrupting the whole search.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpevo.FunctionFolder.USPEX.src import NEW_coord2Zmatrix as zmod
from gpevo.FunctionFolder.USPEX.src import Rotate_rigid_body as rmod

_cy = pytest.importorskip("gpevo.FunctionFolder.sys.rotation_kernels_cy")


def _fmt(n, rng):
    """Valid-ish Z-matrix reference table: 1-based, first rows are 0."""
    f = np.zeros((n, 3), dtype=np.intc)
    f[0] = [0, 0, 0]
    if n > 1:
        f[1] = [1, 0, 0]
    for i in range(2, n):
        f[i] = [rng.integers(1, i + 1), rng.integers(1, i + 1),
                rng.integers(1, i + 1)]
    return f


# ---------------------------------------------------------------------------
#  coord2zmatrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(25))
def test_zmatrix_matches_reference_on_random_molecules(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 45))
    coords = rng.uniform(-5, 5, size=(n, 3))
    fmt = _fmt(n, rng)
    py = zmod._python_coord2zmatrix(coords.copy(), np.asarray(fmt, dtype=int))
    cy = _cy.coord2zmatrix(np.ascontiguousarray(coords, dtype=float), fmt)
    pub = zmod.NEW_coord2Zmatrix(coords, fmt)
    assert np.allclose(cy, py, atol=1e-12, rtol=0), seed
    assert np.allclose(pub, py, atol=1e-12, rtol=0), seed


def test_zmatrix_single_atom():
    coords = np.array([[1.0, 2.0, 3.0]])
    out = zmod.NEW_coord2Zmatrix(coords, np.zeros((1, 3), dtype=np.intc))
    assert np.array_equal(out, coords)


def test_zmatrix_two_atoms_and_zero_components():
    # ay == 0 and az == 0 exercise the two conditional branches.
    for delta in ([1.5, 0.0, 0.0], [1.5, 0.0, 2.0], [1.5, 2.0, 0.0]):
        coords = np.array([[0.0, 0.0, 0.0], list(delta)])
        fmt = np.zeros((2, 3), dtype=np.intc)
        py = zmod._python_coord2zmatrix(coords.copy(), fmt.astype(int))
        cy = zmod.NEW_coord2Zmatrix(coords, fmt)
        assert np.allclose(cy, py, atol=1e-12, rtol=0)


def test_zmatrix_collinear_degenerate_denominator():
    # All atoms on one line -> cross products vanish -> denom2 < 1e-12 branch.
    coords = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
                       [3., 0., 0.], [4., 0., 0.]])
    fmt = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0],
                    [3, 2, 1], [4, 3, 2]], dtype=np.intc)
    py = zmod._python_coord2zmatrix(coords.copy(), fmt.astype(int))
    cy = zmod.NEW_coord2Zmatrix(coords, fmt)
    assert np.allclose(cy, py, atol=1e-12, rtol=0)


def test_zmatrix_fmt_zero_clamps_to_atom_zero():
    rng = np.random.default_rng(3)
    coords = rng.uniform(-3, 3, size=(8, 3))
    fmt = np.zeros((8, 3), dtype=np.intc)   # every reference clamps to atom 0
    fmt[2:, 0] = 1
    py = zmod._python_coord2zmatrix(coords.copy(), fmt.astype(int))
    cy = zmod.NEW_coord2Zmatrix(coords, fmt)
    assert np.allclose(cy, py, atol=1e-12, rtol=0)


def test_zmatrix_does_not_mutate_input():
    rng = np.random.default_rng(9)
    coords = rng.uniform(-3, 3, size=(10, 3))
    before = coords.copy()
    zmod.NEW_coord2Zmatrix(coords, _fmt(10, rng))
    assert np.array_equal(coords, before)


# ---------------------------------------------------------------------------
#  rotate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_rotate_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 60))
    coords = rng.uniform(-5, 5, size=(n, 3))
    point = rng.uniform(-2, 2, size=3)
    axis = rng.uniform(-1, 1, size=3)
    angle = float(rng.uniform(-np.pi, np.pi))
    py = rmod._python_rotate(point.copy(), axis.copy(), coords.copy(), angle)
    cy = rmod.Rotate_rigid_body(point, axis, coords, angle)
    assert np.allclose(cy, py, atol=1e-12, rtol=0), seed


def test_rotate_zero_length_axis():
    coords = np.array([[0., 0., 0.], [1., 2., 3.], [4., 5., 6.]])
    py = rmod._python_rotate(np.zeros(3), np.zeros(3), coords.copy(), 0.7)
    cy = rmod.Rotate_rigid_body([0, 0, 0], [0, 0, 0], coords, 0.7)
    assert np.allclose(cy, py, atol=1e-12, rtol=0)


def test_rotate_accepts_lists_and_non_contiguous_arrays():
    # GetOrientation.py passes python lists; a[:, loop] is a strided view.
    rng = np.random.default_rng(1)
    coords = rng.uniform(-2, 2, size=(36, 3))
    a = rng.uniform(-1, 1, size=(3, 3))
    axis_view = a[:, 1]
    py = rmod._python_rotate(np.asarray([0., 0., 0.]),
                             np.asarray(axis_view, dtype=float),
                             np.asarray(coords, dtype=float), 0.3)
    cy = rmod.Rotate_rigid_body([0, 0, 0], axis_view, coords, 0.3)
    assert np.allclose(cy, py, atol=1e-12, rtol=0)


def test_rotate_is_a_rigid_motion():
    rng = np.random.default_rng(4)
    coords = rng.uniform(-2, 2, size=(20, 3))
    out = rmod.Rotate_rigid_body([0, 0, 0], [0, 0, 1], coords, 1.234)
    d0 = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    d1 = np.linalg.norm(out[:, None, :] - out[None, :, :], axis=2)
    assert np.allclose(d0, d1, atol=1e-12)


# ---------------------------------------------------------------------------
#  fallback / activation
# ---------------------------------------------------------------------------

def test_kernels_are_active():
    assert zmod._cython_coord2zmatrix is not None
    assert rmod._cython_rotate is not None


def test_fallback_when_extension_missing(monkeypatch):
    rng = np.random.default_rng(2)
    coords = rng.uniform(-3, 3, size=(12, 3))
    fmt = _fmt(12, rng)
    monkeypatch.setattr(zmod, "_cython_coord2zmatrix", None)
    monkeypatch.setattr(rmod, "_cython_rotate", None)
    assert np.allclose(
        zmod.NEW_coord2Zmatrix(coords, fmt),
        zmod._python_coord2zmatrix(coords.copy(), fmt.astype(int)), atol=1e-12)
    py = rmod._python_rotate(np.zeros(3), np.array([0.3, 0.5, 0.8]),
                             coords.copy(), 0.7)
    assert np.allclose(
        rmod.Rotate_rigid_body([0, 0, 0], [0.3, 0.5, 0.8], coords, 0.7),
        py, atol=1e-12)
