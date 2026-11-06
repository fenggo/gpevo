"""Parity tests for the intramol_bonds Cython kernel / vectorised fallback.

The original double-loop (kept as ``_reference_bonds``) is the oracle.  A
disagreement would silently change which molecules are flagged as damaged, or
which dihedral twists are accepted, corrupting the whole search.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.sys import mol_bond_table as bt

_cy = pytest.importorskip("gpevo.FunctionFolder.sys.bondtable_cy")


def _setup(atomType=(6, 7, 8, 1)):
    G["ORG_STRUC"] = AttrDict(atomType=list(atomType))


def _case(seed, n=None):
    rng = np.random.default_rng(seed)
    n = n or int(rng.integers(2, 40))
    coords = rng.uniform(0, 6, size=(n, 3))
    types = [int(x) for x in rng.integers(1, 5, size=n)]
    return coords, types


def _all(coords, types):
    r = bt.radii_for(types)
    ref = bt._reference_bonds(coords, types)
    vec = bt._vectorized_bonds(coords, r)
    cy = set(_cy.bonds(np.ascontiguousarray(coords, dtype=float),
                       np.ascontiguousarray(r, dtype=float),
                       bt.BOND_SCALE, coords.shape[0]))
    pub = bt.bonds_from_radii(coords, r)
    return ref, vec, cy, pub


@pytest.mark.parametrize("seed", range(30))
def test_all_paths_agree_on_random_molecules(seed):
    _setup()
    coords, types = _case(seed)
    ref, vec, cy, pub = _all(coords, types)
    assert vec == ref, ("vectorized", seed)
    assert cy == ref, ("cython", seed)
    assert pub == ref, ("public", seed)


@pytest.mark.parametrize("seed", range(20))
def test_all_paths_agree_near_the_bond_threshold(seed):
    """Strict `<` at exactly BOND_SCALE*(r_i+r_j) exercises the boundary."""
    _setup()
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(3, 20))
    types = [int(x) for x in rng.integers(1, 5, size=n)]
    coords = rng.uniform(0, 6, size=(n, 3))
    r = bt.radii_for(types)
    thr = bt.BOND_SCALE * (r[0] + r[1])
    # place atom 1 exactly at the cutoff distance from atom 0 (plus tiny jitter)
    coords[1] = coords[0] + np.array([thr + rng.uniform(-1e-9, 1e-9), 0, 0])
    ref, vec, cy, pub = _all(coords, types)
    assert vec == ref == cy == pub


def test_intramol_bonds_public_api_matches_reference():
    _setup()
    for seed in range(10):
        coords, types = _case(seed)
        assert bt.intramol_bonds(coords, types) == bt._reference_bonds(coords, types)


def test_small_and_empty_molecules():
    _setup()
    r = bt.radii_for([1, 1])
    assert bt.bonds_from_radii(np.zeros((0, 3)), np.zeros(0)) == set()
    assert bt.bonds_from_radii(np.array([[0.0, 0.0, 0.0]]), r[:1]) == set()
    assert bt.intramol_bonds(np.zeros((0, 3)), []) == set()


def test_invalid_type_idx_matches_on_every_path():
    """Unusual type indices keep the ORIGINAL _rcov semantics on every path.

    ``99`` is out of range -> 1.20.  ``0`` hits ``atomType[-1]`` (Python's
    negative-index wraparound) -> 0.76; that is pre-existing behaviour, not
    something this optimisation changes.  What matters is that Cython,
    vectorised and reference all agree.
    """
    _setup(atomType=(6,))
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    types = [1, 99, 0]
    r = bt.radii_for(types)
    assert r[1] == pytest.approx(1.20)
    assert r[2] == pytest.approx(0.76)
    ref = bt._reference_bonds(coords, types)
    assert ref == bt._vectorized_bonds(coords, r)
    assert ref == set(_cy.bonds(np.ascontiguousarray(coords, dtype=float),
                                np.ascontiguousarray(r, dtype=float),
                                bt.BOND_SCALE, coords.shape[0]))


def test_cython_kernel_is_active():
    assert bt._cython_bonds is not None


def test_vectorized_fallback_matches_when_extension_is_missing(monkeypatch):
    _setup()
    coords, types = _case(3)
    r = bt.radii_for(types)
    monkeypatch.setattr(bt, "_cython_bonds", None)
    assert bt.bonds_from_radii(coords, r) == bt._reference_bonds(coords, types)


def test_radii_for_is_reusable_across_geometries():
    """The optimisation the twist loop relies on: radii are geometry-independent."""
    _setup()
    _, types = _case(5)
    r1 = bt.radii_for(types)
    r2 = bt.radii_for(types)
    assert np.array_equal(r1, r2)
