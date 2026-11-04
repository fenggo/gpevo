"""The Cython newMolCheck kernel must agree with the pure-numpy reference.

Both helpers take the same prepared buffers, so this comparison is independent
of ORG_STRUC / STDMOL.  A disagreement means damaged molecules could silently
be accepted (or good ones rejected), which would corrupt the whole search.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpevo.FunctionFolder.USPEX.src import newMolCheck as nmc

_cy = pytest.importorskip("gpevo.FunctionFolder.sys.molcheck_cy")


def _case(seed, n_mols=8, clash=False, skew=False):
    rng = np.random.default_rng(seed)
    n_types = 4
    thr = rng.uniform(0.9, 2.0, size=(n_types, n_types))
    min_matrix = np.ascontiguousarray(0.5 * (thr + thr.T))
    lat = np.diag([40.0, 40.0, 40.0])
    if skew:
        lat = lat + rng.uniform(-2.0, 2.0, size=(3, 3))
    inv_lat = np.linalg.inv(lat)
    centers = np.array(
        [[10, 10, 10], [10, 10, 30], [10, 30, 10], [10, 30, 30],
         [30, 10, 10], [30, 10, 30], [30, 30, 10], [30, 30, 30]],
        dtype=float)[:n_mols]
    mol_coords, mol_types = [], []
    for k, ctr in enumerate(centers):
        n = int(rng.integers(20, 37))
        c = np.ascontiguousarray(ctr + rng.normal(0, 1.0, size=(n, 3)))
        t = [int(x) for x in rng.integers(1, n_types + 1, size=n)]
        mol_coords.append(c)
        mol_types.append(t)
    if clash and n_mols >= 2:
        # put one atom of molecule 1 on top of one of molecule 0
        mol_coords[1] = mol_coords[1].copy()
        mol_coords[1][0] = mol_coords[0][0]
    return mol_coords, mol_types, np.ascontiguousarray(lat), inv_lat, min_matrix


def _both(case):
    coords, types, lat, inv_lat, mm = case
    py = nmc._python_molcheck(coords, types, lat, inv_lat, mm)
    cy = nmc._cython_call(coords, types, lat, inv_lat, mm)
    return py, cy


@pytest.mark.parametrize("seed", range(12))
def test_kernel_matches_reference_on_random_structures(seed):
    py, cy = _both(_case(seed))
    assert (py, cy) in ((0, 0), (1, 1)), (seed, py, cy)


@pytest.mark.parametrize("seed", range(6))
def test_kernel_matches_reference_on_rejected_structures(seed):
    py, cy = _both(_case(seed, clash=True))
    assert py == 0 and cy == 0


@pytest.mark.parametrize("seed", range(6))
def test_kernel_matches_reference_on_skewed_lattices(seed):
    py, cy = _both(_case(seed, skew=True))
    assert py == cy


def test_kernel_matches_reference_on_small_and_empty_molecules():
    mm = np.ascontiguousarray(np.full((4, 4), 1.0))
    lat = np.diag([40.0, 40.0, 40.0])
    inv_lat = np.linalg.inv(lat)
    coords = [np.ascontiguousarray([[0.0, 0.0, 0.0]]),
              np.ascontiguousarray(np.zeros((0, 3))),
              np.ascontiguousarray([[20.0, 0.0, 0.0], [20.0, 1.5, 0.0]])]
    types = [[1], [], [2, 3]]
    assert nmc._python_molcheck(coords, types, lat, inv_lat, mm) == \
           nmc._cython_call(coords, types, lat, inv_lat, mm)


def test_half_integer_fractional_offsets_agree():
    """np.round is half-to-even; a half-cell separation exercises the boundary."""
    mm = np.ascontiguousarray(np.full((4, 4), 1.0))
    lat = np.diag([10.0, 10.0, 10.0])
    inv_lat = np.linalg.inv(lat)
    coords = [np.ascontiguousarray([[0.0, 0.0, 0.0]]),
              np.ascontiguousarray([[5.0, 5.0, 5.0]])]
    types = [[1], [2]]
    assert nmc._python_molcheck(coords, types, lat, inv_lat, mm) == 1
    assert nmc._cython_call(coords, types, lat, inv_lat, mm) == 1


def test_negative_threshold_never_rejects():
    mm = np.ascontiguousarray(np.full((4, 4), -1.0))
    lat = np.diag([40.0, 40.0, 40.0])
    inv_lat = np.linalg.inv(lat)
    coords, types, _, _, _ = _case(0, n_mols=2)
    assert nmc._python_molcheck(coords, types, lat, inv_lat, mm) == \
           nmc._cython_call(coords, types, lat, inv_lat, mm)


def test_newMolCheck_end_to_end_uses_the_extension(monkeypatch):
    from gpevo.runtime import G, AttrDict

    case = _case(3)
    mol_coords, mol_types, lat, _, mm = case
    ORG = AttrDict(atomType=[6, 8, 7, 1])
    ORG.STDMOL = [AttrDict(types=t) for t in mol_types]
    monkeypatch.setitem(G, "ORG_STRUC", ORG)
    molecules = [AttrDict(MOLCOORS=c) for c in mol_coords]
    mtypes = list(range(1, len(mol_coords) + 1))

    assert nmc._cython_molcheck is not None
    assert nmc.newMolCheck(molecules, lat, mtypes, mm) in (0, 1)


def test_newMolCheck_falls_back_without_the_extension(monkeypatch):
    from gpevo.runtime import G, AttrDict

    monkeypatch.setattr(nmc, "_cython_molcheck", None)
    case = _case(3)
    mol_coords, mol_types, lat, _, mm = case
    ORG = AttrDict(atomType=[6, 8, 7, 1])
    ORG.STDMOL = [AttrDict(types=t) for t in mol_types]
    monkeypatch.setitem(G, "ORG_STRUC", ORG)
    molecules = [AttrDict(MOLCOORS=c) for c in mol_coords]
    mtypes = list(range(1, len(mol_coords) + 1))

    py = nmc._python_molcheck(mol_coords, mol_types, lat,
                              np.linalg.inv(lat), mm)
    assert nmc.newMolCheck(molecules, lat, mtypes, mm) == py
