"""Regression tests for the swarm operator's reference frame and pairing.

Covers docs/swarm_operator.md 5.2 A (MtypeLIST pairing) and 5.2 B (cell
alignment).  The B bug: the best structure's Cartesian centres were folded
with the child's lattice, mixing two reference frames; _min_image now maps
the target through the best lattice first.
"""
import copy
import importlib

import numpy as np
import pytest

from gpevo.runtime import G, AttrDict

_swarm = importlib.import_module(
    "gpevo.FunctionFolder.USPEX.310.SwarmMutation_310")
_min_image = _swarm._min_image


def _mol(coords):
    c = np.asarray(coords, dtype=float)
    return AttrDict(MOLCOORS=c.copy(), MOLCENTER=c.mean(0), ZMATRIX=c.copy())


def _ind(mol_coords, mtype_list, lattice):
    return AttrDict(
        MOLECULES=[_mol(c) for c in mol_coords],
        MtypeLIST=list(mtype_list),
        LATTICE=np.asarray(lattice, dtype=float),
    )


def _install(monkeypatch, parent, best, step=0.5, angle=0.0):
    """Wire a deterministic operator call; return the captured offspring list."""
    produced = []
    monkeypatch.setitem(G, "ORG_STRUC", AttrDict(swarmStep=step, swarmAngle=angle))
    monkeypatch.setitem(G, "POP_STRUC",
                        AttrDict(POPULATION=[parent, best], ranking=[1]))
    monkeypatch.setitem(G, "OFF_STRUC", AttrDict(POPULATION=[]))
    monkeypatch.setattr(_swarm, "pick_parent", lambda: 0)
    monkeypatch.setattr(_swarm, "copy_individual", lambda ind: copy.deepcopy(ind))
    monkeypatch.setattr(_swarm, "valid_molecules", lambda *a, **k: True)
    monkeypatch.setattr(_swarm, "recompute_zmatrix", lambda *a, **k: None)
    monkeypatch.setattr(_swarm, "finalize_offspring",
                        lambda child, how: produced.append(child))
    # step factor = step * 0.5, rotation angle = 0 (so coords stay a pure
    # rigid translation and the expected centre is computable in closed form).
    monkeypatch.setattr(np.random, "rand", lambda: 0.5)
    monkeypatch.setattr(np.random, "randn", lambda n: np.zeros(n))
    return produced


# ---------------------------------------------------------------------------
# _min_image: single-frame folding primitive
# ---------------------------------------------------------------------------

def test_min_image_single_frame_folds_into_half_cell():
    L = np.diag([10.0, 10.0, 10.0])
    zero = np.zeros(3)
    # 0.7 -> folds to -0.3; np.round uses half-even, so exactly 0.5 stays +.
    assert _min_image(np.array([7.0, 0, 0]), zero, L)[0] == pytest.approx(-3.0)
    assert _min_image(np.array([5.0, 0, 0]), zero, L)[0] == pytest.approx(5.0)
    assert _min_image(np.array([-5.0, 0, 0]), zero, L)[0] == pytest.approx(-5.0)
    assert _min_image(np.array([-6.0, 0, 0]), zero, L)[0] == pytest.approx(4.0)


def test_min_image_lattice_vector_folds_to_zero_triclinic():
    # Row-vector convention (frac = cart @ inv(cell)) must also hold for a
    # non-orthogonal cell: any integer combination of lattice rows folds to 0.
    L = np.array([[10.0, 0.0, 0.0],
                  [2.0, 12.0, 0.0],
                  [1.0, 3.0, 15.0]])
    zero = np.zeros(3)
    for v in (L[0], L[1], L[2], L[0] + 2 * L[2]):
        assert np.allclose(_min_image(v, zero, L), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Regression: best and child live in different cells (5.2 B)
# ---------------------------------------------------------------------------

def test_min_image_aligns_best_cell_into_child_frame():
    best_L = np.diag([10.0, 30.0, 30.0])
    child_L = np.diag([12.0, 30.0, 30.0])
    center = np.array([2.0, 0.0, 0.0])       # child frame, fraction 1/6
    target = np.array([8.0, 0.0, 0.0])      # best frame, fraction 0.8

    # Legacy behaviour (no source lattice): the mixed-frame difference 8-2
    # folds with the child cell and points the wrong way.
    legacy = _min_image(target, center, child_L)
    assert legacy[0] == pytest.approx(6.0)

    # Fixed: 8 A of the a=10 cell -> fraction 0.8 -> 9.6 A in the a=12 cell;
    # min_image(9.6 - 2, 12) = -4.4 A.
    fixed = _min_image(target, center, child_L, best_L)
    assert fixed[0] == pytest.approx(-4.4)
    assert np.sign(fixed[0]) != np.sign(legacy[0])


def test_min_image_same_source_lattice_is_identity():
    L = np.diag([12.0, 17.0, 19.0])
    center = np.array([2.0, 1.0, 3.0])
    target = np.array([9.5, -4.0, 11.0])
    assert np.allclose(_min_image(target, center, L, L),
                       _min_image(target, center, L))


# ---------------------------------------------------------------------------
# Regression: MtypeLIST consistency assertion (5.2 A)
# ---------------------------------------------------------------------------

def test_swarm_asserts_on_mtype_list_mismatch(monkeypatch):
    parent = _ind([[[1.5, 0, 0], [2.5, 0, 0]],
                   [[5.5, 0, 0], [6.5, 0, 0]]],
                  [1, 2, 1, 2], np.diag([12.0, 30.0, 30.0]))
    # Same molecule count, but a reordered type list: positional pairing
    # would silently cross molecule types.
    parent.MOLECULES = parent.MOLECULES + parent.MOLECULES
    best = _ind([[[7.5, 0, 0], [8.5, 0, 0]]] * 4,
                [1, 1, 2, 2], np.diag([10.0, 30.0, 30.0]))
    _install(monkeypatch, parent, best)

    with pytest.raises(AssertionError, match="MtypeLIST"):
        _swarm.SwarmMutation_310(1)


def test_swarm_grouped_mtype_list_is_accepted(monkeypatch):
    # The real invariant from GetPOP_MOL: [1]*n1 + [2]*n2, identical on both
    # sides.  The assertion must not fire on a well-formed population.
    parent = _ind([[[i + 1.0, 0, 0], [i + 1.5, 0, 0]] for i in range(4)],
                  [1, 1, 2, 2], np.diag([12.0, 30.0, 30.0]))
    best = _ind([[[i + 2.0, 0, 0], [i + 2.5, 0, 0]] for i in range(4)],
                [1, 1, 2, 2], np.diag([10.0, 30.0, 30.0]))
    produced = _install(monkeypatch, parent, best)

    _swarm.SwarmMutation_310(1)

    assert len(produced) == 1
    assert list(produced[0].MtypeLIST) == [1, 1, 2, 2]


def test_swarm_moves_toward_aligned_best_centre(monkeypatch):
    """End-to-end: mismatched cells, deterministic step, zero rotation."""
    parent = _ind([[[1.5, 0, 0], [2.5, 0, 0]],      # centre 2.0
                   [[5.5, 0, 0], [6.5, 0, 0]]],     # centre 6.0
                  [1, 1], np.diag([12.0, 30.0, 30.0]))
    best = _ind([[[7.5, 0, 0], [8.5, 0, 0]],       # centre 8.0 (frac 0.8)
                 [[0.5, 0, 0], [1.5, 0, 0]]],      # centre 1.0 (frac 0.1)
                [1, 1], np.diag([10.0, 30.0, 30.0]))
    produced = _install(monkeypatch, parent, best)

    _swarm.SwarmMutation_310(1)

    assert len(produced) == 1
    off = produced[0]
    # step=0.5, rand()=0.5 -> factor 0.25.  Aligned delta for molecule 0 is
    # -4.4 A, so centre 2.0 -> 0.9 A (the buggy +6.0 would give 3.5 A).
    first = off.MOLECULES[0]
    assert first.MOLCENTER[0] == pytest.approx(0.9)
    assert first.MOLCOORS.mean(0)[0] == pytest.approx(0.9)
    # Rigid translation: intramolecular geometry is bit-for-bit preserved.
    assert np.allclose(first.MOLCOORS[1] - first.MOLCOORS[0], [1.0, 0.0, 0.0])
    # The child keeps its own cell; only molecule coordinates change.
    assert np.allclose(off.LATTICE, parent.LATTICE)
