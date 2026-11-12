"""Regression tests for the swarm operator's reference frame and pairing.

Covers docs/swarm_operator.md 5.2 A (MtypeLIST pairing) and 5.2 B (cell
alignment).  The B bug: the best structure's Cartesian centres were folded
with the child's lattice, mixing two reference frames; _min_image now maps
the target through the best lattice first.
"""
import copy
import importlib
import pickle

import numpy as np
import pytest

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.USPEX.src.Rotate_rigid_body import (
    Rotate_rigid_body as _rotate,
)

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


def _install(monkeypatch, parent, best, step=0.5, angle=0.0, rand_value=0.5,
             inertia=0.0, cognitive=0.0, jitter=0.0):
    """Wire a deterministic operator call; return the captured offspring list.

    jitter defaults to 0 so the existing closed-form centre assertions stay
    deterministic; the collapse regression tests opt in to a positive jitter.
    """
    produced = []
    monkeypatch.setitem(G, "ORG_STRUC", AttrDict(
        swarmStep=step, swarmAngle=angle,
        swarmInertia=inertia, swarmCognitive=cognitive,
        swarmJitter=jitter))
    monkeypatch.setitem(G, "POP_STRUC",
                        AttrDict(POPULATION=[parent, best], ranking=[1]))
    monkeypatch.setitem(G, "OFF_STRUC", AttrDict(POPULATION=[]))
    monkeypatch.setattr(_swarm, "pick_parent", lambda: 0)
    monkeypatch.setattr(_swarm, "copy_individual", lambda ind: copy.deepcopy(ind))
    monkeypatch.setattr(_swarm, "valid_molecules", lambda *a, **k: True)
    monkeypatch.setattr(_swarm, "recompute_zmatrix", lambda *a, **k: None)
    monkeypatch.setattr(_swarm, "finalize_offspring",
                        lambda child, how: produced.append(child))
    # rand() is pinned for both the position step and the slerp fraction;
    # rotation angle = 0 unless a test opts in (so the centre is computable
    # in closed form).
    monkeypatch.setattr(np.random, "rand", lambda: rand_value)
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


# ---------------------------------------------------------------------------
# Orientation: Kabsch + quaternion slerp toward g (5.2 C)
# ---------------------------------------------------------------------------

def _tetra():
    """Four full-rank, centred points (a linear molecule has no orientation)."""
    return np.array([[1.0, 1.0, 1.0],
                     [1.0, -1.0, -1.0],
                     [-1.0, 1.0, -1.0],
                     [-1.0, -1.0, 1.0]])


def _rotation_matrix(axis, angle):
    return _rotate(np.zeros(3), np.asarray(axis, float), np.eye(3), angle)


def _orientation_inds(angle_deg):
    R = _rotation_matrix([0.0, 0.0, 1.0], np.radians(angle_deg))
    parent = _ind([_tetra() + [2.0, 0.0, 0.0]], [1],
                  np.diag([12.0, 30.0, 30.0]))
    best = _ind([(_tetra() @ R) + [8.0, 0.0, 0.0]], [1],
                np.diag([10.0, 30.0, 30.0]))
    return parent, best, R


def _rotation_angle(P, Q):
    q = _swarm._quat_from_matrix(_swarm._kabsch_rotation(P, Q))
    return float(np.degrees(2.0 * np.arccos(min(1.0, abs(q[0])))))


def test_kabsch_rotation_recovers_known_rotation():
    rng = np.random.default_rng(3)
    for _ in range(50):
        P = rng.normal(size=(6, 3))
        theta = rng.uniform(0.01, np.pi - 0.01)
        R = _rotation_matrix(rng.normal(size=3), theta)
        got = _swarm._kabsch_rotation(P, P @ R)
        assert np.allclose(P @ got, P @ R)
        assert np.linalg.det(got) == pytest.approx(1.0)


def test_quat_matrix_roundtrip():
    rng = np.random.default_rng(4)
    for _ in range(50):
        R = _rotation_matrix(rng.normal(size=3), rng.uniform(0.01, np.pi - 0.01))
        q = _swarm._quat_from_matrix(R)
        assert np.linalg.norm(q) == pytest.approx(1.0)
        assert np.allclose(_swarm._quat_to_matrix(q), R, atol=1e-12)
    assert np.allclose(_swarm._quat_to_matrix(_swarm._quat_from_matrix(np.eye(3))),
                       np.eye(3))


def test_quat_slerp_endpoints_and_shortest_path():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = _swarm._quat_from_matrix(_rotation_matrix([0.0, 0.0, 1.0], np.pi / 2))
    assert np.allclose(_swarm._quat_slerp(q0, q1, 0.0), q0)
    assert np.allclose(_swarm._quat_slerp(q0, q1, 1.0), q1)
    # q and -q encode the same rotation -> same slerp result.
    assert np.allclose(_swarm._quat_slerp(q0, -q1, 0.4),
                       _swarm._quat_slerp(q0, q1, 0.4))
    # t=0.5 is the geodesic midpoint, i.e. half the rotation angle.
    R_half = _swarm._quat_to_matrix(_swarm._quat_slerp(q0, q1, 0.5))
    assert np.allclose(R_half, _rotation_matrix([0.0, 0.0, 1.0], np.pi / 4),
                       atol=1e-12)


def test_swarm_orientation_slerps_toward_best(monkeypatch):
    parent, best, R = _orientation_inds(60.0)
    produced = _install(monkeypatch, parent, best, angle=180.0, rand_value=1.0)

    _swarm.SwarmMutation_310(1)

    off = produced[0].MOLECULES[0]
    centred = off.MOLCOORS - off.MOLCOORS.mean(0)
    # t=1 -> the offspring's orientation is exactly g's.
    assert np.allclose(centred, _tetra() @ R, atol=1e-12)
    # Rigid rotation: every atom keeps its distance from the centre.
    assert np.allclose(np.linalg.norm(centred, axis=1),
                       np.linalg.norm(_tetra(), axis=1))


def test_swarm_orientation_honours_slerp_fraction(monkeypatch):
    parent, best, _ = _orientation_inds(60.0)
    produced = _install(monkeypatch, parent, best, angle=180.0, rand_value=0.5)

    _swarm.SwarmMutation_310(1)

    centred = produced[0].MOLECULES[0].MOLCOORS
    centred = centred - centred.mean(0)
    assert _rotation_angle(_tetra(), centred) == pytest.approx(30.0, abs=1e-6)


def test_swarm_angle_caps_orientation_step(monkeypatch):
    parent, best, _ = _orientation_inds(120.0)
    produced = _install(monkeypatch, parent, best, angle=10.0, rand_value=1.0)

    _swarm.SwarmMutation_310(1)

    centred = produced[0].MOLECULES[0].MOLCOORS
    centred = centred - centred.mean(0)
    assert _rotation_angle(_tetra(), centred) == pytest.approx(10.0, abs=1e-6)


def test_swarm_zero_angle_leaves_orientation_unchanged(monkeypatch):
    # swarmAngle=0 must not apply any rotation at all: the old operator drew a
    # random axis/angle, this one must leave the parent's orientation intact.
    parent, best, _ = _orientation_inds(60.0)
    produced = _install(monkeypatch, parent, best, angle=0.0, rand_value=1.0)

    _swarm.SwarmMutation_310(1)

    centred = produced[0].MOLECULES[0].MOLCOORS
    centred = centred - centred.mean(0)
    assert np.allclose(centred, _tetra())


# ---------------------------------------------------------------------------
# pbest / inertia: per-individual velocity state (6.4)
# ---------------------------------------------------------------------------

from gpevo.FunctionFolder.USPEX.src import swarm_state as _state  # noqa: E402


def test_swarm_state_survives_pickle_and_deepcopy():
    ind = _ind([_tetra() + [1.0, 2.0, 3.0]], [1], np.diag([12.0, 13.0, 14.0]))
    st = _state.ensure_swarm_state(ind)
    st.v_cart[:] = [0.1, -0.2, 0.3]
    st.q_v[:] = [np.cos(0.2), 0.0, 0.0, np.sin(0.2)]
    st.pbest_fitness = -3.5

    restored = pickle.loads(pickle.dumps(ind))
    assert isinstance(restored.swarm_state, dict)
    assert restored.swarm_state.pbest_fitness == -3.5
    assert np.allclose(restored.swarm_state.v_cart, st.v_cart)
    assert np.allclose(restored.swarm_state.q_v, st.q_v)
    assert np.allclose(restored.swarm_state.pbest_frac, st.pbest_frac)
    assert np.allclose(restored.swarm_state.pbest_coords[0], st.pbest_coords[0])

    cloned = copy.deepcopy(ind)
    assert np.allclose(cloned.swarm_state.v_cart, st.v_cart)
    assert np.allclose(cloned.swarm_state.q_v, st.q_v)


def test_swarm_state_survives_current_pop_checkpoint(tmp_path):
    from gpevo.runtime import safesave, safeload

    ind = _ind([_tetra() + [2.0, 0.0, 0.0]], [1], np.diag([12.0, 30.0, 30.0]))
    st = _state.ensure_swarm_state(ind)
    st.v_cart[:] = [0.25, -0.5, 0.75]
    st.pbest_fitness = -7.25
    POP = AttrDict(POPULATION=[ind], generation=3)

    safesave("Current_POP.pkl", POP, workdir=str(tmp_path))
    restored = safeload("Current_POP.pkl", workdir=str(tmp_path))

    rstate = restored.POPULATION[0].swarm_state
    assert np.allclose(rstate.v_cart[0], [0.25, -0.5, 0.75])
    assert rstate.pbest_fitness == -7.25
    assert np.allclose(rstate.q_v[0], [1.0, 0.0, 0.0, 0.0])


def test_ensure_swarm_state_initialises_fresh_individual():
    coords = _tetra() + [2.0, 0.0, 0.0]
    ind = _ind([coords], [1], np.diag([12.0, 30.0, 30.0]))
    st = _state.ensure_swarm_state(ind)
    assert st.v_cart.shape == (1, 3) and np.allclose(st.v_cart, 0.0)
    assert st.q_v.shape == (1, 4) and np.allclose(st.q_v[0], [1, 0, 0, 0])
    expected = np.asarray([2.0, 0.0, 0.0]) @ np.linalg.inv(np.diag([12.0, 30.0, 30.0]))
    assert np.allclose(st.pbest_frac[0], expected)
    assert np.allclose(st.pbest_coords[0], _tetra())
    assert st.pbest_fitness is None


def test_ensure_swarm_state_recovers_legacy_state():
    ind = _ind([_tetra()], [1], np.diag([12.0, 30.0, 30.0]))
    ind.swarm_state = AttrDict(v_cart=np.zeros((2, 3)))   # stale / old checkpoint
    st = _state.ensure_swarm_state(ind)
    assert st.v_cart.shape == (1, 3)
    assert st.q_v.shape == (1, 4)
    assert st.pbest_fitness is None

    ind2 = _ind([_tetra()], [1], np.diag([12.0, 30.0, 30.0]))
    ind2.swarm_state = None
    assert _state.ensure_swarm_state(ind2).v_cart.shape == (1, 3)


def test_update_pbest_tracks_minimum_fitness():
    a = _ind([_tetra() + [2.0, 0.0, 0.0]], [1], np.diag([12.0, 30.0, 30.0]))
    b = _ind([_tetra() + [5.0, 0.0, 0.0]], [1], np.diag([12.0, 30.0, 30.0]))
    POP = AttrDict(POPULATION=[a, b])

    _state.update_pbest(POP, np.array([-3.0, -5.0]))
    assert a.swarm_state.pbest_fitness == -3.0
    assert b.swarm_state.pbest_fitness == -5.0
    assert np.allclose(a.swarm_state.pbest_frac[0], [2.0 / 12.0, 0.0, 0.0])
    assert np.allclose(b.swarm_state.pbest_frac[0], [5.0 / 12.0, 0.0, 0.0])

    # Worse fitness and a failed structure must not overwrite the personal best.
    _state.update_pbest(POP, np.array([-1.0, 100000.0]))
    assert a.swarm_state.pbest_fitness == -3.0
    assert b.swarm_state.pbest_fitness == -5.0

    # A genuine improvement refreshes the snapshot.
    a.MOLECULES[0].MOLCOORS = _tetra() + [7.0, 0.0, 0.0]
    a.MOLECULES[0].MOLCENTER = a.MOLECULES[0].MOLCOORS.mean(0)
    _state.update_pbest(POP, np.array([-9.0, -1.0]))
    assert a.swarm_state.pbest_fitness == -9.0
    assert np.allclose(a.swarm_state.pbest_frac[0], [7.0 / 12.0, 0.0, 0.0])
    assert np.allclose(a.swarm_state.pbest_coords[0], _tetra())


def test_calc_fitness_refreshes_pbest(monkeypatch):
    cf = importlib.import_module("gpevo.FunctionFolder.USPEX.310.CalcFitness_310")
    monkeypatch.setattr(cf, "gp_active_learning", lambda ORG, POP, f: f)
    monkeypatch.setattr(cf, "gp_enabled", lambda ORG: False)
    ind = _ind([_tetra() + [2.0, 0.0, 0.0]], [1], np.diag([12.0, 30.0, 30.0]))
    ind.Enthalpies = np.array([100.0])
    ind.numMols = [1]
    POP = AttrDict(POPULATION=[ind], generation=1, fitness=[])
    ORG = AttrDict(optType=1, opt_sign=1, abinitioCode=[1], atomType=[6],
                   numMols=[1], gp=0, resFolder=".", homePath=".")
    monkeypatch.setitem(G, "POP_STRUC", POP)
    monkeypatch.setitem(G, "ORG_STRUC", ORG)

    cf.CalcFitness_310()

    assert ind.swarm_state.pbest_fitness == pytest.approx(100.0)
    assert np.allclose(ind.swarm_state.pbest_frac[0], [2.0 / 12.0, 0.0, 0.0])


def test_swarm_inertia_moves_by_carried_velocity(monkeypatch):
    parent, best, _ = _orientation_inds(0.0)
    st = _state.ensure_swarm_state(parent)
    st.v_cart[:] = [1.0, 0.0, 0.0]
    produced = _install(monkeypatch, parent, best, step=0.0, angle=0.0,
                        inertia=1.0, cognitive=0.0, rand_value=1.0)

    _swarm.SwarmMutation_310(1)

    centre = produced[0].MOLECULES[0].MOLCOORS.mean(0)
    assert centre[0] == pytest.approx(3.0)          # 2.0 + inherited 1.0 A
    assert np.allclose(produced[0].swarm_state.v_cart[0], [1.0, 0.0, 0.0])


def test_swarm_cognitive_pulls_toward_pbest(monkeypatch):
    parent, best, _ = _orientation_inds(0.0)
    st = _state.ensure_swarm_state(parent)
    st.pbest_frac[0] = [5.0 / 12.0, 0.0, 0.0]       # personal best at x=5 A
    st.pbest_fitness = -10.0
    produced = _install(monkeypatch, parent, best, step=0.0, angle=0.0,
                        inertia=0.0, cognitive=1.0, rand_value=1.0)

    _swarm.SwarmMutation_310(1)

    centre = produced[0].MOLECULES[0].MOLCOORS.mean(0)
    assert centre[0] == pytest.approx(5.0)          # 2.0 + min_image(5-2)=3
    assert np.allclose(produced[0].swarm_state.v_cart[0], [3.0, 0.0, 0.0])


def test_swarm_defaults_ignore_stale_velocity(monkeypatch):
    parent, best, _ = _orientation_inds(0.0)
    st = _state.ensure_swarm_state(parent)
    st.v_cart[:] = [9.0, 9.0, 9.0]
    st.q_v[:] = [np.cos(0.5), 0.0, 0.0, np.sin(0.5)]
    produced = _install(monkeypatch, parent, best, step=0.5, angle=0.0,
                        inertia=0.0, cognitive=0.0, rand_value=0.5)

    _swarm.SwarmMutation_310(1)

    centre = produced[0].MOLECULES[0].MOLCOORS.mean(0)
    assert centre[0] == pytest.approx(0.9)          # social term only
    assert np.allclose(produced[0].swarm_state.v_cart[0], [-1.1, 0.0, 0.0])
