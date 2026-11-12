"""optType 14: fitness is the fingerprint cosine distance to a target structure.

The helpers rebuild an external POSCAR/CIF/.traj fingerprint with exactly the
convention of ``compute_structure_metrics`` (fractional coords, species-blocked
in ORG.atomType order, zero-shift intra-molecular pairs dropped), so a distance
of 0 really means "the same crystal structure".
"""
from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest

from gpevo.runtime import G, struct

_cf = import_module("gpevo.FunctionFolder.USPEX.310.CalcFitness_310")
CalcFitness_310 = _cf.CalcFitness_310

_rng = np.random.default_rng(14)
FP_A = _rng.random((4, 6)) + 0.1
FP_B = _rng.random((4, 6)) + 0.1
FP_C = _rng.random((4, 6)) + 0.1


def _org14(**kw):
    base = dict(optType=14, gp=0, opt_sign=1, bestFrac=0.7, varcomp=0,
                weight=1.0, numMols=[1], atomType=[1], numIons=[2],
                molecule=0, dimension=3, RmaxFing=10.0, sigmaFing=0.03,
                deltaFing=0.08, homePath=".", resFolder=".",
                targetStructure=[], targetBondCut=0.0, minDistMatrice=None)
    base.update(kw)
    return struct(**base)


def _individual14(fing=None, enthalpy=-100.0):
    s = struct(Enthalpies=[enthalpy], numIons=[2], numMols=[1])
    if fing is not None:
        s.FINGERPRINT = np.asarray(fing, dtype=float)
    return s


def _run(monkeypatch, org, individuals, targets=None):
    monkeypatch.setattr(_cf, "gp_active_learning", lambda ORG, POP, f: f)
    if targets is not None:
        monkeypatch.setattr(_cf, "_target_fingerprints", lambda ORG: targets)
    monkeypatch.setitem(G, "ORG_STRUC", org)
    monkeypatch.setitem(G, "POP_STRUC", struct(
        generation=0, POPULATION=individuals))
    return CalcFitness_310()


# --- fitness branch ---------------------------------------------------------

def test_identical_fingerprint_is_zero(monkeypatch):
    out = _run(monkeypatch, _org14(), [_individual14(FP_A)], targets=[FP_A])
    assert out[0] == pytest.approx(0.0, abs=1e-12)


def test_different_fingerprint_is_positive(monkeypatch):
    same = _run(monkeypatch, _org14(), [_individual14(FP_A)], targets=[FP_A])[0]
    diff = _run(monkeypatch, _org14(), [_individual14(FP_B)], targets=[FP_A])[0]
    assert 0.0 < diff < 1.0
    assert diff > same


def test_multiple_targets_takes_the_minimum(monkeypatch):
    expected = min(_cf._fp_distance(FP_C, FP_A, _org14()),
                   _cf._fp_distance(FP_C, FP_B, _org14()))
    out = _run(monkeypatch, _org14(), [_individual14(FP_C)],
               targets=[FP_A, FP_B])
    assert out[0] == pytest.approx(expected)


def test_missing_fingerprint_gets_max_distance(monkeypatch, capsys):
    out = _run(monkeypatch, _org14(),
               [_individual14(None), _individual14(np.zeros((0, 6)))],
               targets=[FP_A])
    assert list(out) == pytest.approx([1.0, 1.0])
    assert "no FINGERPRINT" in capsys.readouterr().out


def test_failed_structure_still_wins(monkeypatch):
    out = _run(monkeypatch, _org14(),
               [_individual14(FP_A, enthalpy=100000.0)], targets=[FP_A])
    assert out[0] == pytest.approx(100000.0)


def test_opt_sign_inverts_the_objective(monkeypatch):
    d = _cf._fp_distance(FP_B, FP_A, _org14())
    out = _run(monkeypatch, _org14(opt_sign=-1),
               [_individual14(FP_B)], targets=[FP_A])
    assert out[0] == pytest.approx(-d)


def test_scalar_weight_is_normalized(monkeypatch):
    out = _run(monkeypatch, _org14(weight=1.0),
               [_individual14(FP_A)], targets=[FP_A])
    assert out[0] == pytest.approx(0.0, abs=1e-12)


def test_shape_mismatch_raises():
    with pytest.raises(RuntimeError, match="shape"):
        _cf._fp_distance(np.ones((4, 6)), np.ones((4, 7)), _org14())


def test_missing_target_config_raises(monkeypatch):
    monkeypatch.setattr(_cf, "gp_active_learning", lambda ORG, POP, f: f)
    monkeypatch.setitem(G, "ORG_STRUC", _org14(targetStructure=[]))
    monkeypatch.setitem(G, "POP_STRUC", struct(
        generation=0, POPULATION=[_individual14(FP_A)]))
    with pytest.raises(RuntimeError, match="targetStructure"):
        CalcFitness_310()


# --- target fingerprint helpers ---------------------------------------------

def test_bond_cut_defaults_and_override():
    assert _cf._target_bond_cut(_org14(
        targetBondCut=0.0, minDistMatrice=[[1.9, 1.8], [1.8, 1.75]])) == 1.9
    assert _cf._target_bond_cut(_org14(targetBondCut=2.1)) == 2.1
    assert _cf._target_bond_cut(_org14(
        targetBondCut=0.0, minDistMatrice=None)) == 1.9
    # the real ORG stores minDistMatrice as a numpy array
    assert _cf._target_bond_cut(_org14(
        targetBondCut=0.0,
        minDistMatrice=np.array([[1.9, 1.85], [1.85, 1.75]]))) == 1.9


def test_molecule_ids_groups_bonded_atoms():
    D = np.full((6, 6), 10.0)
    np.fill_diagonal(D, 0.0)
    for a, b in ((0, 1), (0, 2), (3, 4)):
        D[a, b] = D[b, a] = 1.2
    mol = _cf._molecule_ids(D, 1.9)
    assert len(set(mol.tolist())) == 3
    assert mol[0] == mol[1] == mol[2]
    assert mol[3] == mol[4]
    assert mol[5] != mol[0] and mol[5] != mol[3]


def _write_poscar(path):
    from ase import Atoms
    from ase.io import write
    atoms = Atoms(symbols=["C", "C"], cell=np.eye(3) * 5.0,
                  scaled_positions=[[0.1, 0.1, 0.1], [0.6, 0.6, 0.6]],
                  pbc=True)
    write(str(path), atoms, format="vasp", direct=True)


def _file_org(path, **kw):
    return _org14(atomType=[6], numIons=[2], molecule=0,
                  RmaxFing=5.0, deltaFing=0.5, sigmaFing=0.1,
                  targetStructure=[str(path)], **kw)


def test_fingerprint_from_poscar(tmp_path, monkeypatch):
    path = tmp_path / "POSCAR"
    _write_poscar(path)
    org = _file_org(path)
    monkeypatch.setitem(G, "ORG_STRUC", org)
    _cf._TARGET_FP_CACHE.clear()
    fps = _cf._target_fingerprints(org)
    assert len(fps) == 1
    assert fps[0].shape == (1, 10)   # species^2 x round(RmaxFing/deltaFing)
    assert np.all(np.isfinite(fps[0]))
    assert _cf._fp_distance(fps[0], fps[0], org) == pytest.approx(0.0, abs=1e-12)


def test_multiframe_trajectory_is_multiple_targets(tmp_path, monkeypatch):
    from ase import Atoms
    from ase.io.trajectory import TrajectoryWriter
    path = str(tmp_path / "t.traj")
    writer = TrajectoryWriter(path, mode="w")
    for x in (0.10, 0.15):
        writer.write(Atoms(symbols=["C", "C"], cell=np.eye(3) * 5.0,
                           scaled_positions=[[x, x, x],
                                             [x + 0.5, x + 0.5, x + 0.5]],
                           pbc=True))
    writer.close()
    org = _file_org(path)
    monkeypatch.setitem(G, "ORG_STRUC", org)
    _cf._TARGET_FP_CACHE.clear()
    fps = _cf._target_fingerprints(org)
    assert len(fps) == 2
    assert _cf._fp_distance(fps[0], fps[1], org) > 0.0


def test_target_fingerprints_caches_file_reads(tmp_path, monkeypatch):
    import ase.io
    path = tmp_path / "POSCAR"
    _write_poscar(path)
    org = _file_org(path)
    monkeypatch.setitem(G, "ORG_STRUC", org)
    real_read = ase.io.read
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(ase.io, "read", counting)
    _cf._TARGET_FP_CACHE.clear()
    first = _cf._target_fingerprints(org)
    second = _cf._target_fingerprints(org)
    assert len(calls) == 1
    assert first[0] is second[0]


def test_unreadable_target_raises(tmp_path):
    org = _org14(targetStructure=[str(tmp_path / "nope.cif")])
    with pytest.raises(RuntimeError, match="missing or unreadable"):
        _cf._target_fingerprints(org)


def test_non_integer_cell_multiple_is_rejected(tmp_path, monkeypatch):
    from ase import Atoms
    from ase.io import write
    path = tmp_path / "POSCAR"
    write(str(path), Atoms(symbols=["C", "C", "C"], cell=np.eye(3) * 5.0,
                           scaled_positions=[[0.1, 0.1, 0.1],
                                             [0.4, 0.4, 0.4],
                                             [0.7, 0.7, 0.7]], pbc=True),
          format="vasp", direct=True)
    org = _file_org(path)
    monkeypatch.setitem(G, "ORG_STRUC", org)
    _cf._TARGET_FP_CACHE.clear()
    with pytest.raises(RuntimeError, match="composition"):
        _cf._target_fingerprints(org)


# --- INPUT.txt parsing -------------------------------------------------------

def test_createorg_fingerprint_parses_target_options(monkeypatch, tmp_path):
    from gpevo.FunctionFolder.USPEX.src.createORG_Fingerprint import \
        createORG_Fingerprint
    inp = tmp_path / "INPUT.txt"
    inp.write_text(
        "10 : RmaxFing\n0.04 : deltaFing\n0.015 : sigmaFing\n"
        "data_exp/structures_mlp.traj,POSCAR_exp : targetStructure\n"
        "1.9 : targetBondCut\n"
    )
    org = struct(RmaxFing=10.0, deltaFing=0.08, sigmaFing=0.03, molecule=0,
                 varcomp=0, dimension=3, numIons=[2], atomType=[6], numMols=[1],
                 toleranceFing=0.008, toleranceBestHM=0.02,
                 antiSeedsMax=0.0, antiSeedsSigma=0.001,
                 antiSeedsActivation=5000, targetStructure=[],
                 targetBondCut=0.0)
    monkeypatch.setitem(G, "ORG_STRUC", org)
    createORG_Fingerprint(str(inp))
    assert list(org.targetStructure) == [
        "data_exp/structures_mlp.traj", "POSCAR_exp"]
    assert org.targetBondCut == 1.9


def test_startup_check_rejects_missing_target(tmp_path):
    from gpevo.FunctionFolder.USPEX.src.createORGStruc import check_optType_14
    with pytest.raises(RuntimeError, match="targetStructure"):
        check_optType_14(_org14(targetStructure=[]))
    with pytest.raises(RuntimeError, match="missing or unreadable"):
        check_optType_14(_org14(
            targetStructure=[str(tmp_path / "nope.cif")], homePath=str(tmp_path)))


def test_startup_check_accepts_existing_target(tmp_path):
    from gpevo.FunctionFolder.USPEX.src.createORGStruc import check_optType_14
    path = tmp_path / "POSCAR"
    _write_poscar(path)
    check_optType_14(_org14(targetStructure=[str(path)], homePath=str(tmp_path)))
    # other optTypes are not checked
    check_optType_14(_org14(optType=1, targetStructure=[]))


# --- different molecule counts (size-intensive fingerprint) -----------------

def test_composition_scale_accepts_uniform_multiples_only():
    assert _cf._composition_scale([2, 4], [1, 2]) == 2.0
    assert _cf._composition_scale([1, 2], [2, 4]) == 0.5
    assert _cf._composition_scale([3, 6], [3, 6]) == 1.0
    assert _cf._composition_scale([2, 3], [1, 2]) is None   # wrong RATIO
    assert _cf._composition_scale([3], [2]) is None         # 1.5x is not a supercell
    assert _cf._composition_scale([2, 4], [0, 2]) is None
    assert _cf._composition_scale([], [1, 2]) is None


def test_larger_target_is_accepted_with_a_warning(tmp_path, monkeypatch, capsys):
    from ase import Atoms
    from ase.io import write
    base = Atoms(symbols=["C", "C"], cell=np.eye(3) * 5.0,
                 scaled_positions=[[0.1, 0.1, 0.1], [0.6, 0.6, 0.6]],
                 pbc=True)
    path = tmp_path / "POSCAR_big"
    write(str(path), base * (2, 1, 1), format="vasp", direct=True)
    org = _file_org(path)                     # run expects [2], target has [4]
    monkeypatch.setitem(G, "ORG_STRUC", org)
    _cf._TARGET_FP_CACHE.clear()
    fps = _cf._target_fingerprints(org)
    assert len(fps) == 1 and fps[0].shape == (1, 10)
    assert "2x" in capsys.readouterr().out


def test_wrong_species_ratio_is_rejected(tmp_path, monkeypatch):
    from ase import Atoms
    from ase.io import write
    path = tmp_path / "POSCAR"
    write(str(path), Atoms(symbols=["C", "H", "H"], cell=np.eye(3) * 5.0,
                           scaled_positions=[[0.1, 0.1, 0.1],
                                             [0.4, 0.4, 0.4],
                                             [0.7, 0.7, 0.7]], pbc=True),
          format="vasp", direct=True)
    org = _org14(atomType=[6, 1], numIons=[1, 1], molecule=0,
                 RmaxFing=5.0, deltaFing=0.5, sigmaFing=0.1,
                 targetStructure=[str(path)])
    monkeypatch.setitem(G, "ORG_STRUC", org)
    _cf._TARGET_FP_CACHE.clear()
    with pytest.raises(RuntimeError, match="RATIO"):
        _cf._target_fingerprints(org)


# --- targetTolerance / hit logging ------------------------------------------

def test_hit_log_uses_default_tolerance(monkeypatch, capsys):
    _run(monkeypatch, _org14(), [_individual14(FP_A)], targets=[FP_A])
    out = capsys.readouterr().out
    assert "within targetTolerance=0.03" in out
    assert "1/1 structure(s)" in out


def test_hit_log_only_when_within_tolerance(monkeypatch, capsys):
    _run(monkeypatch, _org14(targetTolerance=1e-9),
         [_individual14(FP_B)], targets=[FP_A])
    assert "within targetTolerance" not in capsys.readouterr().out
    _run(monkeypatch, _org14(targetTolerance=2.0),
         [_individual14(FP_B)], targets=[FP_A])
    assert "within targetTolerance=2" in capsys.readouterr().out


def test_stop_fitness_message_has_no_energy_unit():
    from gpevo.FunctionFolder.USPEX.src.StopRun import StopRun
    stop, msg = StopRun(np.array([0.01]), 1, 50, 0, 0.03)
    assert stop == 1 and "eV" not in msg and "0.03" in msg


def test_createorg_fingerprint_parses_target_tolerance(monkeypatch, tmp_path):
    from gpevo.FunctionFolder.USPEX.src.createORG_Fingerprint import \
        createORG_Fingerprint
    inp = tmp_path / "INPUT.txt"
    inp.write_text("0.05 : targetTolerance\n")
    org = struct(RmaxFing=10.0, deltaFing=0.08, sigmaFing=0.03, molecule=0,
                 varcomp=0, dimension=3, numIons=[2], atomType=[6], numMols=[1],
                 toleranceFing=0.008, toleranceBestHM=0.02,
                 antiSeedsMax=0.0, antiSeedsSigma=0.001,
                 antiSeedsActivation=5000, targetStructure=[],
                 targetBondCut=0.0, targetTolerance=0.03)
    monkeypatch.setitem(G, "ORG_STRUC", org)
    createORG_Fingerprint(str(inp))
    assert org.targetTolerance == 0.05


def test_reversed_option_form_is_accepted(monkeypatch, tmp_path):
    from gpevo.FunctionFolder.USPEX.src.createORG_Fingerprint import \
        createORG_Fingerprint
    inp = tmp_path / "INPUT.txt"
    inp.write_text("targetStructure : POSCAR_exp\ntargetBondCut : 1.9\n")
    org = struct(RmaxFing=10.0, deltaFing=0.08, sigmaFing=0.03, molecule=0,
                 varcomp=0, dimension=3, numIons=[2], atomType=[6], numMols=[1],
                 toleranceFing=0.008, toleranceBestHM=0.02,
                 antiSeedsMax=0.0, antiSeedsSigma=0.001,
                 antiSeedsActivation=5000, targetStructure=[],
                 targetBondCut=0.0)
    monkeypatch.setitem(G, "ORG_STRUC", org)
    createORG_Fingerprint(str(inp))
    assert list(org.targetStructure) == ["POSCAR_exp"]
    assert org.targetBondCut == 1.9
