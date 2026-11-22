"""Rejected structures (damaged molecules / fake low-energy) go to a traj.

A single ``resFolder/rejected.traj`` collects structures the EA throws away,
for later force-field training.  Frames carry geometry and provenance only --
the 100000 enthalpy sentinel is a failure marker, not a training label, so no
single-point energy is attached.  Both rejection paths (readMOL molecule check
and the GP fake-residual screen) funnel through the same helper, which writes
at most one frame per structure (first reason wins).
"""
from __future__ import annotations

import numpy as np
import pytest

from gpevo.runtime import G, AttrDict, struct
from gpevo.FunctionFolder.USPEX.src.WriteIndividualsTraj import (
    append_rejected_traj, _REJECTED_FLAG,
)
from gpevo.FunctionFolder.USPEX.src.residual_check import mark_fake_residual
from gpevo.FunctionFolder.AbinitCode.readMOL import _check_molecules
from gpevo.FunctionFolder.sys.mol_bond_table import build_reference_bond_tables


def _setup(tmp_path, atom_type=(6, 1)):
    """Point G at a tmp resFolder with a CH2-like single-molecule setup."""
    ORG = AttrDict()
    ORG.atomType = list(atom_type)
    ORG.homePath = str(tmp_path)
    ORG.resFolder = "results1"
    ORG.checkMolecules = 1
    ORG.minDistMatrice = np.full((len(atom_type), len(atom_type)), 0.1)
    std = AttrDict()
    std.molecule = np.array([[0.0, 0.0, 0.0],
                             [1.09, 0.0, 0.0],
                             [-1.09, 0.0, 0.0]], dtype=float)
    std.types = [1, 2, 2]
    std.ZMATRIX = np.zeros((3, 1))
    ORG.STDMOL = [std]
    POP = AttrDict()
    POP.resFolder = "results1"
    POP.generation = 7
    G["ORG_STRUC"] = ORG
    G["POP_STRUC"] = POP
    build_reference_bond_tables()
    return ORG, POP


def _structure(residual=12657.0):
    """A rejectable individual with a valid 3-atom geometry."""
    s = struct(
        Enthalpies=[-100.0], Step=[4], Error=0, residual=residual,
        distance=residual, Number=42,
        numIons=[1, 2],
        LATTICE=(20.0 * np.eye(3)).tolist(),
        COORDINATES=([[0.0, 0.0, 0.0],
                      [1.09, 0.0, 0.0],
                      [-1.09, 0.0, 0.0]]),
    )
    return s


def _frames(path):
    from ase.io import read

    return list(read(str(path), index=":"))


# --------------------------------------------------------------------------
# helper
# --------------------------------------------------------------------------

def test_helper_writes_geometry_and_provenance_only(tmp_path):
    _setup(tmp_path)
    s = _structure()
    s.moleculeCheck = "FAKE low-energy (excessive GP residual)"
    assert append_rejected_traj(s, "fake_residual", generation=3) is True

    path = tmp_path / "results1" / "rejected.traj"
    assert path.is_file()
    frames = _frames(path)
    assert len(frames) == 1
    atoms = frames[0]
    assert atoms.get_chemical_symbols() == ["C", "H", "H"]
    assert atoms.info["reason"] == "fake_residual"
    assert atoms.info["generation"] == 3
    assert atoms.info["Number"] == 42
    assert atoms.info["residual"] == pytest.approx(12657.0)
    assert "FAKE low-energy" in atoms.info["moleculeCheck"]
    # No physical energy: the 100000 sentinel must not become a label.
    assert atoms.calc is None
    assert getattr(s, _REJECTED_FLAG) is True


def test_helper_is_idempotent_first_reason_wins(tmp_path):
    _setup(tmp_path)
    s = _structure()
    assert append_rejected_traj(s, "molecule_damaged") is True
    assert append_rejected_traj(s, "fake_residual") is False
    frames = _frames(tmp_path / "results1" / "rejected.traj")
    assert len(frames) == 1
    assert frames[0].info["reason"] == "molecule_damaged"


def test_helper_appends_distinct_structures(tmp_path):
    _setup(tmp_path)
    s1, s2 = _structure(), _structure()
    s2.Number = 43
    assert append_rejected_traj(s1, "fake_residual") is True
    assert append_rejected_traj(s2, "molecule_damaged") is True
    assert len(_frames(tmp_path / "results1" / "rejected.traj")) == 2


def test_helper_no_org_is_noop(tmp_path):
    G.pop("ORG_STRUC", None)
    s = _structure()
    assert append_rejected_traj(s, "fake_residual") is False


# --------------------------------------------------------------------------
# integration: fake residual path
# --------------------------------------------------------------------------

def test_fake_residual_rejection_writes_frame(tmp_path):
    _setup(tmp_path)
    s = _structure(residual=12657.0)
    assert mark_fake_residual(s, label="id=42") is True
    frames = _frames(tmp_path / "results1" / "rejected.traj")
    assert len(frames) == 1
    assert frames[0].info["reason"] == "fake_residual"
    # Re-marking is a no-op: no second frame.
    assert mark_fake_residual(s) is False
    assert len(_frames(tmp_path / "results1" / "rejected.traj")) == 1


def test_normal_structure_writes_nothing(tmp_path):
    _setup(tmp_path)
    s = _structure(residual=0.5)
    assert mark_fake_residual(s) is False
    assert not (tmp_path / "results1" / "rejected.traj").exists()


# --------------------------------------------------------------------------
# integration: molecule damage path
# --------------------------------------------------------------------------

def test_damaged_molecule_writes_frame(tmp_path):
    ORG, _ = _setup(tmp_path)
    s = _structure()
    # Rebuild MOLECULES from a damaged geometry (two H pulled together so an
    # intramolecular H-H bond appears that the template does not have).
    damaged = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0]])
    s.MtypeLIST = [1]
    mol = AttrDict()
    mol.MOLCOORS = damaged
    mol.ZMATRIX = np.zeros((3, 1))
    s.MOLECULES = [mol]
    _check_molecules(s, label="42")
    assert "bond increase" in s.moleculeCheck
    frames = _frames(tmp_path / "results1" / "rejected.traj")
    assert len(frames) == 1
    assert frames[0].info["reason"] == "molecule_damaged"
