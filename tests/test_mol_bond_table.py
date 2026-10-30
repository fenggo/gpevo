"""Tests for the intramolecular bond-increase molecule-damage check.

A molecule whose relaxed geometry has gained an intramolecular bond not present
in the standard (template) molecule is flagged as damaged, supplementing the
existing Z-matrix / intermolecular-distance checks in ``readMOL._check_molecules``.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.sys.mol_bond_table import (
    intramol_bonds, build_reference_bond_tables,
)
from gpevo.FunctionFolder.AbinitCode.readMOL import _check_molecules


# CH2-like template: C at origin, two H at +/-1.09 A.  H-H distance (2.18 A) is
# far above the bond cutoff, so the reference bond set is just the two C-H bonds.
_CH2_MOL = AttrDict(
    molecule=np.array([[0.0, 0.0, 0.0],
                       [1.09, 0.0, 0.0],
                       [-1.09, 0.0, 0.0]], dtype=float),
    types=[1, 2, 2],  # 1=C(atomic 6), 2=H(atomic 1)
)


def _make_org(mol_coords):
    """Build a minimal ORG whose single molecule type is CH2-like at mol_coords."""
    ORG = AttrDict()
    ORG.atomType = [6, 1]
    ORG.checkMolecules = 1
    ORG.minDistMatrice = np.full((2, 2), 0.1)  # tiny -> never trips newMolCheck
    std = AttrDict()
    std.molecule = _CH2_MOL.molecule.copy()
    std.types = list(_CH2_MOL.types)
    std.ZMATRIX = np.zeros((3, 1))
    ORG.STDMOL = [std]
    G["ORG_STRUC"] = ORG
    build_reference_bond_tables()
    return ORG


def _make_struct(mol_coords):
    s = AttrDict()
    s.MtypeLIST = [1]
    # Large cell so the periodic-image side of newMolCheck does not wrap a
    # nearby atom onto its own image (that check is for molecules near a boundary).
    s.LATTICE = (50.0 * np.eye(3)).tolist()
    s.Enthalpies = [-100.0]
    s.Step = [2]
    s.Error = 0
    mol = AttrDict()
    mol.MOLCOORS = np.asarray(mol_coords, dtype=float)
    mol.ZMATRIX = np.zeros((3, 1))
    s.MOLECULES = [mol]
    return s


def test_reference_bond_table_has_two_ch_bonds():
    _make_org(_CH2_MOL.molecule)
    ref = intramol_bonds(_CH2_MOL.molecule, _CH2_MOL.types)
    assert ref == {(0, 1), (0, 2)}


def test_build_reference_populates_org():
    ORG = _make_org(_CH2_MOL.molecule)
    assert isinstance(ORG.molRefBonds, list) and len(ORG.molRefBonds) == 1
    assert ORG.molRefBonds[0] == {(0, 1), (0, 2)}


def test_undamaged_molecule_is_not_flagged():
    _make_org(_CH2_MOL.molecule)
    s = _make_struct(_CH2_MOL.molecule)  # identical to template -> no extra bond
    _check_molecules(s, label="1")
    assert getattr(s, "moleculeCheck", None) is None
    assert s.Enthalpies[-1] == pytest.approx(-100.0)


def test_bond_increase_flags_molecule_damaged():
    _make_org(_CH2_MOL.molecule)
    # Two H pulled to 0.1 A apart -> an H-H bond appears that the template lacks.
    damaged = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0]]
    s = _make_struct(damaged)
    _check_molecules(s, label="2")
    assert "bond increase" in s.moleculeCheck
    assert s.Enthalpies[-1] == pytest.approx(100000.0)
    assert s.Error == 4  # second-step relaxation -> counter +4


def test_bond_increase_check_is_gated_by_checkMolecules():
    ORG = _make_org(_CH2_MOL.molecule)
    ORG.checkMolecules = 0
    damaged = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0]]
    s = _make_struct(damaged)
    _check_molecules(s, label="3")
    assert getattr(s, "moleculeCheck", None) is None
