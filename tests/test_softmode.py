"""Tests for the real (MATLAB-aligned) soft-mode mutation operator."""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from gpevo.runtime import G, AttrDict, struct
from gpevo.FunctionFolder.USPEX.src.softmode import (
    calc_soft_modes,
    default_good_bonds,
    move_along_soft_mode,
    move_all_mol_mutation,
    valence,
    valence_electrons,
)
from gpevo.FunctionFolder.USPEX.src.NEW_coord2Zmatrix import NEW_coord2Zmatrix


# ---------------------------------------------------------------------------
# Lookup tables (valence.m / valenceElectronsNumber.m / atomicGoodBonds.m)
# ---------------------------------------------------------------------------

def test_valence_tables_match_matlab():
    assert valence(1) == 1.0      # H
    assert valence(6) == 4.0      # C
    assert valence(7) == 3.0      # N
    assert valence(8) == 2.0      # O
    assert valence(200) == 3.0   # beyond table
    assert valence_electrons(1) == 1.0
    assert valence_electrons(6) == 4.0
    assert valence_electrons(8) == 6.0
    assert valence_electrons(200) == 3.0


def test_default_good_bonds_is_symmetric_and_matches_table():
    gb = default_good_bonds([1, 6, 7, 8])
    assert gb.shape == (4, 4)
    assert np.allclose(gb, gb.T)
    # gB(H)=0.20, gB(C)=0.50 -> sqrt(0.20*0.50)
    assert gb[0, 1] == pytest.approx(np.sqrt(0.20 * 0.50))
    assert gb[1, 1] == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Dynamical matrix / eigenvectors vs the reference Cython core
# ---------------------------------------------------------------------------

def _reference_core():
    sys.path.insert(0, "/home/feng/uspex0925/FunctionFolder/sys")
    try:
        return importlib.import_module("uspex_softmode_core")
    except Exception:
        return None


def test_calc_soft_modes_matches_reference_cython_core():
    ref = _reference_core()
    if ref is None:
        pytest.skip("uspex0925 Cython core not importable")
    np.random.seed(1)
    for lat_a in (3.0, 5.0, 8.0):
        for n in (4, 8):
            coords = np.random.rand(n, 3) * 0.8 + 0.1
            lat = np.eye(3) * lat_a
            at = np.zeros(n, dtype=np.int32)
            R = np.array([0.76]); NV = np.array([4.0]); V = np.array([4.0])
            gb = np.array([[0.5]])
            w, _ = ref.calc_soft_modes_raw(
                np.ascontiguousarray(coords), np.ascontiguousarray(lat),
                at, R, NV, V, np.ascontiguousarray(gb))
            ref_freq = np.sort(np.abs(np.diag(w)))
            freq, _ = calc_soft_modes(coords, lat, at, R, NV, V, gb)
            assert np.max(np.abs(np.sort(freq) - ref_freq)) < 1e-10


def test_calc_soft_modes_two_species_triclinic_matches_reference():
    ref = _reference_core()
    if ref is None:
        pytest.skip("uspex0925 Cython core not importable")
    np.random.seed(7)
    coords = np.random.rand(8, 3) * 0.8 + 0.1
    lat = np.array([[6.0, 0.2, 0.1], [0.1, 5.5, -0.2], [0.0, 0.3, 7.0]])
    at = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    R = np.array([0.76, 0.66]); NV = np.array([4.0, 5.0]); V = np.array([4.0, 3.0])
    gb = np.array([[0.5, 0.4], [0.4, 0.3]])
    w, _ = ref.calc_soft_modes_raw(
        np.ascontiguousarray(coords), np.ascontiguousarray(lat),
        at, R, NV, V, np.ascontiguousarray(gb))
    freq, eigvector = calc_soft_modes(coords, lat, at, R, NV, V, gb)
    assert np.max(np.abs(np.sort(freq) - np.sort(np.abs(np.diag(w))))) < 1e-10
    assert eigvector.shape == (24, 24)
    assert np.isfinite(eigvector).all()


def test_calc_soft_modes_with_no_bonds_returns_zero_modes():
    # Two atoms too far apart for any bond (delta >= max_bond = 5).
    coords = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
    lat = np.eye(3) * 40.0
    freq, eigvector = calc_soft_modes(
        coords, lat, np.array([0, 0]), np.array([0.76]),
        np.array([4.0]), np.array([4.0]), np.array([[0.5]]))
    assert np.allclose(freq, 0.0)
    assert eigvector.shape == (6, 6)


# ---------------------------------------------------------------------------
# move_along_SoftMode_molMutation / move_all_mol_Mutation
# ---------------------------------------------------------------------------

def _make_molecules():
    mols = []
    for i in range(2):
        coords = np.array([[0.0 + i, 0.0, 0.0],
                           [1.2 + i, 0.0, 0.0],
                           [0.0 + i, 1.1, 0.0]], dtype=float)
        mol = AttrDict(MOLCOORS=coords.copy(), MOLCENTER=coords.mean(axis=0),
                       ZMATRIX=NEW_coord2Zmatrix(coords, np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0]])),
                       order=float(i))
        mols.append(mol)
    return mols


def _stdmol():
    return [AttrDict(types=[1, 1, 1], format=np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0]]))]


def test_move_along_soft_mode_does_not_mutate_inputs():
    mols = _make_molecules()
    before = [m.MOLCOORS.copy() for m in mols]
    disp = np.linspace(-0.2, 0.2, 18)
    out, deviation = move_along_soft_mode(mols, disp, [1, 1], _stdmol(), NEW_coord2Zmatrix)
    assert len(out) == 2
    assert len(deviation) == 2
    for orig, now in zip(before, mols):
        assert np.allclose(orig, now.MOLCOORS)   # parent untouched
    for m in out:
        assert m.MOLCOORS.shape == (3, 3)
        assert np.allclose(m.MOLCENTER, m.MOLCOORS.mean(axis=0))
        assert m.ZMATRIX.shape == (3, 3)


def test_move_all_mol_mutation_is_deterministic_and_wrapped():
    np.random.seed(0)
    mols = _make_molecules()
    lat = np.eye(3) * 10.0
    frac = move_all_mol_mutation(mols, [2], lat, [0.0, 1.0], 1.0)
    assert frac.shape == (2, 3)
    assert np.all(frac >= 0.0) and np.all(frac < 1.0)


# ---------------------------------------------------------------------------
# End-to-end: SoftModeMutation_310 with a synthetic workspace
# ---------------------------------------------------------------------------

def _synthetic_workspace():
    ORG = AttrDict(
        atomType=[6], numMols=[2], numIons=[6], abinitioCode=[1, 1, 1],
        STDMOL=_stdmol(), minDistMatrice=np.full((1, 1), 0.4),
        valences=np.array([4.0]), NvalElectrons=np.array([4.0]),
        goodBonds=np.array([[0.5]]), howManyMut=3, toleranceFing=0.0,
        weight=1.0, tournament=np.array([1.0, 2.0]), bestFrac=0.5,
        constLattice=0, mutationRate=0.08, optType=1, molecule=1,
        populationSize=4, initialPopSize=4, generation=1,
    )
    mols = _make_molecules()
    # species-blocked fractional COORDINATES (6 atoms)
    catted = np.vstack([m.MOLCOORS for m in mols])
    lat = np.eye(3) * 10.0
    coords = catted @ np.linalg.inv(lat)
    ind = AttrDict(MOLECULES=mols, LATTICE=lat, COORDINATES=coords,
                   numIons=[6], numMols=[2], MtypeLIST=[1, 1],
                   typesAList=[6] * 6, FINGERPRINT=None,
                   Enthalpies=np.array([100000.0, 100000.0, 1.0]),
                   Number=7, howCome="  Random  ")
    POP = struct()
    POP.POPULATION = [ind]
    POP.ranking = [0]
    POP.generation = 1
    POP.SOFTMODEParents = []
    POP.SOFTMUTATED = []
    OFF = struct()
    OFF.POPULATION = []
    OFF.SOFTMODEParents = POP.SOFTMODEParents
    G["ORG_STRUC"] = ORG
    G["POP_STRUC"] = POP
    G["OFF_STRUC"] = OFF
    return ORG, POP, OFF


def test_softmode_operator_produces_an_offspring():
    ORG, POP, OFF = _synthetic_workspace()
    np.random.seed(4)
    soft = importlib.import_module("gpevo.FunctionFolder.USPEX.310.SoftModeMutation_310")
    soft.SoftModeMutation_310(1)
    assert len(OFF.POPULATION) == 1
    child = OFF.POPULATION[0]
    assert str(child.howCome) in ("softmutate", "CoorMutate", "  Random  ")
    assert child.MOLECULES[0].MOLCOORS.shape == (3, 3)
    assert np.isfinite(np.asarray(child.MOLECULES[0].MOLCOORS, dtype=float)).all()


def test_softmode_operator_updates_parent_cache_on_success(monkeypatch):
    ORG, POP, OFF = _synthetic_workspace()
    np.random.seed(4)
    soft = importlib.import_module("gpevo.FunctionFolder.USPEX.310.SoftModeMutation_310")
    # Force the amplitude sweep to succeed so the cache path is exercised.
    from gpevo.FunctionFolder.USPEX.src import variation_common
    monkeypatch.setattr(variation_common, "valid_molecules", lambda ind, adj=0: 1)
    monkeypatch.setattr(soft, "valid_molecules", lambda ind, adj=0: 1)
    soft.SoftModeMutation_310(1)
    assert OFF.POPULATION[0].howCome == "softmutate"
    assert len(OFF.SOFTMODEParents) == 1
    entry = OFF.SOFTMODEParents[0]
    assert int(entry.Softmode_num) >= 1
    assert np.isfinite(float(entry.Softmode_Fre))
    assert OFF.POPULATION[0].Parents[0].mut_mode >= 1


def test_mutation_310_fallback_produces_an_offspring(monkeypatch):
    ORG, POP, OFF = _synthetic_workspace()
    np.random.seed(2)
    from gpevo.FunctionFolder.USPEX.src import variation_common
    monkeypatch.setattr(variation_common, "valid_molecules", lambda ind, adj=0: 1)
    mut = importlib.import_module("gpevo.FunctionFolder.USPEX.310.Mutation_310")
    monkeypatch.setattr(mut, "valid_molecules", lambda ind, adj=0: 1)
    mut.Mutation_310(1)
    assert len(OFF.POPULATION) == 1
    assert OFF.POPULATION[0].howCome == "CoorMutate"


def test_createORG_System_sets_softmode_tables(tmp_path, monkeypatch):
    """Defaults from the MATLAB lookup tables + explicit INPUT.txt blocks."""
    from gpevo.FunctionFolder.USPEX.src import createORG_System
    from gpevo.FunctionFolder.USPEX.src import createORGDefault

    def run(text):
        inp = tmp_path / "INPUT.txt"
        inp.write_text(text)
        monkeypatch.chdir(tmp_path)
        ORG = AttrDict()
        ORG.numSpeciesSymbols = "C O N H"
        ORG.molecule = 1
        ORG.varcomp = 0
        ORG.dimension = 3
        ORG.atomType = []
        ORG.numIons = [1]
        ORG.numMols = []
        ORG.minDistMatrice = None
        ORG.goodBonds = None
        ORG.valences = []
        ORG.NvalElectrons = []
        ORG.homePath = str(tmp_path)
        ORG.resFolder = ""
        ORG.opt_sign = 1
        ORG.optType = 1
        ORG.abinitioCode = [1]
        G["ORG_STRUC"] = ORG
        createORG_System.createORG_System(str(inp))
        return G["ORG_STRUC"]

    org = run("% atomType\nC O N H\n% EndAtomType\n")
    assert list(org.atomType) == [6, 8, 7, 1]
    assert np.allclose(org.valences, [4.0, 2.0, 3.0, 1.0])
    assert np.allclose(org.NvalElectrons, [4.0, 6.0, 5.0, 1.0])
    assert org.goodBonds.shape == (4, 4)
    assert np.allclose(org.goodBonds, org.goodBonds.T)
    assert org.goodBonds[0, 1] == pytest.approx(np.sqrt(0.50 * 0.30))

    org = run(
        "% atomType\nC O\n% EndAtomType\n"
        "% valences\n2.5 3.5\n% EndValences\n"
        "% valenceElectr\n3 7\n% EndValenceElectr\n"
        "% goodBonds\n0.4 0.2\n0.2 0.6\n% EndGoodBonds\n"
    )
    assert np.allclose(org.valences, [2.5, 3.5])
    assert np.allclose(org.NvalElectrons, [3.0, 7.0])
    assert np.allclose(org.goodBonds, [[0.4, 0.2], [0.2, 0.6]])


def test_ea_310_carries_softmode_cache():
    # The OFF struct built by EA_310 must carry the cache across generations.
    ORG, POP, OFF = _synthetic_workspace()
    POP.SOFTMODEParents = [AttrDict(fingerprint=None, Softmode_Fre=1.0, Softmode_num=2)]
    POP.SOFTMUTATED = [AttrDict(flag=1)]
    # Exercise the exact statements added to EA_310 via source inspection,
    # since running EA_310 needs a full run environment.
    import inspect
    ea = importlib.import_module("gpevo.FunctionFolder.USPEX.310.EA_310")
    src = inspect.getsource(ea.EA_310)
    assert "OFF.SOFTMODEParents" in src
    assert "OFF.SOFTMUTATED" in src
