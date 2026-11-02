"""The five 310 variation operators must emit a fresh ``mol.ZMATRIX``.

MATLAB recomputes the Z-matrix of every molecule on a successful offspring
(LatMutation_310.m:42, Mutation_310.m:53, Permutation_310.m:86,
Rotation_310.m:140/207).  gpevo used to leave the parent's ZMATRIX on the
child; that became a real hazard once ``move_all_mol_Mutation`` started reading
``ZMATRIX(1,:)`` as the molecule centre.

These tests give each parent an intentionally stale (all-zero) ZMATRIX and
check that every operator replaces it with NEW_coord2Zmatrix(MOLCOORS, format).
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from gpevo.runtime import G, AttrDict, struct
from gpevo.FunctionFolder.USPEX.src.NEW_coord2Zmatrix import NEW_coord2Zmatrix

FORMAT = np.array([[1, 1, 1], [1, 1, 1], [1, 2, 3]], dtype=int)


def _molecule(offset=0.0, stale=True):
    coords = np.array([[0.0 + offset, 0.0, 0.0],
                       [1.2 + offset, 0.0, 0.0],
                       [0.0 + offset, 1.1, 0.0]], dtype=float)
    return AttrDict(
        MOLCOORS=coords,
        MOLCENTER=coords.mean(axis=0),
        # Deliberately stale: nothing in the pipeline produces this value.
        ZMATRIX=np.zeros((3, 3), dtype=float) if stale
        else NEW_coord2Zmatrix(coords, FORMAT),
        order=0.0,
    )


def _workspace():
    lat = np.eye(3) * 10.0
    mols = [_molecule(0.0), _molecule(3.0)]
    catted = np.vstack([m.MOLCOORS for m in mols])
    coords = catted @ np.linalg.inv(lat)
    ORG = AttrDict(
        atomType=[6], numMols=[2], numIons=[6], abinitioCode=[1, 1, 1],
        STDMOL=[AttrDict(types=[1, 1, 1], format=FORMAT)],
        minDistMatrice=np.full((1, 1), 0.4), tournament=np.array([1.0, 2.0]),
        bestFrac=0.5, populationSize=4, initialPopSize=4,
        howManyMut=3, mutationRate=0.08, rotationAngleMax=np.pi / 2,
        translationMax=0.5, maxDihedralIter=200, molRefBonds=None,
        howManySwaps=1, swarmStep=0.5, swarmAngle=30.0,
        constLattice=0, optType=1, molecule=1, generation=1,
    )
    ind = AttrDict(
        MOLECULES=mols, LATTICE=lat, COORDINATES=coords,
        numIons=[6], numMols=[2], MtypeLIST=[1, 1], typesAList=[6] * 6,
        FINGERPRINT=None, Enthalpies=np.array([1.0, 1.0, 1.0]),
        Number=7, howCome="  Random  ",
    )
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


def _assert_fresh(child):
    assert len(child.MOLECULES) == 2
    for mol in child.MOLECULES:
        expected = np.real(NEW_coord2Zmatrix(
            np.asarray(mol.MOLCOORS, dtype=float), FORMAT))
        got = np.asarray(mol.ZMATRIX, dtype=float)
        assert got.shape == expected.shape
        assert not np.allclose(got, 0.0), "ZMATRIX stayed stale (all zeros)"
        assert np.allclose(got, expected), "ZMATRIX does not match MOLCOORS"


OPERATORS = [
    ("Rotation_310", "Rotation_310"),
    ("Permutation_310", "Permutation_310"),
    ("LatMutation_310", "LatMutation_310"),
    ("SwarmMutation_310", "SwarmMutation_310"),
]


@pytest.mark.parametrize("mod_name,func_name", OPERATORS)
def test_operator_emits_fresh_zmatrix(mod_name, func_name, monkeypatch):
    ORG, POP, OFF = _workspace()
    np.random.seed(5)
    mod = importlib.import_module("gpevo.FunctionFolder.USPEX.310." + mod_name)
    # Bypass the geometry checks so the success branch always runs; we are
    # testing the ZMATRIX bookkeeping, not the distance gates.
    monkeypatch.setattr(mod, "valid_molecules", lambda ind, adj=0: 1)
    if hasattr(mod, "latticeCheck"):
        monkeypatch.setattr(mod, "latticeCheck", lambda lat: 1)
    getattr(mod, func_name)(1)
    assert len(OFF.POPULATION) == 1, mod_name + " produced no offspring"
    # Each operator must have taken its OWN success branch, not fallen back to
    # Random_310 (which would hide a missing recompute_zmatrix call).
    assert str(OFF.POPULATION[0].howCome).strip() not in ("", "Random"), (
        mod_name + " fell back to Random")
    _assert_fresh(OFF.POPULATION[0])


def test_random_operator_emits_fresh_zmatrix(monkeypatch):
    ORG, POP, OFF = _workspace()
    np.random.seed(5)
    ri = importlib.import_module("gpevo.FunctionFolder.USPEX.310.Random_Init_310")

    def fake_init(good_pop, num_mols, nsym=0):
        pop = AttrDict(
            MOLECULES=[_molecule(0.0), _molecule(3.0)],
            numMols=[2], MtypeLIST=[1, 1], typesAList=[6] * 6,
            numIons=[6], LATTICE=np.eye(3) * 10.0, howCome="  Random  ",
        )
        return good_pop + 1, pop, 1

    monkeypatch.setattr(ri, "Random_Init_310", fake_init)
    rnd = importlib.import_module("gpevo.FunctionFolder.USPEX.310.Random_310")
    monkeypatch.setattr(rnd, "valid_molecules", lambda ind, adj=0: 1)
    rnd.Random_310(1)
    assert len(OFF.POPULATION) == 1
    _assert_fresh(OFF.POPULATION[0])


def test_recompute_zmatrix_helper_fallback_without_template():
    from gpevo.FunctionFolder.USPEX.src.variation_common import recompute_zmatrix
    ORG, POP, OFF = _workspace()
    ORG.STDMOL = []
    ind = AttrDict(MOLECULES=[_molecule(0.0)], MtypeLIST=[1])
    recompute_zmatrix(ind)
    got = np.asarray(ind.MOLECULES[0].ZMATRIX, dtype=float)
    assert np.allclose(got, np.asarray(ind.MOLECULES[0].MOLCOORS, dtype=float))


def test_recompute_zmatrix_helper_is_a_no_op_for_empty_molecules():
    from gpevo.FunctionFolder.USPEX.src.variation_common import recompute_zmatrix
    _workspace()
    ind = AttrDict(MOLECULES=[], MtypeLIST=[])
    assert recompute_zmatrix(ind) is ind
