"""Python equivalent of initialize_POP_STRUC_310.m."""
from __future__ import annotations

import os
from importlib import import_module as _im

import numpy as np

from gpevo.runtime import G, AttrDict, struct
from gpevo.FunctionFolder.USPEX.src.pick_Seeds import pick_Seeds
from gpevo.log_utils import console_print


def _rand_init_mod():
    return _im("gpevo.FunctionFolder.USPEX.310.Random_Init_310").Random_Init_310


def _start_pop_mod():
    return _im("gpevo.FunctionFolder.USPEX.310.Start_POP_310").Start_POP_310


def createCompostion_SingleBlock():
    ORG = G["ORG_STRUC"]
    if not ORG.minAt:
        ORG.minAt = sum(map(int, ORG.numIons))
    if not ORG.maxAt:
        ORG.maxAt = sum(map(int, ORG.numIons))
    # One composition block for non-variable-composition molecular crystal
    ORG.firstGeneSplit = [list(map(int, ORG.numMols))]
    ORG.firstGeneSplitAll = [list(map(int, ORG.numMols))]
    file_path = os.path.join(ORG.homePath, "Seeds", "compositions")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as fp:
        for row in ORG.firstGeneSplit:
            fp.write(" ".join(f"{v:4d}" for v in row) + "\n")


def initialize_POP_STRUC_310():
    ORG = G["ORG_STRUC"]
    POP = struct()
    POP.POPULATION = []
    POP.SOFTMODEParents = []
    POP.SOFTMUTATED = []
    POP.resFolder = ORG.resFolder
    POP.generation = 1
    POP.bodyCount = 0
    POP.bad_rank = 0
    POP.ranking = []
    POP.DoneOrder = []
    POP.fitness = []

    G["POP_STRUC"] = POP
    createCompostion_SingleBlock()

    createCompostion_SingleBlock()  # idempotent
    success_pop = 0
    good_pop = 0
    nsym = 0
    n_split = len(ORG.firstGeneSplit)
    if n_split == 0:
        raise RuntimeError("ERROR: minAt/maxAt is too small, no composition splits were generated")

    total_mol_blocks = ORG.numMols
    # In a single-block 310 run, numMols for an individual equals numMols in
    # ORG_STRUC; no random composition split is applied yet.
    while success_pop < getattr(ORG, "initialPopSize", ORG.populationSize):
        num_mols = list(map(int, total_mol_blocks))
        good_pop, new_pop, nsym = _rand_init_mod()(good_pop, num_mols, nsym)
        if new_pop is None:
            console_print("  Random_Init_310 could not generate a valid structure, retrying sym group")
            continue
        pop_entry = AttrDict()
        pop_entry.MOLECULES = new_pop.MOLECULES
        pop_entry.numMols = list(map(int, new_pop.numMols))
        pop_entry.MtypeLIST = list(map(int, new_pop.MtypeLIST))
        pop_entry.typesAList = [int(t) for t in new_pop.typesAList]
        pop_entry.numIons = [int(v) for v in new_pop.numIons]
        pop_entry.LATTICE = np.asarray(new_pop.LATTICE, dtype=float).reshape(3, 3)
        pop_entry.symg = int(nsym) if nsym not in (None, "", 0) else 1
        pop_entry.howCome = "  Random  "
        POP.POPULATION.append(pop_entry)
        success_pop += 1

    G["POP_STRUC"] = POP
    pick_Seeds()
    _start_pop_mod()()
    G["POP_STRUC"] = POP
    return POP
