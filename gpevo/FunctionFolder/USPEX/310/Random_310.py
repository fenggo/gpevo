"""Python equivalent of Random_310.m (molecular random operator)."""
from __future__ import annotations

from importlib import import_module as _im

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.USPEX.src.variation_common import (
    finalize_offspring, recompute_zmatrix, valid_molecules,
)


def Random_310(count):
    _ = count
    ORG = G["ORG_STRUC"]
    rand_init = _im("gpevo.FunctionFolder.USPEX.310.Random_Init_310").Random_Init_310
    num_mols = list(ORG.numMols)
    for _ in range(5):
        good_pop, pop, nsym = rand_init(0, num_mols, 0)
        if pop is None:
            continue
        # Build an individual exported directly from the random generator.
        ind = AttrDict()
        ind.MOLECULES = pop.MOLECULES
        ind.numMols = list(map(int, pop.numMols))
        ind.MtypeLIST = list(map(int, pop.MtypeLIST))
        ind.typesAList = [int(t) for t in pop.typesAList]
        ind.numIons = [int(v) for v in pop.numIons]
        ind.LATTICE = pop.LATTICE
        ind.symg = int(nsym) if nsym not in (None, "", 0) else 1
        ind.howCome = "  Random  "
        if not valid_molecules(ind, 0):
            continue
        recompute_zmatrix(ind)
        final = finalize_offspring(ind, "  Random  ")
        G["OFF_STRUC"].POPULATION.append(final)
        return
