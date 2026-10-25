"""Python equivalent of Random_Init_310.m."""
from __future__ import annotations

import numpy as np
import os

from gpevo.runtime import G, AttrDict
from importlib import import_module as _im

symope_310 = _im("gpevo.FunctionFolder.USPEX.310.symope_310").symope_310
from gpevo.FunctionFolder.USPEX.src.GetPOP_MOL import GetPOP_MOL
from gpevo.FunctionFolder.USPEX.src.GetOrientation import GetOrientation
from gpevo.FunctionFolder.USPEX.src.newMolCheck import newMolCheck
from gpevo.FunctionFolder.sys.distanceCheck import distanceCheck
from gpevo.FunctionFolder.sys.spaceGroups import spaceGroups
from gpevo.log_utils import console_print
from gpevo.matrix import det


def Random_Init_310(good_pop, num_mols0, nsym=0):
    ORG = G["ORG_STRUC"]
    center_min = ORG.CenterminDistMatrice
    num_mols = list(num_mols0)
    bad_sym = 0
    new_sym = 1
    split = np.asarray(ORG.splitInto or [], dtype=float)

    attempts = 0
    while True:
        attempts += 1
        if attempts > 200:
            # Not even P1 could be built (bad molecule sizes / distances):
            # give up instead of spinning forever.
            return good_pop, None, nsym
        if bad_sym > 5:
            bad_sym = 0
            new_sym = 1
        bad_sym += 1

        if new_sym:
            active = [i for i, v in enumerate(ORG.nsym) if i > 0 and v > 0]
            if not active:
                # Every requested group failed (Random_Init_310.m has no guard
                # for this and dies on ``tmp(0)`` in Octave).  Restore P1.
                active = [1]
                ORG.nsym[1] = 1
                G["ORG_STRUC"] = ORG
            nsym = int(np.random.choice(active))
            new_sym = 0

        if ORG.constLattice:
            lat1 = np.asarray(ORG.lattice, dtype=float)
        else:
            lat1 = np.asarray([ORG.latVolume * sum(num_mols) / max(1, sum(ORG.numMols))], dtype=float)

        split_into = split
        if nsym > 1 and np.sum(split_into) > 3:
            multiplicities = [int(v) if float(v).is_integer() else v for v in split_into]
            num_mols = np.array(num_mols0, dtype=float) / float(np.prod(split_into))
            if lat1.size == 1:
                lat1 = np.array([lat1[0] / split_into[0], lat1[0] / split_into[1], lat1[0] / split_into[2]])
            elif lat1.size == 9:
                lat1 = lat1.reshape(3, 3).copy()
                for i in range(3):
                    lat1[i, :] = lat1[i, :] / split_into[i]

        # Symmetry-aware random site generation
        prev_cwd = os.getcwd()
        try:
            os.chdir("CalcFoldTemp")
            candidate, lat, num_sites, operation, err_s, err_n, P, PB = symope_310(
                nsym, num_mols, lat1, center_min
            )
        finally:
            os.chdir(prev_cwd)

        if err_n == 1:
            # Random_Init_310.m:51 -- drop the group that could not be built.
            # P1 is kept as the last resort: removing it empties the pool, and
            # with an empty pool the Octave port dies on ``tmp(0)``.
            if 1 < nsym < len(ORG.nsym):
                ORG.nsym[nsym] = 0
                G["ORG_STRUC"] = ORG
            new_sym = 1
            continue
        if err_s == 0:
            pop = AttrDict()
            pop.LATTICE = np.asarray(lat)
            if lat.size == 9:
                pop.LATTICE = lat.reshape(3, 3)
            goodbad = False
            if distanceCheck(candidate, pop.LATTICE, np.array(num_mols, dtype=int),
                             center_min - 0.2):
                goodbad = 1
            if goodbad:
                atom_types, mtype_list, num_ions = GetPOP_MOL(num_mols)
                for _ in range(20):
                    site_counts = np.asarray(num_sites)[:, 0] if np.asarray(num_sites).ndim > 1 else np.asarray(num_sites).ravel()
                    molecules = GetOrientation(candidate, pop.LATTICE, site_counts,
                                               operation, mtype_list, nsym, P, PB)
                    if newMolCheck(molecules, pop.LATTICE, mtype_list, ORG.minDistMatrice):
                        atom_types, mtype_list, num_ions = GetPOP_MOL(num_mols)
                        pop.MOLECULES = molecules
                        pop.numMols = list(map(int, num_mols))
                        pop.MtypeLIST = mtype_list
                        pop.typesAList = atom_types
                        pop.numIons = num_ions
                        pop.LATTICE = pop.LATTICE
                        pop.howCome = "  Random  "
                        # Random_Init_310.m:84 -- disp(), terminal only.
                        console_print(
                            f"Molecular Crystal {good_pop + 1} built with the "
                            f"symmetry group {nsym} ({spaceGroups(nsym)})"
                        )
                        return good_pop + 1, pop, nsym
    return good_pop, None, nsym
