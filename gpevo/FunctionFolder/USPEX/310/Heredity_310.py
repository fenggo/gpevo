"""Python port of Heredity_310.m (slice heredity on molecule centres)."""
from __future__ import annotations

from importlib import import_module as _im

import numpy as np

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.sys.latticeCheck import latticeCheck
from gpevo.FunctionFolder.sys.optLattice import optLattice
from gpevo.FunctionFolder.USPEX.src.GetPOP_MOL import GetPOP_MOL
from gpevo.FunctionFolder.USPEX.src.NEW_coord2Zmatrix import NEW_coord2Zmatrix
from gpevo.FunctionFolder.USPEX.src.variation_common import pick_parent, valid_molecules
from gpevo.FunctionFolder.AbinitCode.readMOL import _movecoor
from gpevo.log_utils import console_print


heredity_molecule = _im(
    "gpevo.FunctionFolder.USPEX.310.heredity_molecule"
).heredity_molecule


def Heredity_310(Ind_No):
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    OFF = G["OFF_STRUC"]

    for attempt in range(50):
        if len(POP.POPULATION) < 2:
            break
        # Tournament selection, Heredity_310.m:22-32.
        ind1 = pick_parent()
        ind2 = pick_parent()
        if ind1 is None or ind2 is None or ind1 == ind2:
            continue

        good_heritage = 0
        good_lattice = 0
        for _security in range(20):
            if np.linalg.norm(
                np.asarray(POP.POPULATION[ind1].numMols, dtype=float)
                - np.asarray(POP.POPULATION[ind2].numMols, dtype=float)
            ) > 1e-3:
                break
            num_mols, offspring, lattice, frac_frac, dimension, offset, table = heredity_molecule(ind1, ind2)
            num_blocks = np.asarray(num_mols, dtype=float) / np.maximum(
                np.asarray(ORG.numMols, dtype=float), 1e-12
            )
            if np.any(num_blocks == 0):
                good_lattice = 0
            elif int(getattr(ORG, "constLattice", 0)):
                good_lattice = 1
            else:
                good_lattice = latticeCheck(lattice)
                if good_lattice:
                    coords, lattice = optLattice(offspring @ lattice, lattice)
                    try:
                        offspring = coords @ np.linalg.inv(lattice)
                    except np.linalg.LinAlgError:
                        good_lattice = 0

            if not good_lattice:
                continue

            molecules = []
            for row, (parent_id, mol_id, type_ind) in zip(offspring, table):
                source = POP.POPULATION[ind1 if parent_id == 1 else ind2]
                molecule = source.MOLECULES[int(mol_id) - 1]
                std = ORG.STDMOL[int(type_ind) - 1]
                # Checkpoints written before readMOL reassembled molecules can
                # still hold boundary-split molecules; fold them back first.
                mol = _movecoor(np.asarray(molecule.MOLCOORS, dtype=float),
                                np.asarray(source.LATTICE, dtype=float), std.format)
                center = np.mean(mol, axis=0)
                displacement = np.asarray(row, dtype=float) @ lattice - center
                entry = AttrDict()
                entry.MOLCOORS = mol + displacement
                entry.ZMATRIX = NEW_coord2Zmatrix(entry.MOLCOORS, std.format) if mol.shape[0] > 1 else mol.copy()
                entry.MOLCENTER = np.mean(entry.MOLCOORS, axis=0)
                entry.ID = len(molecules) + 1
                entry.order = 0
                molecules.append(entry)

            types_alist, mtype_list, num_ions = GetPOP_MOL(num_mols)
            probe = AttrDict()
            probe.MOLECULES = molecules
            probe.LATTICE = lattice
            probe.MtypeLIST = mtype_list
            good_heritage = valid_molecules(probe, -0.2)

            if good_heritage:
                offspring_ind = AttrDict()
                offspring_ind.MOLECULES = molecules
                offspring_ind.LATTICE = np.asarray(lattice, dtype=float)
                offspring_ind.numIons = list(num_ions)
                offspring_ind.numMols = list(num_mols)
                offspring_ind.typesAList = list(types_alist)
                offspring_ind.MtypeLIST = list(mtype_list)
                offspring_ind.howCome = " Heredity "
                offspring_ind.Parents = [AttrDict(parent="", enthalpy=0.0)]
                offspring_ind.Number = 0
                offspring_ind.Done = 0
                offspring_ind.ToDo = 1
                offspring_ind.JobID = 0
                offspring_ind.Folder = 0
                offspring_ind.Step = 1
                offspring_ind.Enthalpies = np.array([100000.0] * len(ORG.abinitioCode))
                offspring_ind.K_POINTS = np.ones((len(ORG.abinitioCode), 3))
                offspring_ind.symg = 1

                enthalpy = (
                    float(np.asarray(POP.POPULATION[ind1].Enthalpies)[-1]) * frac_frac
                    + float(np.asarray(POP.POPULATION[ind2].Enthalpies)[-1]) * (1.0 - frac_frac)
                )
                offspring_ind.Parents = [
                    AttrDict(
                        parent=f"{int(POP.POPULATION[ind1].Number)} {int(POP.POPULATION[ind2].Number)}",
                        enthalpy=enthalpy / max(1, int(sum(num_ions))),
                    )
                ]
                offspring_ind.fracFrac = [0.0, frac_frac, 1.0]
                offspring_ind.dimension = dimension
                offspring_ind.offset = list(np.asarray(offset, dtype=float).reshape(-1))

                # The Python port appends offspring instead of writing into a
                # fixed slot (that is what the other 310 operators do).
                OFF.POPULATION.append(offspring_ind)
                G["OFF_STRUC"] = OFF
                console_print(f"Structure {Ind_No}  generated by heredity")
                return offspring_ind

    from importlib import import_module

    import_module("gpevo.FunctionFolder.USPEX.310.Random_310").Random_310(Ind_No)
    return None
