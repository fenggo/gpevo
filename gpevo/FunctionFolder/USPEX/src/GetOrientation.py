"""Python equivalent of GetOrientation.m (positions standard molecules in sites)."""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.USPEX.src.Rotate_rigid_body import Rotate_rigid_body
from gpevo.FunctionFolder.USPEX.src.Operate_molecule import Operate_molecule
from gpevo.FunctionFolder.USPEX.src.NEW_coord2Zmatrix import NEW_coord2Zmatrix
from gpevo.matrix import Frac2Cart


def GetOrientation(candidate, lat, num_sites, operation, mtype_list, nsym, P, PB):
    ORG = G["ORG_STRUC"]
    std = ORG.STDMOL
    lat = np.asarray(lat, dtype=float)
    molecules = []
    count = 0
    for i, site_count in enumerate(num_sites):
        mtype = mtype_list[count]
        mol_std = std[mtype - 1]
        coords = np.array(mol_std.molecule, dtype=float)
        coords = coords - np.mean(coords, axis=0)
        # random rigid-body orientation
        coords = Rotate_rigid_body([0, 0, 0], [1, 0, 0], coords, (np.random.rand() - 0.5) * 2 * np.pi)
        coords = Rotate_rigid_body([0, 0, 0], [0, 1, 0], coords, (np.random.rand() - 0.5) * 2 * np.pi)
        coords = Rotate_rigid_body([0, 0, 0], [0, 0, 1], coords, (np.random.rand() - 0.5) * 2 * np.pi)
        center = np.asarray(candidate[count], dtype=float)
        coords = coords + Frac2Cart(center, lat)

        for j in range(int(site_count)):
            opt = np.asarray(operation[(j * 4):(j * 4 + 4)], dtype=float)
            molecule = AttrDict()
            molecule.MOLCOORS = Operate_molecule(coords, opt, lat, P, PB)
            molecule.ZMATRIX = NEW_coord2Zmatrix(molecule.MOLCOORS, mol_std.format)
            molecule.ID = count + 1
            molecule.MOLCENTER = np.mean(molecule.MOLCOORS, axis=0)
            molecule.Operation = opt
            molecule.P = list(P)
            molecule.PB = list(PB)
            molecule.order = 0
            molecules.append(molecule)
        count += int(site_count)
    return molecules
