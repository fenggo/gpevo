"""Python equivalent of readMOL.m.

Two entry codes are used by the 310 flow: ``code == 3`` after reading a
finished GULP relaxation, and ``code == 0`` from ``pick_Seeds`` when building
an individual out of a seed POSCAR.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G, AttrDict
from gpevo.log_utils import console_print
from gpevo.FunctionFolder.USPEX.src.NEW_coord2Zmatrix import NEW_coord2Zmatrix
from gpevo.FunctionFolder.USPEX.src.newMolCheck import newMolCheck


def _movecoor(coords, lat, fmt):
    """Port of ``Movecoor``: wrap atoms back into one molecule using images."""
    coords = np.asarray(coords, dtype=float).copy()
    if coords.shape[0] > 1:
        rng = np.arange(-2, 3)
        grid = np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T
        shifts = grid @ lat
        fmt = np.asarray(fmt, dtype=int)
        for ind in range(1, coords.shape[0]):
            ref = int(fmt[ind, 0]) - 1
            ref = min(max(ref, 0), coords.shape[0] - 1)
            candidate = coords[ind] + shifts
            dist = np.linalg.norm(candidate - coords[ref], axis=1)
            coords[ind] = candidate[int(np.argmin(dist))]
    return coords


def readMOL(ind_no, code):
    ORG = G["ORG_STRUC"]
    s = G["POP_STRUC"].POPULATION[ind_no]

    if code == 3:
        coords = np.asarray(s.COORDINATES, dtype=float)
        lat = np.asarray(s.LATTICE, dtype=float).reshape(3, 3)
        mtype_list = s.MtypeLIST
        old_types = list(s.typesAList)
        offset = 0
        molecules = []
        for mi, mtype in enumerate(mtype_list):
            std = ORG.STDMOL[mtype - 1]
            n = len(std.types)
            frac = coords[offset:offset + n]
            mol = AttrDict()
            # MATLAB reassembles molecules split by the periodic boundary here
            # (Movecoor) before anything else reads them.
            mol.MOLCOORS = _movecoor(frac @ lat, lat, std.format)
            # MATLAB computes the Z-matrix from the coordinates actually read
            # back, so that the molecule checks can see decompositions.
            mol.ZMATRIX = NEW_coord2Zmatrix(mol.MOLCOORS, std.format)
            mol.MOLCENTER = np.mean(mol.MOLCOORS, axis=0) if n else np.zeros(3)
            mol.ID = mi + 1
            mol.order = 0
            molecules.append(mol)
            offset += n
        # Match MATLAB readMOL for GULP: COORDINATES is stored grouped by
        # atom type, while MOLECULES are reconstructed from the original order.
        reordered = []
        for atom_id in ORG.atomType:
            mask = [i for i, t in enumerate(old_types) if t == atom_id]
            if mask:
                reordered.append(coords[mask, :])
        if reordered:
            s.COORDINATES = np.concatenate(reordered, axis=0)

        s.MOLECULES = molecules
        _check_molecules(s, maxdist=0.6, label=str(int(ind_no) + 1))
        return

    # Seeds / standard-molecule path (MATLAB's ``else`` branch): COORDINATES
    # are grouped per atom type, so rebuild the molecule-sequential order.
    coords = np.asarray(s.COORDINATES, dtype=float)
    lat = np.asarray(s.LATTICE, dtype=float).reshape(3, 3)
    mtype_list = [int(v) for v in s.MtypeLIST]
    num_ions = [int(v) for v in s.numIons]

    item = [0] * len(ORG.atomType)
    rows = []
    for mtype in mtype_list:
        for ind in ORG.STDMOL[mtype - 1].types:
            idx = int(ind) - 1
            if idx < 0 or idx >= len(item):
                continue
            item[idx] += 1
            pos = sum(num_ions[:idx]) + item[idx] - 1
            if pos >= coords.shape[0]:
                break
            rows.append(coords[pos])
    new_coords = np.asarray(rows, dtype=float) if rows else np.zeros((0, 3))

    molecules = []
    offset = 0
    for mi, mtype in enumerate(mtype_list):
        std = ORG.STDMOL[mtype - 1]
        n = len(std.types)
        block = new_coords[offset:offset + n] @ lat
        offset += n
        block = _movecoor(block, lat, std.format)
        mol = AttrDict()
        mol.MOLCOORS = block
        mol.ZMATRIX = NEW_coord2Zmatrix(block, std.format)
        mol.MOLCENTER = block[0] if n == 1 else np.mean(block, axis=0)
        mol.ID = mi + 1
        mol.order = 0
        molecules.append(mol)
    s.MOLECULES = molecules
    _check_molecules(s, maxdist=0.6, label=str(int(ind_no) + 1))


def _check_molecules(s, maxdist=0.6, label=""):
    """Port of the ``ORG.checkMolecules`` block at the end of readMOL.m.

    A molecule whose Z-matrix no longer matches the standard molecule (or that
    fails the intermolecular distance check) is marked as failed: the last
    enthalpy becomes 100000 and, from the second relaxation step on, the error
    counter is raised by 4.
    """
    ORG = G["ORG_STRUC"]
    if not int(getattr(ORG, "checkMolecules", 0)):
        return

    step = getattr(s, "Step", None)
    if step is None:
        return
    try:
        if np.size(step) == 0:
            return
    except TypeError:
        pass

    lattice = np.asarray(s.LATTICE, dtype=float).reshape(3, 3)
    mtype_list = [int(v) for v in s.MtypeLIST]
    for mi, mtype in enumerate(mtype_list):
        std = ORG.STDMOL[mtype - 1]
        zmatrix = np.asarray(s.MOLECULES[mi].ZMATRIX, dtype=float)
        zmatrix_std = np.asarray(std.ZMATRIX, dtype=float)

        decompose = 0
        n_rows = min(zmatrix.shape[0], zmatrix_std.shape[0])
        if n_rows > 1:
            for loop in range(1, n_rows):
                if abs(float(zmatrix[loop, 0]) - float(zmatrix_std[loop, 0])) > maxdist:
                    decompose = 1
                    break

        try:
            poly = newMolCheck(s.MOLECULES, lattice, mtype_list, ORG.minDistMatrice)
        except Exception:
            poly = 1

        if decompose or poly == 0:
            enthalpies = np.asarray(s.Enthalpies, dtype=float).copy()
            enthalpies[-1] = 100000.0
            s.Enthalpies = enthalpies
            s.moleculeCheck = "MOLECULE FORM is not kept"
            # readMOL.m:142 uses disp(), so the warning stays on the terminal
            # and is not written to OUTPUT.txt; the 1-based index is the one
            # passed to the function, not POPULATION(i).Number (0 when the
            # individual has not been registered yet).
            console_print(f"{label}: MOLECULE FORM is not kept, "
                          f"see Warnings for more details ...")
            if int(np.max(np.asarray(step, dtype=float))) >= 2:
                s.Error = int(getattr(s, "Error", 0)) + 4
            break
