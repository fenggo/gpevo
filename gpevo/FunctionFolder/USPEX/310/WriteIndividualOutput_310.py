"""Python equivalent of WriteIndividualOutput_310.m (minimal 310 subset)."""
from __future__ import annotations

import os
import shutil

import numpy as np

from gpevo.runtime import G
from gpevo.log_utils import write_log
from gpevo.FunctionFolder.sys.Write_POSCAR import Write_POSCAR, Write_POSCAR_order
from gpevo.FunctionFolder.sys.anasym_stokes import anasym_stokes, symmetrized_cif
from gpevo.FunctionFolder.USPEX.src.WriteGenerationOutput_310 import output_row


def _append_file(src, dst):
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "ab") as out:
        with open(src, "rb") as inp:
            shutil.copyfileobj(inp, out)


def WriteIndividualOutput_310(Ind_No):
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    s = POP.POPULATION[Ind_No]
    res_folder = getattr(POP, "resFolder", None) or ORG.resFolder
    atom_type = list(ORG.atomType)
    symg = getattr(s, "symg", 0)
    count = int(getattr(s, "Number", Ind_No + 1))

    num_ions_init = list(getattr(s, "INIT_numIons", getattr(s, "numIons", [])))
    lat_init = np.asarray(getattr(s, "INIT_LAT", getattr(s, "LATTICE", np.eye(3))), dtype=float)
    coor_init = np.asarray(getattr(s, "INIT_COORD", getattr(s, "COORDINATES", np.zeros((0, 3)))), dtype=float)
    Write_POSCAR(atom_type, count, symg, num_ions_init, lat_init, coor_init)
    _append_file("POSCAR", os.path.join(res_folder, "gatheredPOSCARS_unrelaxed"))

    num_ions = list(s.numIons)
    lattice = np.asarray(s.LATTICE, dtype=float)
    coor = np.asarray(s.COORDINATES, dtype=float)
    order = getattr(s, "order", None)

    # Symmetry only exists after relaxation, so it is determined here and not
    # at generation start (WriteIndividualOutput_310.m:20-31).
    if getattr(ORG, "doSpaceGroup", 0) == 1:
        s.symg = anasym_stokes(
            lattice, coor, num_ions, atom_type, getattr(ORG, "SGtolerance", 0.1)
        )
        symg = s.symg
        # Same accumulation as MATLAB: one data block per finished structure.
        cif_body = symmetrized_cif(
            lattice, coor, num_ions, atom_type, getattr(ORG, "SGtolerance", 0.1)
        )
        cif_path = os.path.join(res_folder, "symmetrized_structures.cif")
        os.makedirs(os.path.dirname(cif_path) or ".", exist_ok=True)
        with open(cif_path, "a", encoding="utf-8") as fp:
            fp.write(f"data_findsym-STRUC-{count}\n")
            fp.write(cif_body)

    Write_POSCAR(atom_type, count, symg, num_ions, lattice, coor)
    _append_file("POSCAR", os.path.join(res_folder, "gatheredPOSCARS"))
    Write_POSCAR_order(atom_type, count, symg, num_ions, lattice, coor, order)
    _append_file("POSCAR_order", os.path.join(res_folder, "gatheredPOSCARS_order"))

    # Record the structure in USPEX_STRUC.POPULATION (WriteIndividualOutput_310.m:37).
    from gpevo.FunctionFolder.USPEX.src.update_USPEX_INDIVIDUAL import update_USPEX_INDIVIDUAL

    update_USPEX_INDIVIDUAL(s, res_folder, getattr(POP, "generation", 0), atom_type)

    # One row per finished structure, as WriteOUTPUT.m does in MATLAB.
    write_log(output_row(s, Ind_No + 1, gp=getattr(ORG, "gp", 0)) + "\n")
