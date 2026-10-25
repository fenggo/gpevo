"""Python equivalent of src/Reading.m for GULP/molecular crystals."""
from __future__ import annotations

import os
import numpy as np

from gpevo.runtime import G, isempty
from gpevo.FunctionFolder.AbinitCode.Read_AbinitCode import Read_AbinitCode
from gpevo.FunctionFolder.AbinitCode.Read_GULP_Structure import Read_GULP_Structure
from gpevo.FunctionFolder.AbinitCode.readMOL import readMOL
from gpevo.FunctionFolder.sys.distanceCheck import distanceCheck


def Reading(code, ind_no, indic):
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    Error = 0
    cwd = ORG.homePath
    calcfold = os.path.join(cwd, f"CalcFold{indic}")
    os.chdir(calcfold)
    try:
        s = POP.POPULATION[ind_no]
        step = int(s.Step)
        good_bad = Read_AbinitCode(code, 0)
        if good_bad == 1:
            coords, lattice = Read_GULP_Structure()
            s.COORDINATES = coords
            s.LATTICE = lattice
            energy = Read_AbinitCode(code, 1)
            if energy is None or energy > 99999:
                energy = 100000.0
            if isinstance(s.Enthalpies, np.ndarray):
                s.Enthalpies[step - 1] = energy
            else:
                arr = list(s.Enthalpies)
                arr[step - 1] = energy
                s.Enthalpies = np.asarray(arr)
            if ORG.molecule == 1:
                readMOL(ind_no, code)
            POP.POPULATION[ind_no].Step = step + 1
        else:
            Error = 1
            if code in (1, 8, 9, 14):
                s.Error += 1
            else:
                s.Error = ORG.maxErrors + 1
    finally:
        os.chdir(cwd)
    G["POP_STRUC"] = POP
    return Error
