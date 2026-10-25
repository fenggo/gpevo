"""Python equivalent of GetPOP_MOL.m."""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G


def GetPOP_MOL(numMols):
    ORG = G["ORG_STRUC"]
    mtype = []
    for idx, count in enumerate(numMols):
        mtype.extend([idx + 1] * int(count))
    atom_types = []
    for m in mtype:
        mol_types = ORG.STDMOL[m - 1].types
        for t in mol_types:
            if t > 0:
                atom_types.append(ORG.atomType[t - 1])
            elif len(ORG.atomType) == 1:
                atom_types.append(ORG.atomType[0])
    num_ions = [int(sum(1 for t in atom_types if t == v)) for v in ORG.atomType]
    return atom_types, mtype, num_ions
