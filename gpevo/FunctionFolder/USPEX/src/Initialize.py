"""Python equivalent of Initialize.m for calculationType=310."""
from __future__ import annotations

import os
from importlib import import_module as _im

from gpevo.runtime import G, AttrDict, struct


def Initialize():
    ORG = G["ORG_STRUC"]
    home = ORG.homePath
    os.makedirs(os.path.join(home, "Seeds"), exist_ok=True)
    os.makedirs(os.path.join(home, "AntiSeeds"), exist_ok=True)

    calc_type = f"{ORG.dimension}{ORG.molecule}{ORG.varcomp}"

    # Call initialize_POP_STRUC_310
    mod = _im("gpevo.FunctionFolder.USPEX." + calc_type + ".initialize_POP_STRUC_" + calc_type)
    init_func = getattr(mod, "initialize_POP_STRUC_" + calc_type)
    POP = init_func()

    POP.resFolder = ORG.resFolder

    USPEX = struct()
    USPEX.POPULATION = []
    USPEX.GENERATION = []
    USPEX.SYSTEM = AttrDict()
    USPEX.GENERATION.append(AttrDict(quasiEntropy=[], convex_hull=[], composEntropy=[],
                                     BestID=[], ID=[]))
    USPEX.POPULATION.append(AttrDict())
    USPEX.SYSTEM.atomType = list(ORG.atomType)
    USPEX.SYSTEM.atomType_symbol = [element_symbol(z) for z in ORG.atomType]
    USPEX.SYSTEM.Fp_weight = list(ORG.weight)

    G["POP_STRUC"] = POP
    G["USPEX_STRUC"] = USPEX

    from gpevo.runtime import safesave
    safesave("Current_POP.pkl", POP)
    safesave(os.path.join(POP.resFolder, "USPEX.pkl"), USPEX)

    # Table header of OUTPUT.txt, so that the first generation is titled too
    # (Initialize.m:75-83).
    log_file = getattr(ORG, "log_file", None) or "OUTPUT.txt"
    log_path = os.path.join(POP.resFolder, log_file)
    os.makedirs(POP.resFolder, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fp:
        fp.write("  ID   Origin     Composition  Enthalpy(eV)  Volume(A^3)  KPOINTS  SYMMETRY\n")
    return POP


def element_symbol(z):
    names = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S",
             "Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
             "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru",
             "Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce",
             "Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf",
             "Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn"]
    return names[z - 1] if 1 <= z <= len(names) else f"Z{z}"
