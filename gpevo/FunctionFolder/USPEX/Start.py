"""Python equivalent of USPEX/Start.m dispatcher."""
from __future__ import annotations

import os
import time
from importlib import import_module as _im

import numpy as np

from gpevo.runtime import G, reset_workspace, set_home
from gpevo.log_utils import configure_log
from gpevo.FunctionFolder.USPEX.src.createORGStruc import createORGStruc
from gpevo.FunctionFolder.USPEX.src.Initialize import Initialize
from gpevo.FunctionFolder.sys.CreateCalcFolder import CreateCalcFolder
from gpevo.log_utils import console_print


def calcTypeStr():
    ORG = G["ORG_STRUC"]
    return f"{ORG.dimension}{ORG.molecule}{ORG.varcomp}"


def _setup_rng(ORG):
    # Seed numpy and random modules similar to USPEX's clock-based seeding.
    now = time.time()
    seed = int((now * 1000000) % (2**31))
    np.random.seed(seed)
    import random as _random
    _random.seed(seed)


def Start():
    _setup_rng(None)

    # The run directory comes from the current working directory.
    # Bundled resources are resolved from the package itself.
    working = os.getcwd()
    set_home(working)

    ORG = G["ORG_STRUC"]
    console_print("  ")

    if os.path.exists("Current_POP.pkl"):
        from gpevo.runtime import safeload
        POP = safeload("Current_POP.pkl")
        ORG = safeload("Current_ORG.pkl")
        if not getattr(POP, "resFolder", None):
            POP.resFolder = ORG.resFolder
        G["POP_STRUC"] = POP
        G["ORG_STRUC"] = ORG
        # Refresh fingerprint weights.  Older checkpoints were saved before
        # the molecular weight bug was fixed and may therefore contain an
        # atom-type weight vector sized for numMols instead of numIons.
        from gpevo.FunctionFolder.USPEX.src.createORG_Fingerprint import createORG_Fingerprint
        createORG_Fingerprint("INPUT.txt")
        # Reprocess already-relaxed individuals from a finished checkpoint so
        # restarted runs no longer keep N/A descriptor columns.
        from gpevo.FunctionFolder.sys.fingerprint_utils import compute_structure_metrics
        for _s in POP.POPULATION:
            if getattr(_s, "symg", None) is None or str(getattr(_s, "symg", "")).strip().upper() == "N/A":
                _s.symg = 1
            if int(getattr(_s, "Done", 0)) == 1:
                try:
                    compute_structure_metrics(_s, ORG)
                except Exception:
                    pass
        # Work in result folder (matches POP structure)
        if POP.resFolder and os.path.exists(POP.resFolder):
            os.makedirs(POP.resFolder, exist_ok=True)
        CreateCalcFolder()
    else:
        ORG = createORGStruc("INPUT.txt")
        G["ORG_STRUC"] = ORG
        CreateCalcFolder(1)
        Initialize()
        CreateCalcFolder()

    for name in ("ORG_STRUC", "POP_STRUC"):
        if name not in G or G[name] is None:
            raise RuntimeError(f"{name} is not initialized")

    ORG = G["ORG_STRUC"]
    configure_log(os.path.join(ORG.resFolder, ORG.log_file or "OUTPUT.txt"))
    # Save current state before entering generation loop (restartable).
    from gpevo.runtime import safesave
    safesave("Current_ORG.pkl", G["ORG_STRUC"])
    safesave("Current_POP.pkl", G["POP_STRUC"])

    ORG = G["ORG_STRUC"]
    if (ORG.dimension == 3 and ORG.molecule == 1) or \
       (ORG.dimension == 1 and ORG.molecule == 1 and ORG.varcomp == 0):
        for name in os.listdir(ORG.homePath):
            if name.startswith("MOL_"):
                shutil_copy = __import__("shutil").copy
                shutil_copy(os.path.join(ORG.homePath, name), os.path.join(ORG.resFolder, name))
    elif ORG.dimension == 2 and ORG.molecule == 0:
        src = os.path.join(ORG.homePath, "POSCAR_SUBSTRATE")
        if os.path.exists(src):
            shutil_copy(os.path.join(ORG.homePath, "POSCAR_SUBSTRATE"),
                        os.path.join(ORG.resFolder, "POSCAR_SUBSTRATE"))

    calc_type = calcTypeStr()
    module = _im("gpevo.FunctionFolder.USPEX." + calc_type + ".EA_" + calc_type)
    EA = getattr(module, "EA_" + calc_type)
    EA()

    ORG = G["ORG_STRUC"]
    if getattr(ORG, "repeatForStatistics", 1) > 1:
        console_print("Repeat statistics mode is not implemented in this Python port yet")
    else:
        if os.path.exists("still_running"):
            os.remove("still_running")

    console_print("USPEX IS DONE!")
    with open("USPEX_IS_DONE", "w") as fh:
        fh.write("1\n")
