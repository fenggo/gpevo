"""Python equivalent of CreateCalcFolder.m (non-parallel + temp folders)."""
from __future__ import annotations

import os
import shutil

from gpevo.runtime import G, resource_path


def CreateCalcFolder(flag=None):
    ORG = G["ORG_STRUC"]
    home = ORG.homePath
    n = getattr(ORG, "numParallelCalcs", 1)

    if flag == 1:
        os.makedirs(os.path.join(home, "CalcFoldTemp", ".data"), exist_ok=True)
        spg_data = resource_path("FunctionFolder", "spacegroup", "data")
        if os.path.isdir(spg_data):
            for fn in os.listdir(spg_data):
                src = os.path.join(spg_data, fn)
                dst = os.path.join(home, "CalcFoldTemp", ".data", fn)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
    else:
        for fold in range(1, n + 1):
            calc = os.path.join(home, f"CalcFold{fold}")
            os.makedirs(calc, exist_ok=True)

            spec = os.path.join(home, ORG.specificFolder)
            if os.path.isdir(spec):
                for fn in os.listdir(spec):
                    src = os.path.join(spec, fn)
                    dst = os.path.join(calc, fn)
                    if os.path.isfile(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)

    for name in ("Seeds", "AntiSeeds"):
        os.makedirs(os.path.join(home, name), exist_ok=True)
    return 0
