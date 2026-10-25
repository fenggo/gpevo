"""Python equivalent of Write_GULP.m (GULP, code 3)."""
from __future__ import annotations

import os
import subprocess

import numpy as np

from gpevo.runtime import G
from gpevo.FunctionFolder.AbinitCode.Write_GULP_MOL import Write_GULP_MOL


def Write_GULP(ind_no):
    ORG = G["ORG_STRUC"]
    s = G["POP_STRUC"].POPULATION[ind_no]
    step = int(s.Step)
    for name in ("output", "optimized.structure"):
        if os.path.exists(name):
            os.remove(name)

    # Start input from a per-run goptions_<step> file if one exists.
    if os.path.exists("input"):
        os.remove("input")
    seed_file = f"goptions_{step}"
    if os.path.exists(seed_file):
        with open(seed_file) as fh, open("input", "w") as out:
            out.write(fh.read())

    if ORG.molecule == 1:
        Write_GULP_MOL(ind_no, 0)
    else:
        raise NotImplementedError("Non-molecular GULP writer is not ported yet")

    ginput = f"ginput_{step}"
    if os.path.exists(ginput):
        with open(ginput) as fh, open("input", "a") as out:
            out.write("\n" + fh.read())
    if ORG.ExternalPressure:
        with open("input", "a") as out:
            out.write(f"\npressure\n{ORG.ExternalPressure}\n")
    with open("input", "a") as out:
        out.write("\ndump every optimized.structure\n")
