"""Python equivalent of Finish.m: write final output and mark finish."""
from __future__ import annotations

from gpevo.runtime import G
from gpevo.log_utils import console_print


def Finish():
    ORG = G["ORG_STRUC"]
    with open(f"{ORG.resFolder}/Fitness.txt", "w") as fp:
        pop = G["POP_STRUC"].POPULATION
        for i, s in enumerate(pop, 1):
            e = s.Enthalpies[-1] if hasattr(s, "Enthalpies") and len(s.Enthalpies) else 0
            fp.write(f"{i:5d} {e:20.10f}\n")
    console_print("USPEX IS DONE!")
    with open("USPEX_IS_DONE", "w") as fp:
        fp.write("1\n")
    raise SystemExit(0)
