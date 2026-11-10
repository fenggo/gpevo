"""Python equivalent of Finish.m: write final output and mark finish."""
from __future__ import annotations

from datetime import datetime

from gpevo.runtime import G
from gpevo.log_utils import write_log, console_print

# MATLAB's ``datestr(now)`` prints English month abbreviations whatever the
# locale is; Python's ``%b`` follows LC_TIME ("10月" under zh_CN), so the
# months are listed explicitly to keep the stamp byte-identical to USPEX.
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _job_finished_stamp():
    """``datestr(now)`` equivalent: ``dd-mmm-yyyy HH:MM:SS``."""
    now = datetime.now()
    return (f"{now.day:02d}-{_MONTHS[now.month - 1]}-{now.year} "
            f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}")


def Finish():
    ORG = G["ORG_STRUC"]
    with open(f"{ORG.resFolder}/Fitness.txt", "w") as fp:
        pop = G["POP_STRUC"].POPULATION
        for i, s in enumerate(pop, 1):
            e = s.Enthalpies[-1] if hasattr(s, "Enthalpies") and len(s.Enthalpies) else 0
            fp.write(f"{i:5d} {e:20.10f}\n")
    # Finish.m:6-20 writes this block with fprintf(fp, ...) only, so it lands in
    # OUTPUT.txt and not on the terminal -- hence write_log, not print.
    write_log(f"This calculation runs {int(G['POP_STRUC'].bodyCount):4d} "
              f"structural relaxations\n")
    write_log("\n")
    write_log(f"            Job Finished at       {_job_finished_stamp():>30}\n")
    # Finish.m parks the generation past the limit and saves the checkpoint
    # before quitting: a restart loading it then sees EA_310's while-loop as
    # false and does not produce another generation.  Leaving generation at
    # its last value made restarting a finished run resume the search.
    POP = G["POP_STRUC"]
    POP.generation = int(ORG.numGenerations) + 1
    from gpevo.runtime import safesave

    safesave("Current_POP.pkl", POP)
    console_print("USPEX IS DONE!")
    with open("USPEX_IS_DONE", "w") as fp:
        fp.write("1\n")
    raise SystemExit(0)
