"""Python equivalent of SubmitJobs_310.m."""
from __future__ import annotations

import os

from gpevo.runtime import G
from gpevo.FunctionFolder.AbinitCode.Write_AbinitCode import Write_AbinitCode
from gpevo.FunctionFolder.Submission.submitJob import submitJob

_narg = 3  # placeholder


def SubmitJobs_310():
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    for indic in range(1, ORG.numParallelCalcs + 1):
        which = [i for i, s in enumerate(POP.POPULATION) if s.Folder == indic]
        do_now = None
        if not which:
            still_left = [i for i, s in enumerate(POP.POPULATION) if s.ToDo]
            if still_left:
                do_now = still_left[0]
                s = POP.POPULATION[do_now]
                s.Folder = indic
                s.ToDo = 0
        else:
            idx = which[0]
            if POP.POPULATION[idx].JobID == 0:
                do_now = idx

        if do_now is None:
            continue
        s = POP.POPULATION[do_now]
        step = int(s.Step)
        if step > len(ORG.abinitioCode):
            s.JobID = 0.01
        elif ORG.abinitioCode[step - 1] == 0:
            s.JobID = 0.02
        else:
            calcfold = os.path.join(ORG.homePath, f"CalcFold{indic}")
            os.makedirs(calcfold, exist_ok=True)
            os.chdir(calcfold)
            try:
                Write_AbinitCode(ORG.abinitioCode[step - 1], do_now)
                s.JobID = submitJob(do_now)
            finally:
                os.chdir(ORG.homePath)
    G["POP_STRUC"] = POP
