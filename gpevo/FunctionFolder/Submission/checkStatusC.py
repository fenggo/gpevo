"""Python equivalent of Submission/checkStatusC.m.

Implements the scheduler status probes for the USPEX cluster platforms.
All probes use ``subprocess`` and therefore do not require a MATLAB/Octave
installation.
"""
from __future__ import annotations

import os

from gpevo.runtime import G
from gpevo.FunctionFolder.Submission._scheduler import run_shell
from gpevo.FunctionFolder.Submission.checkStatus_local import checkStatus_local
from gpevo.FunctionFolder.Submission.checkStatus_remote import checkStatus_remote


def checkStatusC(ind_no):
    ORG = G["ORG_STRUC"]
    s = G["POP_STRUC"].POPULATION[ind_no]
    job_id = getattr(s, "JobID", 0)
    if job_id in (0.01, 0.02):
        return 1
    if job_id == 0:
        return 0

    platform = getattr(ORG, "platform", 0)
    folder = getattr(s, "Folder", 1)
    cwd = os.path.join(ORG.homePath, f"CalcFold{folder}")

    if platform == 0:
        return 1
    if platform == 1:
        return checkStatus_local(job_id)
    if platform == 2:
        return checkStatus_remote(job_id, ORG.remoteFolder, folder)
    if platform == 6:
        proc = run_shell(f"qstat -j {job_id}", cwd=cwd)
        status = proc.stdout or ""
        return 1 if ("does not exist" in status.lower() or "exist" in status.lower()) else 0
    if platform in (3, 8, 9):
        proc = run_shell(f"qstat {job_id}", cwd=cwd)
        status = proc.stdout or ""
        queue_noun = "cfn_" if platform in (3, 9) else "batch"
        running = (f"R {queue_noun}" in status) or (f"Q {queue_noun}" in status)
        return 0 if running else 1
    if platform in (4, 5):
        proc = run_shell(f"bjobs {job_id}", cwd=cwd)
        status = proc.stdout or ""
        if "not found" in status.lower() or "is not found" in status.lower() or "found" in status.lower():
            return 1
        for done_key in ("DONE", "EXIT", "UNKWN", "ZOMBI"):
            if done_key in status.upper():
                return 1
        return 0
    if platform in (7, 10):
        job_file = os.path.join(ORG.homePath, "jobinfo.dat")
        if platform == 7:
            run_shell(f"squeue > {job_file}", cwd=cwd)
        else:
            run_shell(f"squeue > {job_file}", cwd=cwd)
        try:
            with open(job_file, errors="replace") as fp:
                status = fp.read()
        except OSError:
            status = ""
        lines = [ln for ln in status.splitlines() if "JOBID" not in ln and "CG" not in ln]
        for line in lines:
            if str(job_id) in line:
                return 0
        return 1
    raise NotImplementedError(f"checkStatusC is not ported for platform={platform}")
