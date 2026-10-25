"""Python equivalent of Submission/checkStatus_local.m."""
from __future__ import annotations

from gpevo.FunctionFolder.Submission._scheduler import run_shell


def checkStatus_local(job_id):
    proc = run_shell(f"qstat {job_id}")
    status = proc.stdout
    running = ("R cfn_" in status) or ("Q cfn_" in status)
    if running:
        return 0
    run_shell("rm -f USPEX*")
    return 1
