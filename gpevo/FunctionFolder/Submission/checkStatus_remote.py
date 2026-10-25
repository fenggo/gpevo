"""Python equivalent of Submission/checkStatus_remote.m."""
from __future__ import annotations

from gpevo.FunctionFolder.Submission._scheduler import run_shell


def checkStatus_remote(job_id, remote_folder="", folder=1):
    Home = "/nfs/user08/qiazhu"
    Address = "qiazhu@seawulf.stonybrook.edu"
    Path = f"{Home}/{remote_folder}/CalcFold{folder}"
    proc = run_shell(
        f"ssh -i ~/.ssh/seawulf {Address} /usr/local/pkg/torque/bin/qstat {job_id}"
    )
    status = proc.stdout
    running = ("R batch" in status) or ("Q batch" in status)
    if running:
        return 0
    for name in ("OSZICAR", "CONTCAR"):
        run_shell(f"scp -i ~/.ssh/seawulf {Address}:{Path}/{name} .")
    return 1
