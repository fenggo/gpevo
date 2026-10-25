"""Python equivalent of Submission/submitJob_local.m."""
from __future__ import annotations

import os

from gpevo.FunctionFolder.Submission._scheduler import run_shell, parse_qsub_jobid


def submitJob_local(step=1, command_executable=""):
    script = "\n".join([
        "#!/bin/sh",
        "#PBS -l nodes=1:ppn=8,walltime=1:30:00 -q cfn_short",
        "#PBS -N USPEX",
        "#PBS -j oe",
        "#PBS -V ",
        "cd ${PBS_O_WORKDIR}",
        command_executable or "mpirun -np 4 vasp1 > vasp.out",
    ]) + "\n"
    with open("myrun", "w") as fp:
        fp.write(script)
    proc = run_shell("qsub myrun")
    if proc.returncode != 0:
        print(proc.stdout)
    return parse_qsub_jobid(proc.stdout)
