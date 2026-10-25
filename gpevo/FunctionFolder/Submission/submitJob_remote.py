"""Python equivalent of Submission/submitJob_remote.m."""
from __future__ import annotations

import os

from gpevo.FunctionFolder.Submission._scheduler import run_shell


def submitJob_remote(remote_folder="", folder=1):
    # Keep the MATLAB customization points explicit so a user can edit them
    # for their own ssh/scp/torque setup.
    Home = "/nfs/user08/qiazhu"
    Address = "qiazhu@seawulf.stonybrook.edu"
    Path = f"{Home}/{remote_folder}/CalcFold{folder}"

    script = "\n".join([
        "#!/bin/sh",
        "#PBS -l nodes=2:ppn=2,walltime=1:30:00",
        "#PBS -N USPEX",
        "#PBS -j oe",
        "#PBS -V ",
        "cd ${PBS_O_WORKDIR}",
        "/usr/local/pkg/openmpi-1.4.5/bin/mpirun -np 4 vasp1 > vasp.out",
    ]) + "\n"
    with open("runvasp.sh", "w") as fp:
        fp.write(script)

    run_shell(f"ssh -i ~/.ssh/seawulf {Address} mkdir -p {Path}")
    for name in ("POSCAR", "INCAR", "POTCAR", "KPOINTS", "runvasp.sh"):
        if os.path.exists(name):
            run_shell(f"scp -i ~/.ssh/seawulf {name} {Address}:{Path}")

    proc = run_shell(
        f"ssh -i ~/.ssh/seawulf {Address} /usr/local/pkg/torque/bin/qsub {Path}/runvasp.sh"
    )
    out = proc.stdout.strip()
    if "error" in out.lower():
        return 0
    if "." in out:
        return out.split()[0].split(".")[0]
    return out.split()[0] if out else 0
