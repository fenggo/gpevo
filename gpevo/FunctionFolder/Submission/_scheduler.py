"""Shared scheduler helpers for the Python USPEX submission port."""
from __future__ import annotations

import os
import subprocess

from gpevo.runtime import G


# OpenMPI loads its InfiniBand/UCX components on every launch.  On a host
# without an RDMA NIC (no libibverbs device) those components are useless and,
# worse, once the UCX runtime is installed they segfault in MPI_Finalize:
#   ompi_osc_base_finalize -> mca_osc_ucx.so -> ucp_ep_destroy -> SIGSEGV
# The crash happens after the code has written its output, so the results are
# intact, but mpirun returns 139 and the job looks failed.  `osc`/`pml` fall
# back to ob1/pt2pt over shared memory and TCP, which is what a single-node
# run wants anyway; `btl=^openib` and the warn flag only silence the
# "unable to find any relevant network interfaces" noise.
# Entries are defaults only -- anything the user already exported (or set in
# INPUT.txt) wins.
MPI_ENV_DEFAULTS = {
    "OMPI_MCA_oob": "^ud",
    "OMPI_MCA_osc": "^ucx",
    "OMPI_MCA_pml": "^ucx",
    "OMPI_MCA_btl": "^openib",
    "OMPI_MCA_btl_base_warn_component_unused": "0",
}


def subprocess_env():
    """``os.environ`` plus the MPI defaults above, without overwriting them."""
    env = os.environ.copy()
    for key, value in MPI_ENV_DEFAULTS.items():
        env.setdefault(key, value)
    return env


def run_shell(command: str, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        shell=True,
        cwd=cwd or os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=subprocess_env(),
    )


def mark_job_record(job_number):
    """Create the CalcFoldTemp marker/Jobs.history entries used by MATLAB."""
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    if getattr(ORG, "platform", 0) <= 0:
        return
    temp = os.path.join(ORG.homePath, "CalcFoldTemp")
    os.makedirs(temp, exist_ok=True)
    marker = os.path.join(temp, f"{job_number}")
    try:
        with open(marker, "w") as fh:
            fh.write(f"Generation {POP.generation}\n")
    except OSError:
        pass
    try:
        with open(os.path.join(temp, "Jobs.history"), "a") as fh:
            fh.write(f"{job_number}  Submitted\n")
    except OSError:
        pass


def line_contains(status_text: str, needle: str) -> bool:
    return needle in (status_text or "")


def parse_qsub_jobid(output: str):
    """PBS/Torque ``qsub`` output is typically ``12345.host``."""
    if not output:
        return 0
    text = output.strip().split()[0]
    return text.split(".")[0]


def parse_bsub_jobid(output: str):
    """LSF ``bsub`` output: ``Job <12345> is submitted...``."""
    if not output:
        return 0
    start = output.find("<")
    end = output.find(">", start + 1)
    if start >= 0 and end > start:
        return output[start + 1:end]
    first = output.strip().split()[0]
    return first if first.isdigit() else 0


def parse_sbatch_jobid(output: str):
    """SLURM ``sbatch`` output: ``Submitted batch job 12345``."""
    if not output:
        return 0
    parts = output.strip().split()
    for i, part in enumerate(parts[:-1]):
        if part == "job" and parts[i + 1].isdigit():
            return parts[i + 1]
    for part in parts:
        if part.isdigit():
            return part
    return 0
