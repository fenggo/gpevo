"""Python equivalent of Submission/submitJob.m.

Supports the non-parallel platform and the scheduler platforms shipped with
USPEX (PBS/Torque, LSF, SGE, SLURM, and the local/remote customization
points).  For a given platform ``submitJob`` returns either a scheduler job
id or one of the special in-process codes ``0.01``/``0.02``.
"""
from __future__ import annotations

import os
import subprocess

from gpevo.runtime import G, resource_path
from gpevo.log_utils import console_print
from gpevo.FunctionFolder.Submission._scheduler import (
    mark_job_record,
    parse_bsub_jobid,
    parse_qsub_jobid,
    parse_sbatch_jobid,
    run_shell,
    subprocess_env,
)


def _format_bsub_jobname(ind_no, step):
    return f"USPEX-{ind_no}S{step}"


def _write_text(name, text):
    with open(name, "w") as fp:
        fp.write(text)


def _code_exec_line(ORG, code, step):
    """Return the scheduler-script line for a supported ab-initio code."""
    step = int(step)
    # The GP evaluation step has no commandExecutable entry of its own; on a
    # scheduler it still has to run as a command on the compute node.
    from gpevo.FunctionFolder.USPEX.src.gp_ml import is_gp_step

    if is_gp_step(ORG, step):
        from gpevo.FunctionFolder.USPEX.src.gp_ml import _ncpu

        return (f"uspexkit gp --n={_ncpu(ORG)} --data={ORG.gpData} "
                f"--den={ORG.gpDen} --u={ORG.gpU} --k={ORG.gpK} "
                f"--mode={ORG.gpMode} --optype={ORG.optType}")
    cmd = ORG.commandExecutable[step - 1] if len(ORG.commandExecutable) >= step else ""
    cmd = str(cmd or "")
    mapping = {
        1: "mpirun.lsf vasp-vdw",
        3: "gulp < input > output",
        4: "mpirun.lsf lammps < lammps.in > lammps.out",
        9: "mpirun.lsf aims.x > FHI_output",
        11: "RunCASTEP.sh -np 8 cstp >> log",
        13: "MOPAC2012.exe input.mop",
        14: "mpirun.lsf BoltzTraP BoltzTraP.def > BoltzTraP.output",
    }
    if code in mapping:
        return mapping[int(code)]
    return cmd


def _submit_pbs(platform, ORG, s, step, ind_no=0):
    if platform == 3:
        body = (
            "#!/bin/sh\n"
            "#PBS -l nodes=1:ppn=8,walltime=1:30:00\n"
            "#PBS -N USPEX\n"
            "#PBS -j oe\n"
            "#PBS -V \n"
            "cd ${PBS_O_WORKDIR}\n"
            "/home1/qzhu/source/MPI/bin/mpirun -np 8 /home1/qzhu/source/vasp.5.2/vasp > vasp.out\n"
        )
        script = "myrun_CFN"
    elif platform == 9:
        body = (
            "#!/bin/sh\n"
            "#PBS -l nodes=1:ppn=8,walltime=1:30:00 -q cfn_workshop\n"
            f"#PBS -N USPEX{ind_no}S{s.Step}\n"
            "#PBS -j oe\n"
            "#PBS -V \n"
            "ulimit -s unlimited\n"
            "source /opt/intel/2013/bin/compilervars.sh intel64\n"
            "cd ${PBS_O_WORKDIR}\n"
            "/software/mpi/openmpi/1.8.1-intel/bin/mpirun -np 8 /software/Workshop14/bin/vaspP > vasp.out\n"
        )
        script = "myrun_CFN"
    elif platform == 8:
        body = (
            "#!/bin/sh\n"
            "#PBS -N USPEX\n"
            "#PBS -l nodes=1:ppn=8\n"
            "#PBS -l walltime=6:00:00\n"
            "#PBS -q batch\n"
            "#PBS -V \n"
            "#PBS -S /bin/bash\n"
            "source /export/opensource/vasp.5.2/vaspenvSet.sh\n"
            "EXEC=~/bin/vasp53\n"
            "MPI_HOME=/export/compiler/intel/impi/4.1.0.024\n"
            "NP=`wc -l < $PBS_NODEFILE`\n"
            "cd ${PBS_O_WORKDIR}\n"
            "$MPI_HOME/bin/mpirun -machinefile $PBS_NODEFILE -np $NP $EXEC > out\n"
        )
        script = "jobrun"
    else:
        raise NotImplementedError(f"Unsupported PBS platform {platform}")

    _write_text(script, body)
    proc = run_shell(f"qsub {script}")
    if proc.returncode != 0:
        print(proc.stdout)
    return parse_qsub_jobid(proc.stdout)


def _submit_lsf(ORG, s, step, ind_no=0, platform=4):
    code = int(ORG.abinitioCode[step - 1])
    if code == 12:
        num_proc = 1
        min_proc = 1
    else:
        num_proc = int(ORG.numProcessors[step - 1])
        if num_proc > 8:
            min_proc = 8
            num_proc = 8
        else:
            min_proc = num_proc
    total_now_path = os.path.join(ORG.homePath, f"CalcFold{s.Folder}")
    lines = [
        "#!/bin/sh",
        "#BSUB -sp 60",
        "#BSUB -q  intel",
        "#BSUB -a  intelmpi",
        "#BSUB -o  output",
        f"#BSUB -R  \"span[ptile={num_proc}]\"",
        f"#BSUB -J {_format_bsub_jobname(ind_no, step)}",
        "#BSUB -W 06:00",
        f"#BSUB -n {min_proc} ",
        "cd $LS_SUBCWD",
    ]
    if code == 12 and ORG.dimension != -4:
        lines.append("bash ./tinker.sh")
    elif code == 12:
        lines.append(f"python {resource_path('FunctionFolder', 'USPEX', 'M400', 'random_protein', 'random_protein.py')} input 1 pseudo memory > output")
    else:
        lines.append(_code_exec_line(ORG, code, step))
    _write_text("myrun_QSH", "\n".join(lines) + "\n")
    proc = run_shell("bsub < myrun_QSH > job.info")
    id1 = parse_bsub_jobid(proc.stdout)
    if not id1:
        try:
            with open("job.info") as fp:
                id1 = parse_bsub_jobid(fp.read())
        except OSError:
            pass
    return id1 or 0


def _submit_sge(ORG, s, step, ind_no=0):
    code = int(ORG.abinitioCode[step - 1])
    script_lines = [
        "#!/bin/bash",
        "#$ -S /bin/bash",
        "#$ -N r30",
        "#$ -cwd",
        "#$ -l arch=darwin-x86",
    ]
    if ORG.numProcessors and step <= len(ORG.numProcessors):
        script_lines.append(f"#$ -pe openmpi {int(ORG.numProcessors[step - 1])}")
    script_lines += [". ~/.bashrc", "hostname", _code_exec_line(ORG, code, step)]
    _write_text("job", "\n".join(script_lines) + "\n")
    proc = run_shell("qsub job")
    out = proc.stdout.strip()
    if not out:
        return 0
    parts = out.split()
    for i, part in enumerate(parts[:-1]):
        if part == "job" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    for part in parts:
        if part.isdigit():
            return int(part)
    return 0


def _submit_slurm(ORG, s, step, platform, ind_no=0):
    code = int(ORG.abinitioCode[step - 1])
    if platform == 7:
        partition = "work"
    else:
        partition = "all"
    pop = G["POP_STRUC"]
    jobname = f"{pop.generation}I{ind_no}S{step}"
    lines = [
        "#!/bin/sh",
        "#SBATCH -o out",
        f"#SBATCH -p {partition}",
        f"#SBATCH -J {jobname}-USPEX",
        "#SBATCH -t 06:00:00",
        "#SBATCH -N 1",
        "#SBATCH -n 16",
        "",
        f"cd {os.path.join(ORG.homePath, f'CalcFold{s.Folder}')}",
        _code_exec_line(ORG, code, step),
    ]
    script = "myrun_MIPT" if platform == 7 else "myrun_UNN"
    _write_text(script, "\n".join(lines) + "\n")
    proc = run_shell(f"sbatch {script}")
    return parse_sbatch_jobid(proc.stdout)


def submitJob(ind_no):
    ORG = G["ORG_STRUC"]
    s = G["POP_STRUC"].POPULATION[ind_no]
    step = int(s.Step)
    code = int(ORG.abinitioCode[step - 1]) if len(ORG.abinitioCode) >= step else 0
    how_come = getattr(s, "howCome", "")
    platform = getattr(ORG, "platform", 0)

    if code == 0:
        return 0.02
    if platform == 0:
        return _submit_nonparallel(ind_no, step, code, how_come)

    from gpevo.FunctionFolder.Submission.submitJob_local import submitJob_local
    from gpevo.FunctionFolder.Submission.submitJob_remote import submitJob_remote

    if platform == 1:
        job_id = submitJob_local(step, ORG.commandExecutable[step - 1] if len(ORG.commandExecutable) >= step else "")
    elif platform == 2:
        job_id = submitJob_remote(ORG.remoteFolder, s.Folder)
    elif platform in (3, 8, 9):
        job_id = _submit_pbs(platform, ORG, s, step, ind_no)
    elif platform in (4, 5):
        job_id = _submit_lsf(ORG, s, step, ind_no, platform)
    elif platform == 6:
        job_id = _submit_sge(ORG, s, step, ind_no)
    elif platform in (7, 10):
        job_id = _submit_slurm(ORG, s, step, platform, ind_no)
    else:
        raise NotImplementedError(f"submitJob is not ported for platform={platform}")

    mark_job_record(job_id)
    return job_id


def _submit_nonparallel(ind_no, step, code, how_come):
    ORG = G["ORG_STRUC"]
    s = G["POP_STRUC"].POPULATION[ind_no]
    cmd = ORG.commandExecutable[step - 1] if len(ORG.commandExecutable) >= step else ""
    if "gulp" in cmd:
        if os.path.exists("input_orderd"):
            os.rename("input_orderd", f"inp_{G['POP_STRUC'].generation}_{ind_no}_{step}")

    if "mlflow" in cmd and "Random" in how_come:
        cmd = cmd.strip() + " --o=1"
    elif "gpcsp" in cmd:
        cmd = cmd.strip() + f" --r={ORG.resFolder}"

    cwd = os.path.join(ORG.homePath, f"CalcFold{s.Folder}")

    # `1 : gp` appends a GP evaluation step after the relaxations; it runs
    # uspexkit.gp(...) in-process instead of an external command.
    from gpevo.FunctionFolder.USPEX.src.gp_ml import (
        apply_gp_metrics, gp_db_ready, gp_db_unavailable, is_gp_step, run_gp,
    )

    if is_gp_step(ORG, step):
        # gp() returns its metrics; storing them on the structure replaces the
        # old round-trip through CalcFold1/gp.csv.  Nothing can be fitted until
        # the training set is seeded at the end of a generation, so skip it
        # (once per generation) instead of failing on every structure.
        if gp_db_ready(ORG):
            result = run_gp(ORG, cwd=cwd, id_=G["POP_STRUC"].bodyCount + 1)
            if result is None:
                # Terminal only: like the non-zero exit code of the subprocess it
                # replaces, this is a per-job diagnostic, not part of the run log.
                console_print("  Warning: uspexkit gp did not complete")
            else:
                apply_gp_metrics(s, result)
        else:
            gp_db_unavailable(ORG, getattr(G["POP_STRUC"], "generation", 0))
        return 100

    # print(f"  Running in CalcFold{s.Folder}: {cmd}")
    proc = subprocess.run(cmd, shell=True, cwd=cwd, env=subprocess_env())
    if proc.returncode != 0:
        # Terminal only: a non-zero exit of the external code is a per-job
        # diagnostic, not part of the run log.
        console_print(f"  Warning: command returned {proc.returncode}")
    return 100
