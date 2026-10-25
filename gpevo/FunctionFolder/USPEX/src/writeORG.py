"""Python port of ``FunctionFolder/USPEX/src/writeORG.m``.

Writes the run description block of ``OUTPUT.txt``: the investigated system,
the evolutionary-algorithm settings and the ab initio relaxation steps.  It is
called at the end of ``createORGStruc`` (``createORGStruc.m:221``), i.e. once
the whole INPUT has been parsed and before the first structure is generated.
"""
from __future__ import annotations

import os

from gpevo.runtime import G
from gpevo.log_utils import write_log
from gpevo.FunctionFolder.sys.alignLine import alignLine

# abinitioCode -> label, as writeORG.m:199-234.
ABINITIO_CODES = {
    0: " no relaxation",
    1: "     VASP     ",
    2: "    SIESTA    ",
    3: "     GULP     ",
    4: "    LAMMPS    ",
    5: " Neu Network  ",
    6: "   DMACRYS    ",
    7: "    CP2K      ",
    8: "    PWSCF     ",
    9: "   FHI-aims   ",
    10: "     ATK      ",
    11: "   CASTEP     ",
    12: "   Tinker     ",
    13: "    MOPAC     ",
    14: "  BoltzTraP   ",
}

# writeORG.m:250-264, keyed by ORG.platform.
PLATFORM_TEXT = {
    0: "The calculations are performed in nonParallel mode on the local machine",
    1: "The script for job submission is prepared seperately in Submission/*_local.m",
    2: "The script for job submission is prepared seperately in Submission/*_remote.m",
    3: "The calculations are performed in Parallel mode on CFN supercomputer",
    4: "The calculations are performed in Parallel mode on QSH supercomputer",
    5: "The calculations are performed in Parallel mode on QSH supercomputer",
    6: "The calculations are performed in Parallel mode on xservDE supercomputer",
    7: "The calculations are performed in Parallel mode on MIPT supercomputer",
}


def _element_symbol(number):
    from gpevo.FunctionFolder.USPEX.src.Initialize import element_symbol

    return element_symbol(int(number))


def _investigated_system(ORG):
    """The 'The investigated system is:' line (writeORG.m:122-149)."""
    text = "    The investigated system is: "
    if int(getattr(ORG, "varcomp", 0)) == 1:
        blocks = []
        for row in ORG.numIons:
            parts = []
            for j, count in enumerate(row):
                if int(count) <= 0:
                    continue
                if int(getattr(ORG, "molecule", 0)) == 1:
                    name = ORG.STDMOL[j].name
                    parts.append(f"{name:>6s}_{int(count):1d}")
                else:
                    parts.append(f"{_element_symbol(ORG.atomType[j]):>2s}_{int(count):1d}")
            blocks.append("[" + "".join(parts) + "]")
        text += " --- ".join(blocks)
    elif int(getattr(ORG, "molecule", 0)) == 1:
        for i, count in enumerate(ORG.numMols):
            text += f"{ORG.STDMOL[i].name:>6s}_{int(count):1d}"
    else:
        for i, count in enumerate(ORG.numIons):
            text += f"{_element_symbol(ORG.atomType[i]):>2s}_{int(count):2d}  "
    return text


def _volume_lines(ORG):
    """writeORG.m:163-188."""
    if int(getattr(ORG, "constLattice", 0)):
        lines = ["\n* This is a fixed lattice calculation "]
        lattice = ORG.lattice
        for row in lattice:
            lines.append("".join(f"{float(v):6.3f}  " for v in row))
        return lines
    if int(getattr(ORG, "varcomp", 0)) == 1:
        lines = ["\n*  Initial Volume for atom/block : "]
        for i, row in enumerate(ORG.numIons):
            label = "    Volume of atom/block  "
            for j, count in enumerate(row):
                if int(count) != 0:
                    if int(getattr(ORG, "molecule", 0)) == 1:
                        label += f" {ORG.STDMOL[j].name}_{int(count)}"
                    else:
                        label += f" {_element_symbol(ORG.atomType[j])}_{int(count)}"
            volume = ORG.latVolume[i] if hasattr(ORG.latVolume, "__len__") else ORG.latVolume
            lines.append(f"{label}  : {float(volume):6.3f}   A^3")
        return lines
    return [f"     Estimated Volume : {float(ORG.latVolume):6.3f}  A^3"]


def writeORG():
    """Append the run description block to ``OUTPUT.txt``."""
    ORG = G["ORG_STRUC"]
    lines = [_investigated_system(ORG)]
    lines.append(alignLine("Block for evolutionary algorithm"))
    lines.append(f"                 Number of Generations  :    {int(ORG.numGenerations):4d}")
    lines.append(f"               Initial Population Size  :    {int(ORG.initialPopSize):4d}")
    lines.append(f"               General Population Size  :    {int(ORG.populationSize):4d}")
    lines.append(alignLine("AB INITIO CALCULATIONS"))

    pressure = getattr(ORG, "ExternalPressure", None)
    if pressure is not None:
        lines.append(f"*  External Pressure is : {float(pressure):6.4f}  GPa*")
    lines.extend(_volume_lines(ORG))
    lines.append("")

    codes = list(getattr(ORG, "abinitioCode", []) or [])
    total = len(codes)
    lines.append(f"*  There are {total:2d} local relaxation steps for each individual structure  *")
    platform = int(getattr(ORG, "platform", 0))
    if platform == 0:
        lines.append("Step  Abinitio Code                  Execute Command            K-resolution ")
    else:
        lines.append("Step  Abinitio Code    K-resolution ")

    kresol = list(getattr(ORG, "Kresol", []) or [])
    commands = list(getattr(ORG, "commandExecutable", []) or [])
    for step in range(total):
        code = ABINITIO_CODES.get(int(codes[step]), None)
        if code is None:
            lines.append("*" * 72)
            lines.append("**      ERROR:    The code you selected is not vaild!             **")
            lines.append("**                 Program STOPS!!!!!!!!!!!                       **")
            lines.append("*" * 72)
            continue
        # writeORG.m:235-242: GULP/LAMMPS/NN/DMACRYS have no k-point grid.
        resolution = 0.0 if int(codes[step]) in (3, 4, 5, 6) else float(
            kresol[step] if step < len(kresol) else 0.0
        )
        command = commands[step] if step < len(commands) else ""
        if platform == 0:
            lines.append(f" {step + 1:2d} {code:12s}  {command:36s}  {resolution:12.3f}")
        else:
            lines.append(f" {step + 1:2d} {code:12s}  {resolution:12.3f}")

    lines.append("")
    lines.append(PLATFORM_TEXT.get(platform, PLATFORM_TEXT[0]))
    lines.append(f"{int(getattr(ORG, 'numParallelCalcs', 1)):4d} parallel calculations are performed simutaneously")
    lines.append("")

    text = "\n".join(lines) + "\n"
    write_log(text)

    # write_log() is a no-op until configure_log() has opened the log (that
    # happens in Start.py), so make sure the block is on disk either way.
    path = os.path.join(str(getattr(ORG, "resFolder", "")), str(getattr(ORG, "log_file", "") or "OUTPUT.txt"))
    from gpevo.log_utils import _LOG_FP

    if _LOG_FP is None and path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8", errors="replace") as fp:
            fp.write(text)
    return text
