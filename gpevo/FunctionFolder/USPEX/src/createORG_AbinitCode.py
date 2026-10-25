"""Python equivalent of createORG_AbinitCode.m (310-relevant subset)."""
from __future__ import annotations

from gpevo.runtime import G
from gpevo.inputparser import InputParser, scannum


def _nonempty_block(lines):
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # command + colon notation
        if ":" in s:
            s = s.split(":", 1)[0].strip()
        s = s.split("!", 1)[0].strip()
        if s:
            out.append(s)
    return out


def createORG_AbinitCode(inputFile):
    ORG = G["ORG_STRUC"]
    p = InputParser(inputFile)

    raw = p.block("abinitioCode", "ENDabinit")
    vals = scannum(" ".join(raw))
    if vals is None:
        ORG.abinitioCode = [1]
    elif isinstance(vals, list):
        ORG.abinitioCode = [int(x) for x in vals]
    else:
        ORG.abinitioCode = [int(vals)]

    raw = p.scalar("whichCluster")
    if raw is not None:
        try:
            ORG.platform = abs(int(float(raw)))
        except ValueError:
            name = raw.strip().lower()
            mapping = {"nonparallel": 0, "cfn": 3, "qsh": 4, "qsh2": 5,
                       "xservde": 6, "mipt": 7, "nwpu": 8, "cfn-w": 9, "unn": 10}
            ORG.platform = mapping.get(name, 0)

    raw = p.block("KresolStart", "Kresolend")
    if raw:
        ORG.Kresol = scannum(" ".join(raw))
        if not isinstance(ORG.Kresol, list):
            ORG.Kresol = [ORG.Kresol]
    else:
        n = len(ORG.abinitioCode)
        ORG.Kresol = [0.2 - i * 0.12 / max(1, n - 1) for i in range(n)]

    raw = p.block("vacuumSize", "EndVacuumSize")
    if raw:
        ORG.vacuumSize = scannum(" ".join(raw))
    else:
        ORG.vacuumSize = 10.0

    raw = p.block("numProcessors", "EndProcessors")
    if raw:
        ORG.numProcessors = scannum(" ".join(raw))
        if not isinstance(ORG.numProcessors, list):
            ORG.numProcessors = [ORG.numProcessors]
    else:
        ORG.numProcessors = [8] * len(ORG.abinitioCode)

    raw = p.scalar("maxErrors")
    if raw is not None:
        ORG.maxErrors = int(float(raw))

    raw = p.scalar("numParallelCalcs")
    if raw is not None:
        ORG.numParallelCalcs = int(float(raw))

    raw = p.block("commandExecutable", "EndExecutable")
    if raw:
        ORG.commandExecutable = _nonempty_block(raw)
    else:
        ORG.commandExecutable = [""] * len(ORG.abinitioCode)

    raw = p.scalar("ExternalPressure")
    if raw is not None:
        ORG.ExternalPressure = float(raw)
    else:
        ORG.ExternalPressure = 0.0

    # Gaussian-process evaluation (uspexkit gp): `1 : gp`, `data : <dir>`,
    # `den : <density>`.  Each may also be written as `<value> : <keyword>`.
    raw = p.option("gp")
    ORG.gp = 0
    if raw is not None:
        try:
            ORG.gp = 1 if int(float(raw)) != 0 else 0
        except ValueError:
            ORG.gp = 0
    raw = p.option("data")
    ORG.gpData = str(raw) if raw else "data"
    raw = p.option("den")
    ORG.gpDen = 0.0
    if raw is not None:
        try:
            ORG.gpDen = float(raw)
        except ValueError:
            pass
    raw = p.option("ncpu")
    ORG.gpNcpu = 0
    if raw is not None:
        try:
            ORG.gpNcpu = int(float(raw))
        except ValueError:
            pass
    raw = p.option("u")
    if raw is not None:
        try:
            ORG.gpU = float(raw)
        except ValueError:
            pass
    raw = p.option("k")
    if raw is not None:
        try:
            ORG.gpK = int(float(raw))
        except ValueError:
            pass
    raw = p.option("mode")
    if raw is not None:
        ORG.gpMode = str(raw)

    if ORG.gp:
        # One extra step after the relaxations.  It repeats the last
        # abinitioCode so that the GULP `input` file uspexkit.gp() reads is
        # written for it; submitJob then runs the GP instead of the command.
        ORG.abinitioCode.append(ORG.abinitioCode[-1])
        if ORG.Kresol:
            ORG.Kresol.append(ORG.Kresol[-1])
        if ORG.numProcessors:
            ORG.numProcessors.append(ORG.numProcessors[-1])

    G["ORG_STRUC"] = ORG
