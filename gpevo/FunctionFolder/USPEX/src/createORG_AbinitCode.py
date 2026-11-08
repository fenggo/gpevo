"""Python equivalent of createORG_AbinitCode.m (310-relevant subset)."""
from __future__ import annotations

import glob
import os
import shutil

from gpevo.runtime import G
from gpevo.inputparser import InputParser, scannum, strip_comment

# Only needed for the training-set row count below; gp_ml imports nothing from
# this module, so there is no import cycle.
from gpevo.FunctionFolder.USPEX.src.gp_ml import csv_rows

# Options of the GP block; `gp` is the switch, the rest only matter when it is on.
_GP_OPTIONS = ("gp", "ncpu", "data", "den", "u", "k", "mode", "nboot",
               "cache", "explore", "gpbackend", "gpiter", "gplr", "gpplot")


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


def _gp_option_lines(p):
    """INPUT.txt lines that look like GP options, comment characters included.

    The raw text is kept so the warning shows exactly what was written; the
    scan is on the first token of either side of the colon, which is also how
    ``InputParser.option`` matches, so a line it lists is one it can read.
    """
    found = []
    for raw in p.lines:
        line = strip_comment(raw).strip()
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        for side in (right, left):
            tokens = side.split()
            if tokens and tokens[0].lower() in _GP_OPTIONS:
                found.append(line)
                break
    return found


# uspexkit.gp() reads these from ../<data> relative to a CalcFold directory.
_GP_DATA_FILES = ("feature_mlp.csv", "feature.csv", "structures.traj")


def _check_gp_data(ORG):
    """Report the state of the GP training directory (uspexkit.gp).

    A missing or incomplete directory is no longer fatal: the best structures
    of the first generation are labelled at its end to seed it (see
    ``gp_ml.gp_active_learning``).  What still needs checking here is whether
    that labelling can actually run.
    """
    from gpevo.log_utils import console_print

    home = getattr(ORG, "homePath", None) or "."
    data_dir = os.path.join(home, str(getattr(ORG, "gpData", "data") or "data"))
    if not os.path.isdir(data_dir):
        console_print(f"  Note: GP data directory '{data_dir}' does not exist yet;")
        console_print("        it is created automatically and seeded with the best "
                      "structures of the first generation.")
        _check_reference_run(ORG)
        return
    missing = [name for name in _GP_DATA_FILES
               if not os.path.isfile(os.path.join(data_dir, name))]
    if missing:
        console_print(f"  Note: GP data directory '{data_dir}' is missing "
                      f"{', '.join(missing)};")
        console_print("        missing files are written when the first labelled "
                      "structures are added.")
    rows = csv_rows(os.path.join(data_dir, "feature.csv"))
    if rows >= 2:
        console_print(f"  GP training set: {rows} labelled rows in {data_dir}")
        return
    console_print(f"  GP training set holds only {rows} labelled row(s) in "
                  f"{data_dir}: it is seeded from the first generation.")
    _check_reference_run(ORG)


def _check_reference_run(ORG):
    """Is the reference calculation behind the labels actually runnable?

    The labels come from ``uspexkit.calc`` -> ``siesta_opt`` (mpirun + the
    pseudopotentials of ``Specific/``); when either is missing the loop prints
    a warning per generation and nothing ever enters the training set, so it is
    worth saying so once, up front.
    """
    from gpevo.log_utils import console_print

    missing = [name for name in ("siesta", "mpirun") if shutil.which(name) is None]
    if missing:
        console_print("  Warning: {} not found on PATH: labelled structures cannot "
                      "be produced, so the GP training set cannot grow.".format(
                          " and ".join(missing)))
    home = getattr(ORG, "homePath", None) or "."
    specific = os.path.join(home, str(getattr(ORG, "specificFolder", "Specific")))
    if not glob.glob(os.path.join(specific, "*.psf")):
        console_print(f"  Warning: no pseudopotential (*.psf) in {specific}; the "
                      "reference calculation needs them.")


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
    raw = p.option("nboot")
    if raw is not None:
        try:
            ORG.gpNboot = max(0, int(float(raw)))
        except ValueError:
            pass
    raw = p.option("cache")
    if raw is not None:
        try:
            ORG.gpCache = 1 if int(float(raw)) != 0 else 0
        except ValueError:
            pass
    raw = p.option("explore")
    if raw is not None:
        try:
            ORG.gpExplore = 1 if int(float(raw)) != 0 else 0
        except ValueError:
            pass
    raw = p.option("gpplot")
    ORG.gpPlot = 0                  # off unless `1 : gpplot` is written
    if raw is not None:
        try:
            ORG.gpPlot = 1 if int(float(raw)) != 0 else 0
        except ValueError:
            pass
    raw = p.option("gpbackend")
    if raw is not None:
        backend = str(raw).lower()
        if backend in ("sklearn", "gpytorch"):
            ORG.gpBackend = backend
        else:
            from gpevo.log_utils import console_print
            console_print(f"  Warning: unknown gpbackend '{raw}', "
                          "falling back to sklearn")
    raw = p.option("gpiter")
    if raw is not None:
        try:
            ORG.gpTrainingIter = max(1, int(float(raw)))
        except ValueError:
            pass
    raw = p.option("gplr")
    if raw is not None:
        try:
            ORG.gpLr = float(raw)
        except ValueError:
            pass

    if ORG.gp:
        # One extra step after the relaxations.  It repeats the last
        # abinitioCode so that the GULP `input` file uspexkit.gp() reads is
        # written for it; submitJob then runs the GP instead of the command.
        ORG.abinitioCode.append(ORG.abinitioCode[-1])
        if ORG.Kresol:
            ORG.Kresol.append(ORG.Kresol[-1])
        if ORG.numProcessors:
            ORG.numProcessors.append(ORG.numProcessors[-1])
        _check_gp_data(ORG)
    elif _gp_option_lines(p):
        # GP options are present but the switch did not turn on.  Without this
        # the whole GP block silently does nothing -- no gp.csv, no GP columns
        # in OUTPUT.txt, no active learning -- which is easy to miss.
        from gpevo.log_utils import console_print

        console_print("  Warning: INPUT.txt contains GP options but 'gp' is "
                      "0 or unreadable;")
        console_print("           the GP step, its predictions and the GP "
                      "columns of OUTPUT.txt are disabled.")
        console_print("           offending lines:")
        for line in _gp_option_lines(p):
            console_print(f"             {line}")

    G["ORG_STRUC"] = ORG
