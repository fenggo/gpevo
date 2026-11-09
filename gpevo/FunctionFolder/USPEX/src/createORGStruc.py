"""Python equivalent of createORGStruc.m for the 310 path."""
from __future__ import annotations

import os
import math

import numpy as np

from gpevo.runtime import G, mkdir, home_path, struct, unix, isempty
from gpevo.inputparser import InputParser, scannum
from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault
from gpevo.FunctionFolder.USPEX.src.createORG_System import createORG_System
from gpevo.FunctionFolder.USPEX.src.createORG_AbinitCode import createORG_AbinitCode
from gpevo.FunctionFolder.USPEX.src.createORG_Symmetry import createORG_Symmetry
from gpevo.FunctionFolder.USPEX.src.createORG_EA import createORG_EA
from gpevo.FunctionFolder.USPEX.src.createORG_Fingerprint import createORG_Fingerprint
from gpevo.FunctionFolder.USPEX.src.createORG_Molecule import createORG_Molecule
from gpevo.matrix import latConverter, det
from gpevo.log_utils import console_print


def check_optType_13(ORG):
    """Refuse to run optType 13 without the GP step.

    optType 13 optimises the Hilbert-space distance to the training set.  That
    distance **is** the GP residual — ``CalcFitness_310.m:46-53`` reads
    ``POPULATION(i).distance``, and only the GP step ever fills it — so without
    ``1 : gp`` every fitness would silently become NaN.  Fail fast instead.
    """
    if int(getattr(ORG, "optType", 1) or 1) != 13:
        return
    if int(getattr(ORG, "gp", 0) or 0) != 0:
        return
    console_print("optType 13 optimises `distance` (Hilbert-space distance to the")
    console_print("training set), which is the residual returned by the GP step.")
    console_print("Without `1 : gp` no structure ever gets a distance, so every")
    console_print("fitness would be NaN. The program is STOPS here........")
    raise RuntimeError(
        "optType 13 requires the GP step: set `1 : gp` in INPUT.txt "
        "(and make sure its training data directory is usable)."
    )


def createORGStruc(inputFile="INPUT.txt"):
    ORG = createORGDefault()
    G["ORG_STRUC"] = ORG

    # Result folder: results1, results2, ...
    folder = 0
    while True:
        folder += 1
        res = os.path.join(ORG.homePath, f"results{folder}")
        if not os.path.exists(res):
            os.makedirs(res, exist_ok=True)
            break
    ORG.resFolder = res

    # Banner / citation block at the top of OUTPUT.txt (createORGStruc.m:56-95).
    from gpevo.FunctionFolder.sys.createHeader import write_output_header

    write_output_header(os.path.join(res, ORG.log_file or "OUTPUT.txt"))

    # Parse system components in same order as MATLAB
    createORG_System(inputFile)
    createORG_AbinitCode(inputFile)
    createORG_Symmetry(inputFile)
    createORG_EA(inputFile)

    if os.path.exists(os.path.join(ORG.homePath, "MOL_1")):
        createORG_Molecule(inputFile)
    else:
        if ORG.molecule == 1:
            console_print("For Molecules, you did not give MOL_x files")
            console_print("The program is STOPS here........")
            raise RuntimeError("Missing MOL_* files; copy the test MOL_* files into the run directory.")

    createORG_Fingerprint(inputFile)

    check_optType_13(ORG)

    # Latticevalues
    p = InputParser(inputFile)
    block = p.block("Latticevalues", "Endvalues")
    lat = scannum(" ".join(block)) if block else None
    if lat is None:
        ORG.constLattice = 0
        # estimate input volume from existing binary tool or simple default
        ORG.latVolume = 1000.0
    else:
        if isinstance(lat, list):
            if len(lat) == 6:
                ORG.constLattice = 1
                ORG.lattice = latConverter(np.array(lat))
                ORG.latVolume = det(ORG.lattice)
            elif len(lat) == 9:
                ORG.constLattice = 1
                ORG.lattice = np.asarray(lat).reshape(3, 3)
                ORG.latVolume = det(ORG.lattice)
            else:
                ORG.latVolume = float(lat[0])
                ORG.constLattice = 0
        else:
            ORG.latVolume = float(lat)
            ORG.constLattice = 0

    if ORG.dimension in (0, 2, -3):
        ORG.constLattice = 1

    raw = p.scalar("initialPopSize")
    if raw is None:
        raw = ORG.populationSize
    ORG.initialPopSize = int(float(raw))

    raw = p.scalar("repeatForStatistics")
    if raw is not None:
        ORG.repeatForStatistics = int(float(raw))

    raw = p.scalar("checkConnectivity")
    if raw is None:
        raw = 1 if ORG.optType == 3 else 0
    ORG.checkConnectivity = int(float(raw))

    raw = p.scalar("PhaseDiagram")
    if raw is not None:
        ORG.PhaseDiagram = int(float(raw))
    elif ORG.dimension != 3:
        ORG.PhaseDiagram = 0

    raw = p.scalar("stopFitness")
    if raw is not None:
        ORG.stopFitness = float(raw)

    # Run description block in OUTPUT.txt (createORGStruc.m:221).
    from gpevo.FunctionFolder.USPEX.src.writeORG import writeORG

    writeORG()

    # Save ORG snapshot for incremental continuation
    from gpevo.runtime import safesave
    safesave("Current_ORG.gpevo.pkl", ORG)
    G["ORG_STRUC"] = ORG
    return ORG
