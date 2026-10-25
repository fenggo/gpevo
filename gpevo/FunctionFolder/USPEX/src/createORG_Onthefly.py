"""Python equivalent of createORG_Onthefly.m (re-read adjustable blocks)."""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G
from gpevo.inputparser import InputParser


def _numeric_rows(lines):
    """Drop blank and ``%``-comment lines so blocks may carry annotations."""
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("%")]


def createORG_Onthefly(inputFile):
    ORG = G["ORG_STRUC"]
    p = InputParser(inputFile)

    hard_core = p.block("IonDistances", "EndDistances")
    if hard_core:
        try:
            matrix = np.array(
                [[float(v) for v in line.split()] for line in _numeric_rows(hard_core)],
                dtype=float,
            )
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("block is not square: %s" % (matrix.shape,))
            size = min(len(ORG.atomType), matrix.shape[0])
            for i in range(size):
                for j in range(i, size):
                    value = matrix[i, j]
                    ORG.minDistMatrice[i, j] = value
                    ORG.minDistMatrice[j, i] = value
        except (ValueError, IndexError, TypeError) as exc:
            print(
                "createORG_Onthefly: cannot read the IonDistances block of "
                f"{inputFile} ({exc}); keeping the previous minDistMatrice."
            )

    raw = p.scalar("numParallelCalcs")
    if raw is not None:
        try:
            ORG.numParallelCalcs = int(float(raw))
        except ValueError:
            pass

    centers = p.block("MolCenters", "EndMol")
    if centers:
        try:
            ORG.CenterminDistMatrice = np.array(
                [[float(v) for v in line.split()] for line in _numeric_rows(centers)],
                dtype=float,
            )
        except ValueError as exc:
            print(
                "createORG_Onthefly: cannot read the MolCenters block of "
                f"{inputFile} ({exc}); keeping the previous CenterminDistMatrice."
            )

    G["ORG_STRUC"] = ORG
