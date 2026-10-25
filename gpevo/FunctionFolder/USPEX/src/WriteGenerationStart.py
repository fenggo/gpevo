"""Python equivalent of ``FunctionFolder/USPEX/src/WriteGenerationStart.m``."""
from __future__ import annotations

import os

from gpevo.runtime import G


def WriteGenerationStart():
    """Initialize bookkeeping/directories for a new generation."""
    POP = G["POP_STRUC"]
    POP.ranking = 0
    POP.bad_rank = 0
    POP.current_dir = os.path.join(
        str(getattr(POP, "resFolder", "")) or os.getcwd(),
        f"generation{POP.generation}",
    )
    aux_dir = os.path.join(POP.resFolder or os.getcwd(), "AuxiliaryFiles")
    os.makedirs(aux_dir, exist_ok=True)
    os.makedirs(POP.current_dir, exist_ok=True)
    G["POP_STRUC"] = POP
