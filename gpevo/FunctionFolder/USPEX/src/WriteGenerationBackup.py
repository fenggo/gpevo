"""Python equivalent of WriteGenerationBackup.m (per-generation snapshot)."""
from __future__ import annotations

import os

from gpevo.runtime import G, safesave


def WriteGenerationBackup():
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    os.makedirs(ORG.resFolder, exist_ok=True)

    # Keep the top-level checkpoint: Start() restarts a run from it.
    safesave("Current_ORG.pkl", ORG)
    safesave("Current_POP.pkl", POP)

    # Per-generation snapshot, as WriteGenerationBackup.m does.
    current_dir = getattr(POP, "current_dir", None) or os.path.join(
        ORG.resFolder, f"generation{int(getattr(POP, 'generation', 0))}"
    )
    os.makedirs(current_dir, exist_ok=True)
    safesave(os.path.join(current_dir, "POP_STRUC.pkl"), POP)
    safesave(os.path.join(current_dir, "ORG_STRUC.pkl"), ORG)
    if G.get("USPEX_STRUC") is not None:
        safesave(os.path.join(current_dir, "USPEX.pkl"), G["USPEX_STRUC"])
    if G.get("POOL_STRUC") is not None:
        safesave(os.path.join(current_dir, "POOL.pkl"), G["POOL_STRUC"])
