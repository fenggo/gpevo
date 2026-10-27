"""Python equivalent of pickAntiSeeds.m.

Reads the user-supplied ``AntiSeeds/POSCARS`` file (several concatenated
VASP-style blocks), computes each structure's fingerprint and stores it in
``G["ANTISEEDS"]`` as a list of ``{FINGERPRINT, Sigma, Max}`` entries.

Port of ``../uspex-gp/FunctionFolder/USPEX/src/pickAntiSeeds.m``: the reference
``cd AntiSeeds`` + ``fopen('POSCARS')`` loop is replaced here by the shared
``_read_poscar_blocks`` parser, and its ``makeMatrices``/``fingerprint_calc``
pair by ``compute_full_atom_fingerprint``.
"""

from __future__ import annotations

import os

import numpy as np

from gpevo.log_utils import console_print
from gpevo.runtime import G, safesave

# Checkpoint for the global ANTISEEDS structure (MATLAB: ANTISEEDS.mat).
ANTISEEDS_FILE = "Current_ANTISEEDS.pkl"


def pickAntiSeeds():
    """Load every structure in ``AntiSeeds/POSCARS`` as an anti-seed.

    Returns the list of entries (also stored in ``G["ANTISEEDS"]`` and saved
    next to ``Current_ORG.pkl``).  A missing or unreadable POSCARS leaves the
    list empty without raising, matching the ``try/catch`` of the reference.
    """
    ORG = G.get("ORG_STRUC")
    if ORG is None:
        return []

    home = getattr(ORG, "homePath", None) or os.getcwd()
    poscars = os.path.join(home, "AntiSeeds", "POSCARS")
    if not os.path.isfile(poscars):
        return []

    from gpevo.FunctionFolder.USPEX.src.pick_Seeds import _read_poscar_blocks

    try:
        blocks = _read_poscar_blocks(poscars)
    except Exception as exc:
        console_print(f"  Warning: could not read AntiSeeds/POSCARS ({exc})")
        return []
    if not blocks:
        return []

    console_print("Read AntiSeeds ...")

    from gpevo.FunctionFolder.sys.fingerprint_utils import (
        compute_full_atom_fingerprint,
    )

    rmax = float(getattr(ORG, "RmaxFing", 10.0))
    sigma = float(getattr(ORG, "sigmaFing", 0.03))
    delta = float(getattr(ORG, "deltaFing", 0.08))
    dimension = int(getattr(ORG, "dimension", 3))

    entries = []
    for block in blocks:
        lattice = np.asarray(block["lattice"], dtype=np.float64).reshape(3, 3)
        num_ions = [int(v) for v in block["counts"]]
        coords = np.asarray(block["coords"], dtype=np.float64)
        if not block.get("cartesian", False):
            # Direct/fractional -> Cartesian, which is what the fingerprint
            # routines expect (they build periodic images from the lattice rows).
            coords = coords @ lattice
        try:
            (_, fing, _), _volume = compute_full_atom_fingerprint(
                lattice,
                coords,
                num_ions,
                rmax=rmax,
                sigma=sigma,
                delta=delta,
                dimension=dimension,
                # pickAntiSeeds.m:30-31 filters no intra-molecular pairs.
                intra_map=None,
            )
        except Exception as exc:
            console_print(f"  Warning: AntiSeed fingerprint failed ({exc})")
            continue

        entries.append(
            {
                "FINGERPRINT": np.asarray(fing, dtype=np.float64),
                "Sigma": float(getattr(ORG, "antiSeedsSigma", 0.001)),
                "Max": float(getattr(ORG, "antiSeedsMax", 0.0)),
            }
        )
        console_print(f" --> AntiSeed {len(entries)} added ...")

    G["ANTISEEDS"] = entries
    safesave(ANTISEEDS_FILE, entries, workdir=home)
    return entries
