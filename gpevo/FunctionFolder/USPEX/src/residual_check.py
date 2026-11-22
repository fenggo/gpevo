"""Residual-based rejection of fake low-energy structures.

A collapsed geometry (atoms overlapping) produces a catastrophically large
GP residual.  The molecule checks in ``readMOL._check_molecules`` look at
intermolecular distances and at *new* intramolecular bonds only; CL-20 already
contains N-N bonds, so a short intramolecular N-N contact slips through both
gates.  This module is a deliberately separate, model-level screen -- it does
not replace the molecule checks, it only catches the extreme outliers they
miss (residual ~1e4 versus the normal 0.3-3).
"""
from __future__ import annotations

import math

import numpy as np

from gpevo.log_utils import console_print
from gpevo.runtime import G, FAKE_RESIDUAL_CUT

_FAKE_FLAG = "fakeResidualCheck"
_FAKE_MARK = "FAKE low-energy (excessive GP residual)"


def _cutoff(ORG=None):
    """Effective cutoff: INPUT `fakerescut` on *ORG*, else the module default.

    Falls back to ``G['ORG_STRUC']`` when *ORG* is omitted so callers that
    already hold the ORG can pass it explicitly, while simple predicate calls
    still work.  An invalid/missing value uses FAKE_RESIDUAL_CUT (100.0).
    """
    if ORG is None:
        ORG = G.get("ORG_STRUC")
    value = getattr(ORG, "fakeResCut", None) if ORG is not None else None
    try:
        value = float(value)
        if math.isfinite(value) and value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return FAKE_RESIDUAL_CUT


def _residual_value(s):
    """Return the finite residual on *s* as a scalar, or None.

    ``s.residual`` is the primary field; ``s.distance`` is the legacy mirror
    written by ``apply_gp_metrics`` for optType 13.  Missing, non-finite or
    non-scalar values yield None: lack of evidence never condemns a structure.
    """
    value = getattr(s, "residual", None)
    if value is None:
        value = getattr(s, "distance", None)
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.size != 1:
        return None
    scalar = float(array.reshape(-1)[0])
    return scalar if math.isfinite(scalar) else None


def is_fake_residual(s, cutoff=None, ORG=None) -> bool:
    """True only for a finite residual strictly greater than the cutoff.

    The cutoff comes from ``ORG.fakeResCut`` (INPUT `fakerescut`), falling back
    to FAKE_RESIDUAL_CUT.  An explicit *cutoff* overrides both.
    """
    value = _residual_value(s)
    if value is None:
        return False
    limit = float(cutoff) if cutoff is not None else _cutoff(ORG)
    return value > limit


def mark_fake_residual(s, label="", cutoff=None, ORG=None) -> bool:
    """Apply the ``readMOL.py`` failure pattern in place when *s* looks fake.

    Sets ``Enthalpies[-1] = 100000`` (the value the fitness and ``good``
    statistics treat as failed), appends a ``moleculeCheck`` marker, raises
    ``Error`` by 4 from step 2 on, and prints a terminal warning.  The GP
    fields (``residual`` / ``distance`` / ``gp`` / ``gp_feature``) are left
    untouched so the audit trail survives.

    Returns True only when a NEW mark was written -- repeated calls are
    idempotent and do not re-raise Error or re-print the warning.
    """
    limit = float(cutoff) if cutoff is not None else _cutoff(ORG)
    if not is_fake_residual(s, limit):
        return False
    if getattr(s, _FAKE_FLAG, False):
        return False

    value = _residual_value(s)
    enthalpies = np.asarray(s.Enthalpies, dtype=float).copy()
    enthalpies[-1] = 100000.0
    s.Enthalpies = enthalpies

    previous = str(getattr(s, "moleculeCheck", "") or "").strip()
    s.moleculeCheck = f"{previous} {_FAKE_MARK}".strip() if previous else _FAKE_MARK
    setattr(s, _FAKE_FLAG, True)

    step = getattr(s, "Step", None)
    try:
        if step is not None and np.size(step) and float(np.max(np.asarray(step, dtype=float))) >= 2:
            s.Error = int(getattr(s, "Error", 0)) + 4
    except (TypeError, ValueError):
        pass

    prefix = f"{label}: " if label else ""
    console_print(
        f"{prefix}FAKE low-energy rejected: GP residual {value:.4g} > "
        f"{limit:g}; Enthalpies[-1]=100000"
    )
    # Keep the rejected geometry for later force-field training; no physical
    # energy is attached (Enthalpies[-1] is the 100000 failure sentinel).
    # Import lazily: residual_check is imported early and the writer pulls in
    # ASE.  At most one frame per structure, first reason wins.
    try:
        from gpevo.FunctionFolder.USPEX.src.WriteIndividualsTraj import (
            append_rejected_traj)

        append_rejected_traj(
            s, "fake_residual",
            generation=getattr(G.get("POP_STRUC"), "generation", 0))
    except Exception:
        pass
    return True
