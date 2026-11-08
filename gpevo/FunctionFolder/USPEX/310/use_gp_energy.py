"""Prefer the GP-predicted enthalpy for optType 1 fitness.

Symmetric with ``use_gp_density`` (optType 11): when the GP step is enabled and
the individual carries a GP-predicted enthalpy, optType 1 ranks on that
prediction rather than on the force-field (GULP/ReaxFF) enthalpy.  ``Enthalpies``
itself is left alone -- CalcFitness still uses it for the failed-structure check.
"""
from __future__ import annotations

import math


def use_gp_energy(pop, enthalpy):
    """Prefer ``pop.energy_gp`` when it is a finite scalar, else keep fallback."""
    try:
        value = getattr(pop, "energy_gp", None)
        if value is None or isinstance(value, str):
            return enthalpy
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return enthalpy
