"""Python equivalent of ``FunctionFolder/USPEX/310/use_gp_density.m``."""
from __future__ import annotations

import math


def use_gp_density(pop, density):
    """Prefer ``pop.density_gp`` when it is a finite scalar, else keep fallback."""
    try:
        value = getattr(pop, "density_gp", None)
        if value is None or isinstance(value, str):
            return density
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return density
