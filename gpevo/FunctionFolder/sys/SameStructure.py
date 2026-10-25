"""Python port of ``FunctionFolder/sys/SameStructure.m``."""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G
from .fingerprint_utils import _cosine_distance as cosine_distance


def SameStructure(id1, id2, pop):
    """Return True when the two structures are judged identical.

    ``id1``/``id2`` are zero-based indices into ``pop.POPULATION``; the MATLAB
    original uses one-based indices.
    """
    population = pop.POPULATION
    first = population[id1]
    second = population[id2]

    num_ions1 = [int(v) for v in first.numIons]
    num_ions2 = [int(v) for v in second.numIons]
    if num_ions1 != num_ions2:
        return False

    ORG = G["ORG_STRUC"]
    molecule = int(getattr(ORG, "molecule", 0))
    tolerance_f = float(getattr(ORG, "toleranceFing", 0.0))
    tolerance_e = 0.005 if molecule else 0.015
    tolerance_o = 0.002 if molecule else 0.075

    fingerprint1 = getattr(first, "FINGERPRINT", None)
    fingerprint2 = getattr(second, "FINGERPRINT", None)
    if fingerprint1 is None or fingerprint2 is None:
        return False

    rows = np.asarray(fingerprint1).shape[0]
    weight = np.asarray(getattr(ORG, "weight", 1.0), dtype=float)
    if weight.ndim == 0 or weight.size != rows:
        weight = np.ones(rows, dtype=float)

    enthalpy1 = float(np.asarray(first.Enthalpies, dtype=float).ravel()[-1]) / max(1, sum(num_ions1))
    enthalpy2 = float(np.asarray(second.Enthalpies, dtype=float).ravel()[-1]) / max(1, sum(num_ions2))

    dist_f = cosine_distance(fingerprint1, fingerprint2, weight)

    order1 = np.asarray(getattr(first, "order", []), dtype=float)
    order2 = np.asarray(getattr(second, "order", []), dtype=float)
    order1 = order1.ravel()
    order2 = order2.ravel()
    mean_order1 = float(order1.mean()) if order1.size else 0.0
    mean_order2 = float(order2.mean()) if order2.size else 0.0
    dist_o = abs(mean_order1 - mean_order2)
    diff = abs(enthalpy1 - enthalpy2)

    dimension = int(getattr(ORG, "dimension", 3))
    if dimension in (0, 2):
        return bool(dist_f < tolerance_f and diff < tolerance_e)
    return bool((dist_f < tolerance_f or dist_o < tolerance_o) and diff < tolerance_e)
