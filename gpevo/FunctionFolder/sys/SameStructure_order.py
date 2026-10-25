"""Python port of ``FunctionFolder/sys/SameStructure_order.m``.

Order-aware variant used for 3D: two structures count as the same when the
composition matches and both the mean order parameter and the enthalpy per ion
are within tolerance.  Unlike ``SameStructure`` it never looks at fingerprints.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G


def _ion_count(record):
    try:
        return int(sum(int(v) for v in record.numIons))
    except (TypeError, ValueError):
        return 0


def _mean_order(record, molecule):
    if molecule:
        values = [float(getattr(mol, "order", 0.0)) for mol in getattr(record, "MOLECULES", []) or []]
    else:
        values = [float(v) for v in np.asarray(getattr(record, "order", []), dtype=float).reshape(-1)]
    if not values:
        return 0.0
    return float(np.mean(values))


def SameStructure_order(ID1, ID2, POP):
    """``ID1``/``ID2`` are zero-based indices into ``POP.POPULATION``."""
    ORG = G["ORG_STRUC"]
    population = POP.POPULATION
    if ID1 is None or ID2 is None:
        return False
    if ID1 < 0 or ID2 < 0 or ID1 >= len(population) or ID2 >= len(population):
        return False
    first = population[ID1]
    second = population[ID2]
    if first is None or second is None:
        return False

    num_ions1 = list(getattr(first, "numIons", []) or [])
    num_ions2 = list(getattr(second, "numIons", []) or [])
    if [int(v) for v in num_ions1] != [int(v) for v in num_ions2]:
        return False

    tolerance_o = 0.075
    tolerance_e = 0.015

    total1 = max(1, _ion_count(first))
    total2 = max(1, _ion_count(second))
    e1 = float(np.asarray(first.Enthalpies, dtype=float).ravel()[-1]) / total1
    e2 = float(np.asarray(second.Enthalpies, dtype=float).ravel()[-1]) / total2

    molecule = int(getattr(ORG, "molecule", 0))
    ao1 = _mean_order(first, molecule)
    ao2 = _mean_order(second, molecule)

    return bool(abs(ao1 - ao2) < tolerance_o and abs(e1 - e2) < tolerance_e)
