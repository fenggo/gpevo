"""Python port of ``FunctionFolder/sys/degradeSimilar.m``.

Structures whose fingerprint is empty, whose enthalpy at the current step is
above the failure threshold, or that duplicate an already accepted structure
are pushed to the end of ``ranking`` and counted in ``bad_rank``.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G
from .SameStructure import SameStructure


def degradeSimilar(tolerance, ranking1, fitness1, step):
    # ``tolerance`` is kept for call-site compatibility: MATLAB passes it but
    # SameStructure reads ORG_STRUC.toleranceFing directly.
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]

    bad_rank = 0
    ranking = list(ranking1)
    fitness = np.asarray(fitness1, dtype=float).copy()
    if fitness.ndim == 0:
        fitness = np.asarray([float(fitness)])

    it = 1
    while it < len(ranking):
        moved = False
        for j in range(it):
            individual = POP.POPULATION[ranking[it]]
            fingerprint = getattr(individual, "FINGERPRINT", None)
            if fingerprint is None or np.size(fingerprint) == 0:
                moved = True
            else:
                enthalpies = np.asarray(individual.Enthalpies, dtype=float).ravel()
                index = min(max(int(step) - 1, 0), max(enthalpies.size - 1, 0))
                bad_enthalpy = bool(enthalpies.size and enthalpies[index] >= 9999)
                moved = bad_enthalpy or SameStructure(ranking[j], ranking[it], POP)
            if moved:
                temp_rank = ranking.pop(it)
                ranking.append(temp_rank)
                if int(getattr(ORG, "molecule", 0)) == 1:
                    fitness[ranking[-1]] = 100000.0
                bad_rank += 1
                it -= 1
                break
        if it + bad_rank == len(ranking):
            break
        it += 1

    return ranking, bad_rank, fitness
