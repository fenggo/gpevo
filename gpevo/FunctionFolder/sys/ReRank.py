"""Python port of ``FunctionFolder/sys/ReRank.m``.

MATLAB sorts the population by the total number of ions (descending) and
rewrites ``POPULATION`` in that order.  It does **not** publish
``POP.ranking``; that is produced by ``FitnessRankingCorrection`` from the
actual fitness values.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G


def ReRank():
    POP = G["POP_STRUC"]
    population = POP.POPULATION
    n_atoms = []
    for individual in population:
        num_ions = getattr(individual, "numIons", [])
        try:
            n_atoms.append(float(sum(int(v) for v in num_ions)))
        except (TypeError, ValueError):
            n_atoms.append(0.0)

    # Stable descending sort keeps the original relative order inside ties,
    # which for fixed-composition runs (310) means the order is unchanged.
    order = sorted(range(len(n_atoms)), key=lambda i: (-n_atoms[i], i))
    POP.POPULATION = [population[i] for i in order]
    G["POP_STRUC"] = POP
