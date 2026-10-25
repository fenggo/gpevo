"""Python port of Correlation.m.

Correlates the average structure order with the fitness of the individuals that
completed successfully, and publishes the result on ``ORG`` so that the ordered
heredity (``heredity_molecule``) knows the preferred ordering direction.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G


def Correlation(fitness):
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]

    values = np.asarray(fitness, dtype=float).reshape(-1)
    population = list(POP.POPULATION)
    done_order = np.asarray(getattr(POP, "DoneOrder", None) or [0] * len(population),
                            dtype=float).reshape(-1)

    rows = []
    for i, individual in enumerate(population):
        if i >= values.size or values[i] >= 99999:
            continue
        if i < done_order.size and done_order[i] <= 0:
            continue
        order = np.asarray(getattr(individual, "order", None), dtype=float).reshape(-1)
        num_ions = np.asarray(individual.numIons, dtype=float).reshape(-1)
        total = float(np.sum(num_ions))
        if not np.isfinite(total) or total <= 0:
            continue
        rows.append([float(np.sum(order)) / total, float(values[i])])

    coefficient = 0.0
    if len(rows) > 1:
        table = np.asarray(rows, dtype=float)
        if np.std(table[:, 0]) > 0 and np.std(table[:, 1]) > 0:
            value = float(np.corrcoef(table[:, 0], table[:, 1])[0, 1])
            coefficient = value if np.isfinite(value) else 0.0

    ORG.correlation_coefficient = coefficient
    ORG.cor_dir = int(np.sign(coefficient)) if coefficient else 0
    G["ORG_STRUC"] = ORG
    return coefficient
