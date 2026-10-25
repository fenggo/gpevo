"""Python port of composEntropy.m (composition diversity of the best part)."""
from __future__ import annotations

import math

import numpy as np

from gpevo.runtime import G


def composEntropy():
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    from gpevo.FunctionFolder.USPEX.src.variation_common import valid_ranking

    ranking = valid_ranking(POP)
    n_top = int(round(len(ranking) * float(getattr(ORG, "bestFrac", 0.5))))
    n_top = min(n_top, len(ranking))

    aver = 0.0
    for i in range(n_top):
        for j in range(i + 1, n_top):
            aver += 1.0 - _cosine(POP.POPULATION[ranking[i]], POP.POPULATION[ranking[j]])

    if aver < 0.00001:
        return 0.0

    aver = 1.0
    c_entr = 0.0
    for i in range(n_top):
        for j in range(i + 1, n_top):
            dist = 1.0 - _cosine(POP.POPULATION[ranking[i]], POP.POPULATION[ranking[j]])
            if abs(1.0 - dist) < 0.001:
                c_entr -= dist / aver
            elif dist > 0:
                c_entr += dist * math.log(dist / aver) / aver
    if n_top > 1:
        c_entr = -2.0 * c_entr / (n_top * n_top - n_top)
    return c_entr


def _cosine(first, second):
    f1 = np.asarray(first.numIons, dtype=float)
    f2 = np.asarray(second.numIons, dtype=float)
    denom1 = float(np.dot(f1, f1))
    denom2 = float(np.dot(f2, f2))
    if denom1 <= 0 or denom2 <= 0:
        return 1.0
    return float(np.dot(f1, f2)) / math.sqrt(denom1 * denom2)
