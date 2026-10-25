"""Python port of oldQuasiEntropy.m (fingerprint diversity of the best part)."""
from __future__ import annotations

import math

import numpy as np

from gpevo.runtime import G
from gpevo.FunctionFolder.sys.fingerprint_utils import _cosine_distance as cosine_distance


def oldQuasiEntropy():
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    from gpevo.FunctionFolder.USPEX.src.variation_common import valid_ranking

    ranking = valid_ranking(POP)
    n_top = int(round(len(ranking) * float(getattr(ORG, "bestFrac", 0.5))))
    n_top = min(n_top, len(ranking))

    entr = 0.0
    for i in range(n_top):
        for j in range(i + 1, n_top):
            f1 = getattr(POP.POPULATION[ranking[i]], "FINGERPRINT", None)
            f2 = getattr(POP.POPULATION[ranking[j]], "FINGERPRINT", None)
            if f1 is None or f2 is None or np.size(f1) == 0 or np.size(f2) == 0:
                cos_dist = 1.0
            else:
                weight = 1.0 if int(getattr(ORG, "varcomp", 0)) == 1 else ORG.weight
                rows = np.asarray(f1).shape[0]
                weight = np.asarray(weight, dtype=float)
                if weight.ndim == 0 or weight.size != rows:
                    weight = np.ones(rows, dtype=float)
                try:
                    cos_dist = float(cosine_distance(f1, f2, weight))
                except Exception:
                    cos_dist = 1.0
            if 1.0 - cos_dist > 0:
                entr += (1.0 - cos_dist) * math.log(1.0 - cos_dist)

    if n_top > 1:
        return 2.0 * (-entr) / (n_top * n_top - n_top)
    return 0.0
