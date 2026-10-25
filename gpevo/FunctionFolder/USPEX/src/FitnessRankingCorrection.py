"""Python port of FitnessRankingCorrection.m.

Sorts fitness ascending, pushes duplicated / failed structures to the tail via
``degradeSimilar`` and publishes ``POP.ranking`` / ``POP.bad_rank`` /
``POP.fitness``.  These are what the elitist scheme and parent selection read,
so this must run before ``KeepBestStructures``.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G, safesave
from gpevo.FunctionFolder.sys.degradeSimilar import degradeSimilar


def FitnessRankingCorrection(fitness):
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]

    fitness = np.asarray(fitness, dtype=float).copy()
    # MATLAB's sort is stable; keep tied fitness values in input order.
    ranking = [int(i) for i in np.argsort(fitness, kind="stable")]

    bad_rank = 0
    step = len(ORG.abinitioCode)
    ranking, bad_rank, fitness = degradeSimilar(
        float(getattr(ORG, "toleranceFing", 0.0)), ranking, fitness, step
    )

    soft_mut_only = getattr(ORG, "softMutOnly", None)
    if soft_mut_only is not None:
        generation = int(getattr(POP, "generation", 0))
        idx = min(generation + 1, len(soft_mut_only) - 1)
        if soft_mut_only[idx] == 1 and generation > 0:
            tolerance = 1.0 / ORG.populationSize
            delta_tol = tolerance / 2
            decent_rank = round(len(ranking) * ORG.bestFrac)
            decent_ranking = ranking[:decent_rank]
            done = False
            while not done:
                ranking, bad_rank, fitness = degradeSimilar(tolerance, decent_ranking, fitness, step)
                n_good = len(ranking) - bad_rank
                if 0.9 * ORG.populationSize <= n_good <= 1.1 * ORG.populationSize:
                    done = True
                elif n_good < 0.9 * ORG.populationSize:
                    tolerance = tolerance - delta_tol
                else:
                    tolerance = tolerance + delta_tol
                delta_tol = delta_tol / 2
                if delta_tol < 0.00001:
                    done = True

    POP.fitness = fitness
    POP.ranking = ranking
    POP.bad_rank = bad_rank
    G["POP_STRUC"] = POP

    # update_STUFF.m rebuilds ORG.tournament from the new ranking; the
    # variation operators read it for tournament selection.
    from gpevo.FunctionFolder.USPEX.src.variation_common import update_tournament

    update_tournament()
    safesave("Current_POP.pkl", POP)
    return fitness
