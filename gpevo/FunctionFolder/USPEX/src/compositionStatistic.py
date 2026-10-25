"""Python port of compositionStatistic.m.

Counts, per composition block, how many structures came from which operator and
stores the table in ``USPEX_STRUC.statistic``.  It works on the cumulative
record ``USPEX_STRUC.POPULATION``, so it only sees structures once
``update_USPEX_INDIVIDUAL`` has been wired in.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G

# Column offsets inside ``statistic`` (MATLAB is 1-based, here 0-based):
# total, random, heredity, mutations, seeds, copex, other.
MUTATION_COLUMN = 3
OTHER_COLUMN = 6

# Classified by the trimmed/lowercased ``howCome``.  Both the MATLAB names
# ('Permutate', ' Rotate ') and the names this Python port writes
# ('Permutation', 'Rotation', 'LatMutate', 'softmutate') are accepted.
ORIGINS = {
    "random": 1,
    "heredity": 2,
    "seeds": 4,
    "copex": 5,
}
MUTATION_ORIGINS = {
    "transmutate", "softmutate", "softmutation", "latmutate", "latmutation",
    "coormutate", "permutate", "permutation", "rotate", "rotation",
}


def _block_of(individual, ORG):
    num_blocks = getattr(individual, "numBlocks", None)
    if num_blocks is None:
        try:
            if int(getattr(ORG, "molecule", 0)) == 1:
                num_blocks = np.asarray(individual.numMols, dtype=float) / np.asarray(
                    ORG.numIons, dtype=float
                )
            else:
                num_blocks = np.asarray(individual.numIons, dtype=float) / np.asarray(
                    ORG.numIons, dtype=float
                )
        except Exception:
            num_blocks = np.ones(len(ORG.numIons), dtype=float)
        individual.numBlocks = num_blocks
    return np.asarray(num_blocks, dtype=float).reshape(-1)


def compositionStatistic(composition=None):
    ORG = G["ORG_STRUC"]
    USPEX = G.get("USPEX_STRUC")
    if USPEX is None:
        return None

    if composition is None:
        composition = getattr(ORG, "firstGeneSplitAll", None)
    if composition is None:
        composition = np.ones((1, len(ORG.numIons)), dtype=float)
    composition = np.asarray(composition, dtype=float)
    if composition.ndim == 1:
        composition = composition.reshape(1, -1)

    n_blocks = composition.shape[1]
    n_comp = composition.shape[0]
    statistic = np.zeros((n_comp, n_blocks * 2 + 7), dtype=float)
    statistic[:, :n_blocks] = composition
    for i in range(n_comp):
        total = float(np.sum(composition[i]))
        if total > 0:
            statistic[i, n_blocks:n_blocks * 2] = composition[i] / total

    population = getattr(USPEX, "POPULATION", []) or []
    for individual in population:
        if individual is None:
            continue
        num_blocks = _block_of(individual, ORG)
        if num_blocks.size != n_blocks:
            num_blocks = np.ones(n_blocks, dtype=float)

        which = None
        for i in range(n_comp):
            if np.max(np.abs(composition[i] - num_blocks)) < 1e-3:
                which = i
                break
        if which is None:
            n_comp += 1
            which = n_comp - 1
            composition = np.vstack([composition, num_blocks])
            total = float(np.sum(num_blocks))
            row = np.zeros(n_blocks * 2 + 7, dtype=float)
            row[:n_blocks] = num_blocks
            if total > 0:
                row[n_blocks:n_blocks * 2] = num_blocks / total
            statistic = np.vstack([statistic, row])

        statistic[which, n_blocks * 2] += 1
        how_come = str(getattr(individual, "howCome", "") or "").strip().lower()
        if how_come in ORIGINS:
            statistic[which, n_blocks * 2 + ORIGINS[how_come]] += 1
        elif how_come in MUTATION_ORIGINS:
            statistic[which, n_blocks * 2 + MUTATION_COLUMN] += 1
        else:
            statistic[which, n_blocks * 2 + OTHER_COLUMN] += 1

    USPEX.statistic = statistic
    ORG.firstGeneSplitAll = composition
    G["USPEX_STRUC"] = USPEX
    G["ORG_STRUC"] = ORG
    return statistic
