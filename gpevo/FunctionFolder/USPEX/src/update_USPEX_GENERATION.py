"""Python port of update_USPEX_GENERATION.m.

Writes the per-generation summary into ``USPEX_STRUC.GENERATION`` and pushes
the fitness values into the cumulative structure record.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G
from gpevo.FunctionFolder.USPEX.src.oldQuasiEntropy import oldQuasiEntropy
from gpevo.FunctionFolder.USPEX.src.composEntropy import composEntropy


def update_USPEX_GENERATION(IND, fitness, type=1):
    POP = G["POP_STRUC"]
    USPEX = G.get("USPEX_STRUC")
    if USPEX is None:
        from gpevo.runtime import AttrDict

        USPEX = AttrDict(POPULATION=[], GENERATION=[])
        G["USPEX_STRUC"] = USPEX

    generations = list(getattr(USPEX, "GENERATION", []) or [])
    generation = int(getattr(POP, "generation", 0))
    while len(generations) <= generation:
        generations.append(AttrDict_empty())
    entry = generations[generation]

    fitness = np.asarray(fitness, dtype=float).reshape(-1)
    entry.quasiEntropy = oldQuasiEntropy()
    entry.Fitness = float(np.min(fitness)) if fitness.size else 0.0
    if int(type) == 2:
        entry.convex_hull = getattr(POP, "convex_hull", [])
        entry.composEntropy = composEntropy()

    best_ids = []
    for index in list(IND or []):
        index = int(index)
        if index > 0 and index <= len(POP.POPULATION):
            best_ids.append(int(getattr(POP.POPULATION[index - 1], "Number", 0)))
    entry.BestID = best_ids
    entry.ID = len(getattr(USPEX, "POPULATION", []) or [])

    # fitness of every structure of this generation, in completion order
    done_order = np.asarray(getattr(POP, "DoneOrder", None), dtype=float).reshape(-1)
    population = getattr(USPEX, "POPULATION", []) or []
    if done_order.size == len(POP.POPULATION):
        order = list(np.argsort(done_order))
    else:
        order = list(range(len(POP.POPULATION)))
    for position, individual_index in enumerate(order):
        if position >= fitness.size:
            break
        done_value = done_order[individual_index] if individual_index < done_order.size else 1
        if done_value > 0:
            number = int(getattr(POP.POPULATION[individual_index], "Number", 0) or 0)
            if 0 < number <= len(population) and population[number - 1] is not None:
                population[number - 1].Fitness = float(fitness[position])

    USPEX.GENERATION = generations
    G["USPEX_STRUC"] = USPEX
    return USPEX


def AttrDict_empty():
    from gpevo.runtime import AttrDict

    return AttrDict(quasiEntropy=0.0, Fitness=0.0, BestID=[], ID=0)
