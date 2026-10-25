"""Python port of update_STUFF.m.

Rebuilds the operator fractions (static / AutoFrac / RL bandit), the
``ORG.tournament`` table and the ``softMutOnly`` schedule for the coming
generation; called from EA_310 right after the generation backup is written.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G, safesave
from gpevo.FunctionFolder.USPEX.src.createORG_Fingerprint import createORG_Fingerprint
from gpevo.FunctionFolder.USPEX.src.createORG_EA import createORG_EA
from gpevo.FunctionFolder.USPEX.src.createORG_Symmetry import createORG_Symmetry
from gpevo.FunctionFolder.USPEX.src.createORG_Onthefly import createORG_Onthefly
from gpevo.FunctionFolder.USPEX.src.update_STUFF_old import update_STUFF_old
from gpevo.FunctionFolder.USPEX.src.variation_common import build_tournament
from gpevo.log_utils import console_print

# AutoFrac operator names, in the order used by update_STUFF.m.
PARENTS = ["Random", "Heredity", "Permutate", "TransMutate", "LatMutate", "softmutate", "Rotate"]
FRACTIONS = ["fracRand", "fracGene", "fracPerm", "fracTrans", "fracLatMut", "fracAtomsMut", "fracRotMut"]
HOW_MANY = [
    "howManyRand", "howManyOffsprings", "howManyPermutations", "howManyTrans",
    "howManyMutations", "howManyAtomMutations", "howManyRotations",
]

# RL bandit arms -> howMany* fields (RL_map_n in update_STUFF.m).
RL_HOW_MANY = [
    "howManyOffsprings", "howManyRand", "howManyPermutations",
    "howManyRotations", "howManyMutations", "howManyAtomMutations",
]
RL_FRACTIONS = ["fracGene", "fracRand", "fracPerm", "fracRotMut", "fracLatMut", "fracAtomsMut"]


def _how_many_proliferate(ORG, POP):
    ranking = list(getattr(POP, "ranking", None) or range(len(POP.POPULATION)))
    n_pro = int(round(len(ranking) * float(getattr(ORG, "bestFrac", 0.5))))
    if n_pro == 0:
        n_pro = 1
    return int(min(n_pro, len(ranking) - int(getattr(POP, "bad_rank", 0) or 0)))


def _auto_frac(ORG, POP, how_many_proliferate):
    """AutoFrac branch: reward the operators that produced surviving offspring.

    Needs ``USPEX_STRUC.GENERATION`` history; when that is not maintained the
    caller falls back to the static fractions.
    """
    USPEX = G.get("USPEX_STRUC")
    generation = int(getattr(POP, "generation", 0))
    if USPEX is None or generation <= 1:
        return None
    generations = list(getattr(USPEX, "GENERATION", []) or [])
    if len(generations) < generation:
        return None
    # ``ID`` is the number of structures recorded up to that generation.
    prev1 = int(generations[generation - 1].get("ID", 0) or 0)
    prev2 = int(generations[generation - 2].get("ID", 0) or 0) if generation >= 2 else 0
    history = list(getattr(USPEX, "POPULATION", []) or [])

    # 3D compares with the order-aware variant (update_STUFF.m:62-66).
    try:
        if int(getattr(ORG, "dimension", 3)) == 3:
            from gpevo.FunctionFolder.sys.SameStructure_order import (
                SameStructure_order as SameStructure,
            )
        else:
            from gpevo.FunctionFolder.sys.SameStructure import SameStructure
    except Exception:
        SameStructure = None

    ranking = list(getattr(POP, "ranking", None) or range(len(POP.POPULATION)))
    n_ops = len(PARENTS)
    accepted = np.zeros(n_ops, dtype=float)
    produced = np.zeros(n_ops, dtype=float)

    for position in range(min(how_many_proliferate, len(ranking))):
        individual = POP.POPULATION[ranking[position]]
        accept = True
        # discard structures that already appeared in the previous generations
        if SameStructure is not None and history and prev1 > prev2:
            holder = USPEX
            for index in range(prev2, prev1):
                record = history[index] if index < len(history) else None
                if record is None:
                    continue
                try:
                    if SameStructure(index, int(getattr(individual, "Number", 0) or 0) - 1, holder):
                        accept = False
                        break
                except Exception:
                    continue
        if not accept:
            continue
        how_come = str(getattr(individual, "howCome", "") or "")
        for index, key in enumerate(PARENTS):
            if key in how_come:
                accepted[index] += 1
                break

    for position in range(len(ranking)):
        how_come = str(getattr(POP.POPULATION[ranking[position]], "howCome", "") or "")
        for index, key in enumerate(PARENTS):
            if key in how_come:
                produced[index] += 1
                break

    quality = np.zeros(n_ops, dtype=float)
    total = 0.0
    count = 0
    for index in range(n_ops):
        if produced[index] > 0:
            accepted[index] = max(1.0, accepted[index])
            quality[index] = accepted[index] / produced[index]
            total += quality[index]
            count += 1
    if count == 0:
        return None
    mean_quality = total / count
    for index in range(n_ops):
        if produced[index] > 0:
            quality[index] = (quality[index] / mean_quality + 1) / 2

    fractions = np.array([float(getattr(ORG, name, 0.0) or 0.0) for name in FRACTIONS])
    for index in range(n_ops):
        if produced[index] > 0:
            fractions[index] = 0.55 * fractions[index] + 0.45 * quality[index]
    fractions[0] = max(fractions[0], 0.10)   # random
    fractions[1] = max(fractions[1], 0.10)   # heredity
    fractions[5] = max(fractions[5], 0.10)   # softmutate
    return fractions / fractions.sum()


def update_STUFF(inputFile, goodFrac=None, ranking=None):
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]

    createORG_Fingerprint(inputFile)
    createORG_EA(inputFile)
    createORG_Symmetry(inputFile)
    createORG_Onthefly(inputFile)

    how_many_proliferate = _how_many_proliferate(ORG, POP)

    if int(getattr(ORG, "RL_strategy", 0)) >= 1:
        try:
            from gpevo.FunctionFolder.USPEX.src.RL_agent import RL_agent

            rl_fractions = np.asarray(RL_agent(int(ORG.populationSize)), dtype=float)
            rl_fractions = rl_fractions / rl_fractions.sum()
            for index, name in enumerate(RL_FRACTIONS):
                setattr(ORG, name, float(rl_fractions[index]))
            counts = [int(round(int(ORG.populationSize) * float(v))) for v in rl_fractions]
            difference = int(ORG.populationSize) - int(sum(counts))
            if difference and counts:
                largest = int(np.argmax(rl_fractions))
                counts[largest] += difference
            for index, name in enumerate(RL_HOW_MANY):
                setattr(ORG, name, int(counts[index]))
            ORG.fracTrans = 0.0
            ORG.howManyTrans = 0
        except Exception as exc:
            console_print(f"RL bandit disabled: {exc}")
            update_STUFF_old(inputFile, goodFrac, ranking)
    elif not int(getattr(ORG, "AutoFrac", 0)):
        update_STUFF_old(inputFile, goodFrac, ranking)
    else:
        fractions = _auto_frac(ORG, POP, how_many_proliferate)
        if fractions is None:
            update_STUFF_old(inputFile, goodFrac, ranking)
        else:
            population = int(ORG.populationSize)
            for index, name in enumerate(FRACTIONS):
                setattr(ORG, name, float(fractions[index]))
            for index, name in enumerate(HOW_MANY):
                setattr(ORG, name, int(round(population * float(fractions[index]))))

    # ---- tournament table (update_STUFF.m:128-140) ----------------------
    how_many_proliferate = _how_many_proliferate(ORG, POP)
    ORG.tournament = build_tournament(how_many_proliferate)
    if int(getattr(POP, "generation", 1)) == 0:
        initial = int(getattr(ORG, "initialPopSize", 0) or 0)
        if how_many_proliferate > initial:
            ORG.tournament = build_tournament(max(1, initial))

    # ---- softMutOnly schedule -------------------------------------------
    soft = list(getattr(ORG, "softMutOnly", None) or [])
    needed = int(getattr(ORG, "numGenerations", 0)) + 1
    if len(soft) < needed:
        ORG.softMutOnly = [0] * needed

    G["ORG_STRUC"] = ORG
    safesave("Current_ORG.pkl", ORG)
    return ORG
