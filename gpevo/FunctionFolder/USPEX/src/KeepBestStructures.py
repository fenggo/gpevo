"""Python port of KeepBestStructures.m (elitist scheme)."""
from __future__ import annotations

import copy

import numpy as np

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.sys.fingerprint_utils import _cosine_distance as cosine_distance
from gpevo.log_utils import console_print


def _prepare(child, source, ORG, how_come, step):
    """Common bookkeeping of a carried-over structure (KeepBestStructures.m)."""
    child.howCome = how_come
    child.Step = step
    child.Done = 0
    child.ToDo = 1
    child.JobID = 0
    child.Folder = 0
    child.Number = 0

    enthalpy = float(np.asarray(source.Enthalpies, dtype=float).ravel()[-1])
    try:
        total_ions = int(sum(int(v) for v in source.numIons))
    except (TypeError, ValueError):
        total_ions = 0
    if total_ions > 0:
        enthalpy = enthalpy / total_ions
    child.Parents = [AttrDict(parent=str(getattr(source, "Number", "")), enthalpy=enthalpy)]
    return child


def _store(OFF, child):
    """Replace the trailing placeholder if there is one, else append."""
    if OFF.POPULATION and not getattr(OFF.POPULATION[-1], "numIons", None):
        OFF.POPULATION[-1] = child
    else:
        OFF.POPULATION.append(child)


def _distance(ORG, first, second):
    """Fingerprint distance used for the similarity checks."""
    fing_a = getattr(first, "FINGERPRINT", None)
    fing_b = getattr(second, "FINGERPRINT", None)
    if fing_a is None or fing_b is None:
        return None
    rows = np.asarray(fing_a).shape[0]
    weight = np.asarray(getattr(ORG, "weight", 1.0), dtype=float)
    if weight.ndim == 0 or weight.size != rows:
        weight = np.ones(rows, dtype=float)
    try:
        return float(cosine_distance(fing_a, fing_b, weight))
    except Exception:
        return None


def KeepBestStructures():
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    OFF = G["OFF_STRUC"]

    from gpevo.FunctionFolder.USPEX.src.variation_common import valid_ranking

    ranking = valid_ranking(POP)
    bad_rank = int(getattr(POP, "bad_rank", 0) or 0)
    fitness = np.asarray(getattr(POP, "fitness", None), dtype=float).reshape(-1)
    if fitness.size != len(POP.POPULATION):
        fitness = np.zeros(len(POP.POPULATION), dtype=float)

    n_steps = len(ORG.abinitioCode)
    generation = int(getattr(POP, "generation", 0))
    best_frac = float(getattr(ORG, "bestFrac", 0.5))
    dynamical = int(getattr(ORG, "dynamicalBestHM", 0))
    reopt_old = int(getattr(ORG, "reoptOld", 0))

    keep_hm = int(getattr(ORG, "keepBestHM", 0) or 0)
    decent_rank = int(min(round(len(ranking) * best_frac), len(ranking) - bad_rank))
    decent_rank = max(decent_rank, 0)
    if keep_hm > decent_rank:
        keep_hm = decent_rank

    n_best = max(1, int(round(best_frac * len(ranking)))) if ranking else 0
    top = [fitness[i] for i in ranking[:n_best]]
    first_mean = float(np.mean(top)) if top else 0.0
    spread = float(np.sqrt(np.var(top))) if top else 0.0

    hull = list(getattr(POP, "convex_hull", None) or [])
    addon_diff = 0

    def step_for():
        return max(n_steps - 1, 1) if reopt_old else n_steps + 1

    # ---- structures on the convex hull are always carried over ----------
    if hull:
        for row in hull:
            try:
                index = int(row[-1])
            except (TypeError, ValueError, IndexError):
                continue
            if index <= 0 or index > len(POP.POPULATION):
                continue
            source = POP.POPULATION[index - 1]
            child = _prepare(copy.deepcopy(source), source, ORG, "convexHull", step_for())
            _store(OFF, child)
            addon_diff += 1
        if int(getattr(ORG, "dimension", 3)) == 3:
            keep_hm = 0
        else:
            keep_hm = max(0, keep_hm - len(hull))

    # ---- softMutOnly generation: keep everything that survived ----------
    soft_mut_only = list(getattr(ORG, "softMutOnly", None) or [])
    if soft_mut_only and generation < len(soft_mut_only) and int(soft_mut_only[generation]) == 1:
        for position in range(len(ranking) - bad_rank):
            source = POP.POPULATION[ranking[position]]
            child = _prepare(copy.deepcopy(source), source, ORG, "keptBest", step_for())
            _store(OFF, child)
            addon_diff += 1
        console_print(f"{len(ranking) - bad_rank} structures were kept from the convex hull")
        G["OFF_STRUC"] = OFF
        return addon_diff

    # ---- normal case: keep up to ``keepBestHM`` distinct best structures -
    tolerance = float(getattr(ORG, "toleranceBestHM", 0.02))
    delta_tol = tolerance / 2.0
    if dynamical == 2:
        tolerance = 0.5
        delta_tol = tolerance / 2.0

    hull_ids = set()
    for row in hull:
        try:
            hull_ids.add(int(row[-1]) - 1)
        except (TypeError, ValueError, IndexError):
            continue

    chosen = [False] * len(ranking)
    kept = 0
    done = False
    while not done:
        chosen = [False] * len(ranking)
        kept = 0
        position = 0
        while kept < keep_hm:
            if position >= len(ranking):
                break
            if position >= decent_rank and dynamical > 0:
                break
            candidate = ranking[position]
            if candidate in hull_ids:
                position += 1
                continue

            good = True
            if int(getattr(ORG, "doFing", 0)) and generation > int(getattr(ORG, "doFing", 0)) - 1:
                candidate_individual = POP.POPULATION[candidate]
                # against the structures already stored from the hull
                for stored in OFF.POPULATION[-addon_diff:] if addon_diff else []:
                    dist = _distance(ORG, candidate_individual, stored)
                    if dist is not None and dist < tolerance:
                        good = False
                        break
                # against the structures already chosen in this pass
                if good:
                    for other, flag in enumerate(chosen):
                        if not flag:
                            continue
                        dist = _distance(ORG, candidate_individual, POP.POPULATION[ranking[other]])
                        if dist is not None and dist < tolerance:
                            good = False
                            break

            if (len(POP.POPULATION) - position) - (keep_hm - kept) <= 1 and dynamical == 0:
                good = True
            if dynamical and position > 0 and candidate < fitness.size:
                if float(fitness[candidate]) > first_mean + spread:
                    break

            if good:
                chosen[position] = True
                kept += 1
            position += 1

        if dynamical < 2:
            done = True
        else:
            if kept == keep_hm:
                done = True
            elif kept < keep_hm:
                tolerance -= delta_tol
            else:
                tolerance += delta_tol
            delta_tol /= 2.0
            if delta_tol < 0.000001:
                done = True

    for position, flag in enumerate(chosen):
        if not flag:
            continue
        source = POP.POPULATION[ranking[position]]
        child = _prepare(copy.deepcopy(source), source, ORG, "keptBest", step_for())
        _store(OFF, child)
        addon_diff += 1

    console_print(f"{kept} structures were kept as best")
    G["OFF_STRUC"] = OFF
    return addon_diff
