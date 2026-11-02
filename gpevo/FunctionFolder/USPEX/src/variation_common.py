"""Shared helpers for USPEX 310 variation operators."""
from __future__ import annotations

import copy
import numpy as np

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.USPEX.src.newMolCheck import newMolCheck


def copy_individual(ind):
    return copy.deepcopy(ind)


def recompute_zmatrix(ind):
    """Refresh ``mol.ZMATRIX`` from ``MOLCOORS`` after a variation move.

    MATLAB's 310 operators recompute the Z-matrix of every molecule on the
    successful offspring (e.g. ``LatMutation_310.m:42``,
    ``Mutation_310.m:53``, ``Permutation_310.m:86``).  Leaving the parent's
    ZMATRIX on the child makes it stale: ``move_all_mol_Mutation`` reads
    ``ZMATRIX(1,:)`` as the molecule centre, so a stale value feeds wrong
    displacements into the soft-mode / coordinate-mutation fallbacks.

    ``ind`` needs ``MOLECULES`` and ``MtypeLIST``; ``ORG.STDMOL`` supplies each
    molecule's ``format``.  Molecules whose template is unavailable keep a
    coordinate-based fallback so ``ZMATRIX[0]`` is at least current.
    """
    ORG = G.get("ORG_STRUC")
    stdmol = getattr(ORG, "STDMOL", None) if ORG is not None else None
    mt_attr = getattr(ind, "MtypeLIST", None)
    mtype_list = list(mt_attr) if mt_attr is not None else []
    mols_attr = getattr(ind, "MOLECULES", None)
    molecules = list(mols_attr) if mols_attr is not None else []
    if not molecules:
        return ind
    from gpevo.FunctionFolder.USPEX.src.NEW_coord2Zmatrix import NEW_coord2Zmatrix

    for j, mol in enumerate(molecules):
        coords = np.asarray(mol.MOLCOORS, dtype=float)
        fmt = None
        if stdmol is not None and len(stdmol) and j < len(mtype_list):
            mt = int(mtype_list[j])
            if 1 <= mt <= len(stdmol):
                fmt = getattr(stdmol[mt - 1], "format", None)
        if fmt is None:
            mol.ZMATRIX = coords.copy()
            continue
        try:
            mol.ZMATRIX = np.real(NEW_coord2Zmatrix(coords, fmt))
        except Exception:
            mol.ZMATRIX = coords.copy()
    return ind


def append_off(ind):
    OFF = G["OFF_STRUC"]
    OFF.POPULATION.append(ind)


def recompute_types(ind):
    ORG = G["ORG_STRUC"]
    from gpevo.FunctionFolder.USPEX.src.GetPOP_MOL import GetPOP_MOL
    types_alist, mtype_list, num_ions = GetPOP_MOL(ind.numMols)
    ind.MtypeLIST = mtype_list
    ind.typesAList = types_alist
    ind.numIons = num_ions


def valid_ranking(POP):
    """``POP.ranking`` restricted to indices that exist in the population.

    The ranking is produced by ``FitnessRankingCorrection`` for the previous
    population, so it can be stale (and out of range) after the next generation
    has been assembled.
    """
    ranking = list(getattr(POP, "ranking", None) or range(len(POP.POPULATION)))
    size = len(POP.POPULATION)
    filtered = [int(i) for i in ranking if 0 <= int(i) < size]
    return filtered or list(range(size))


def build_tournament(count):
    """Build ``ORG.tournament`` exactly as update_STUFF.m does.

    Rank position ``p`` (0-based, best first) carries weight ``(N-p)^2``; the
    array is stored as the cumulative sum so that drawing a uniform number in
    ``[0, max(tournament)]`` and keeping the last entry above it reproduces
    ``find(ORG.tournament > r)`` in MATLAB.
    """
    count = int(max(0, count))
    table = [0.0] * count
    if count:
        table[count - 1] = 1.0
    for loop in range(2, count + 1):
        table[count - loop] = table[count - loop + 1] + loop ** 2
    return table


def update_tournament():
    """Recompute ``ORG.tournament`` for the current ranking (update_STUFF.m)."""
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    ranking = valid_ranking(POP)
    best_frac = float(getattr(ORG, "bestFrac", 0.5))
    n_pro = int(round(len(ranking) * best_frac)) or 1
    how_many = int(min(n_pro, len(ranking) - int(getattr(POP, "bad_rank", 0) or 0)))
    how_many = max(how_many, 1) if ranking else 0
    ORG.tournament = build_tournament(how_many)

    if int(getattr(POP, "generation", 1)) == 0:
        initial = int(getattr(ORG, "initialPopSize", 0) or 0)
        if how_many > initial:
            ORG.tournament = build_tournament(max(1, initial))
    G["ORG_STRUC"] = ORG
    return ORG.tournament


def pick_parent():
    """Tournament selection of one parent (Heredity_310.m:22-32)."""
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    ranking = valid_ranking(POP)
    if not ranking:
        return None

    tournament = np.asarray(getattr(ORG, "tournament", None), dtype=float).reshape(-1)
    if tournament.size == 0:
        tournament = np.asarray(update_tournament(), dtype=float).reshape(-1)
    if tournament.size == 0:
        return int(ranking[int(np.random.randint(0, len(ranking)))])

    draw = float(np.random.uniform(0.0, float(np.max(tournament))))
    above = np.nonzero(tournament > draw)[0]
    position = int(above[-1]) if above.size else int(tournament.size - 1)
    position = min(position, len(ranking) - 1)
    return int(ranking[position])


def valid_molecules(ind, threshold_adjustment=0):
    """Run ``newMolCheck`` with ``minDistMatrice + threshold_adjustment``.

    The original 310 operators use different tolerances around the base
    ``minDistMatrice``, e.g. ``-0.2`` for heredity/rotation and ``-0.1`` for
    soft-mode mutation.
    """
    ORG = G["ORG_STRUC"]
    try:
        matrix = np.asarray(ORG.minDistMatrice, dtype=float)
        if threshold_adjustment:
            matrix = matrix + float(threshold_adjustment)
        return newMolCheck(ind.MOLECULES, ind.LATTICE, ind.MtypeLIST, matrix)
    except Exception:
        return 0


def finalize_offspring(ind, how_come):
    ORG = G["ORG_STRUC"]
    ind.howCome = how_come
    ind.Done = 0
    ind.ToDo = 1
    ind.JobID = 0
    ind.Folder = 0
    ind.Number = 0
    ind.Step = 1
    ind.Enthalpies = np.array([100000.0] * len(ORG.abinitioCode))
    ind.K_POINTS = np.ones((len(ORG.abinitioCode), 3))
    recompute_types(ind)
    return ind
