"""Python equivalent of Permutation_310.m (molecular permutation operator).

In a molecular crystal a permutation exchanges the **positions** of two
molecules: each molecule is translated rigidly to where the other one stood.
Internal geometry, orientation and the molecule type of every molecule are
kept, so ``MOLECULES``/``MtypeLIST`` stay consistent with the standard
molecule each of them belongs to (``Permutation_310.m:60-88`` and
``swapIons_mutation_final.m``).

Swapping whole ``MOLECULES`` entries instead (what an earlier version did)
reordered the coordinate blocks while ``MtypeLIST`` was rebuilt from
``numMols``: for two molecules of different sizes the blocks then no longer
matched their template, which produced torn molecules.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G
from gpevo.FunctionFolder.USPEX.src.variation_common import (
    build_tournament, copy_individual, finalize_offspring, pick_parent,
    recompute_zmatrix, valid_molecules,
)
from gpevo.log_utils import console_print


def _order_pick(indices, molecules):
    """Pick one molecule out of ``indices``, biased towards low ``order``.

    ``swapIons_mutation_final.m`` sorts the candidates by their ``order`` and
    then draws with a tournament array, which favours the first (lowest order)
    entry.
    """
    if len(indices) == 1:
        return indices[0]

    values = []
    for i in indices:
        order = getattr(molecules[i], "order", None)
        if order is None:
            values.append(np.inf)
        else:
            array = np.asarray(order, dtype=float)
            values.append(float(np.mean(array)) if array.size else np.inf)
    ranks = sorted(range(len(indices)), key=lambda k: values[k])

    table = np.asarray(build_tournament(len(indices)), dtype=float)
    draw = float(np.random.uniform(0.0, float(np.max(table))))
    above = np.nonzero(table > draw)[0]
    rank = int(above[-1]) if above.size else int(table.size - 1)
    return int(indices[ranks[rank]])


def _pick_pair(molecules, mtype_list):
    """Choose the two molecules whose positions are exchanged."""
    groups = {}
    for i, mtype in enumerate(mtype_list):
        groups.setdefault(int(mtype), []).append(i)

    keys = list(groups)
    if len(keys) >= 2:
        # swapIons_mutation_final picks one molecule out of each of two groups.
        a, b = int(np.random.randint(len(keys))), int(np.random.randint(len(keys) - 1))
        if b >= a:
            b += 1
        return _order_pick(groups[keys[a]], molecules), _order_pick(groups[keys[b]], molecules)

    candidates = list(range(len(molecules)))
    first = _order_pick(candidates, molecules)
    others = [i for i in candidates if i != first]
    if not others:
        return first, first
    return first, _order_pick(others, molecules)


def Permutation_310(count):
    _ = count
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]

    for _ in range(50):
        idx = pick_parent()
        if idx is None or idx >= len(POP.POPULATION):
            continue
        child = copy_individual(POP.POPULATION[idx])
        molecules = child.MOLECULES
        if len(molecules) < 2:
            continue

        # Breathing lattice (Permutation_310.m:39-47): the cell is expanded to
        # make room for the exchange and every molecule moves outward with it.
        # Only the molecular centres are scaled, so the internal geometry is
        # left untouched -- scaling MOLCOORS itself would stretch the bonds of
        # every molecule by 5-25%, which a reactive force field then relaxes
        # (or breaks) instead of just re-packing the structure.
        breath_factor = 1.05 + np.random.random() * 0.20
        lattice = np.asarray(child.LATTICE, dtype=float) * breath_factor
        child.LATTICE = lattice
        for mol in molecules:
            coords = np.asarray(mol.MOLCOORS, dtype=float)
            center = np.mean(coords, axis=0)
            mol.MOLCOORS = coords + (breath_factor - 1.0) * center

        n_swaps = int(getattr(ORG, "howManySwaps", 0) or 0)
        n_swaps = min(max(1, n_swaps), len(molecules) // 2)

        centers = [np.mean(np.asarray(mol.MOLCOORS, dtype=float), axis=0) for mol in molecules]
        swapped = 0
        for _ in range(n_swaps):
            i, j = _pick_pair(molecules, child.MtypeLIST)
            if i == j or i >= len(centers) or j >= len(centers):
                continue
            centers[i], centers[j] = centers[j], centers[i]
            swapped += 1
        if not swapped:
            continue

        # Rigid translation: every molecule keeps its geometry and orientation
        # (Permutation_310.m:69-78).  The new centre is wrapped back into the
        # cell, which is a pure lattice translation of the same structure.
        inv = np.linalg.inv(lattice)
        for k, mol in enumerate(molecules):
            coords = np.asarray(mol.MOLCOORS, dtype=float)
            old_center = np.mean(coords, axis=0)
            frac = centers[k] @ inv
            frac = frac - np.floor(frac)
            target = frac @ lattice
            mol.MOLCOORS = coords + (target - old_center)
            mol.MOLCENTER = target

        if not valid_molecules(child, 0):
            continue
        recompute_zmatrix(child)
        finalize_offspring(child, "Permutation")
        G["OFF_STRUC"].POPULATION.append(child)
        console_print(f"Structure {count}  generated by permutation")
        return

    console_print("failed to do permutation in 50 attempts, switch to Random")
    from importlib import import_module as _im
    _im("gpevo.FunctionFolder.USPEX.310.Random_310").Random_310(count)
