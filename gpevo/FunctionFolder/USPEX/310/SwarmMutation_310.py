"""Python port of a molecule-aware rigid-body swarm move (PSO-like).

This is *not* a port of ``uspex-gp``'s PSO module: that module places every
atom independently (``Random_PSO.m`` does ``candidate = rand(sum(numIons),3)``)
and mutates atoms along soft modes, which tears molecules apart.  Here the
swarm update is applied to whole molecules:

    x <- x + U(0, swarmStep) * (g - x)          per molecule centre
    orientation <- random rigid rotation        (<= swarmAngle)

where ``g`` is the matching molecule of the best structure of the generation.
Every operation is a rigid-body transform of ``MOLCOORS``, so intramolecular
geometry is preserved by construction -- the move only re-packs and
re-orients molecules, and the relaxation afterwards repairs the details.

Compared with a textbook PSO there is no velocity/momentum term: individuals
do not carry one, and adding it would mean new per-structure state that has
to survive pickling.  The random step U(0, swarmStep) supplies the stochastic
part instead.

``g`` lives in the best structure's cell, which need not match the child's
(heredity / latmutation mix cells).  ``_min_image`` therefore maps ``g`` into
the child's reference frame via fractional coordinates before folding the
displacement; otherwise the two cells are mixed (docs/swarm_operator.md 5.2 B).
Positional pairing of molecules is guarded by a ``MtypeLIST`` assertion so a
reordered molecule list can never silently cross molecule types (5.2 A).
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G
from gpevo.FunctionFolder.USPEX.src.variation_common import (
    copy_individual, finalize_offspring, pick_parent, recompute_zmatrix,
    valid_molecules, valid_ranking,
)
from gpevo.FunctionFolder.USPEX.src.Rotate_rigid_body import Rotate_rigid_body
from gpevo.log_utils import console_print


def _min_image(target, center, lattice, target_lattice=None):
    """Minimum-image displacement from ``center`` to ``target`` in ``lattice``'s frame.

    ``center`` is a Cartesian point in the child's cell (``lattice``);
    ``target`` is the matching point of the guiding ``best`` structure.
    When ``target_lattice`` is given, ``target`` is first expressed as
    fractional coordinates in that cell and mapped back into ``lattice``, so
    both endpoints share ONE reference frame.  Without it the displacement
    mixes the best and child cells (docs/swarm_operator.md 5.2 B) and can even
    reverse direction: a best molecule at fraction 0.8 of a=10 sits at 8 A,
    which folded with the a=12 child cell reads as +6 A instead of the
    correct -4.4 A toward 0.8*12 = 9.6 A.

    ``target_lattice=None`` keeps the legacy single-frame behaviour, i.e.
    ``target`` is already expressed in ``lattice``'s Cartesian frame.
    """
    cell = np.asarray(lattice, dtype=float).reshape(3, 3)
    tgt = np.asarray(target, dtype=float)
    ctr = np.asarray(center, dtype=float)
    try:
        inverse = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        return tgt - ctr
    if target_lattice is not None:
        source = np.asarray(target_lattice, dtype=float).reshape(3, 3)
        try:
            tgt = (tgt @ np.linalg.inv(source)) @ cell
        except np.linalg.LinAlgError:
            pass                      # singular source cell: keep Cartesian tgt
    frac = (tgt - ctr) @ inverse
    frac = frac - np.round(frac)      # half-even, matches newMolCheck
    return frac @ cell


def _centers(ind):
    return np.array([np.asarray(mol.MOLCOORS, dtype=float).mean(axis=0)
                     for mol in ind.MOLECULES], dtype=float)


def SwarmMutation_310(count):
    """Produce one offspring by moving the parent's molecules toward the best."""
    POP = G["POP_STRUC"]
    ORG = G["ORG_STRUC"]

    step = float(getattr(ORG, "swarmStep", 0.5) or 0.0)
    angle_max = float(getattr(ORG, "swarmAngle", 30.0) or 0.0) * np.pi / 180.0

    ranking = valid_ranking(POP)
    if not ranking or ranking[0] >= len(POP.POPULATION):
        console_print("failed to do Swarm mutation (no best structure), "
                      "switch to Random")
        from importlib import import_module as _im
        _im("gpevo.FunctionFolder.USPEX.310.Random_310").Random_310(count)
        return
    best = POP.POPULATION[ranking[0]]
    best_lattice = np.asarray(best.LATTICE, dtype=float)

    for attempt in range(1, 51):
        idx = pick_parent()
        if idx is None or idx >= len(POP.POPULATION):
            continue
        child = copy_individual(POP.POPULATION[idx])
        if len(child.MOLECULES) != len(best.MOLECULES):
            break                       # different composition -> no mapping
        child_list = getattr(child, "MtypeLIST", None)
        best_list = getattr(best, "MtypeLIST", None)
        child_types = tuple(int(t) for t in child_list) if child_list is not None else ()
        best_types = tuple(int(t) for t in best_list) if best_list is not None else ()
        assert child_types == best_types, (
            "SwarmMutation_310: parent and best MtypeLIST differ "
            f"({child_types} vs {best_types}); positional pairing would "
            "cross molecule types")

        lattice = np.asarray(child.LATTICE, dtype=float)
        targets = _centers(best)
        for position, mol in enumerate(child.MOLECULES):
            coords = np.asarray(mol.MOLCOORS, dtype=float)
            center = coords.mean(axis=0)
            delta = _min_image(targets[position], center, lattice, best_lattice)
            new_center = center + (step * np.random.rand()) * delta

            coords = coords + (new_center - center)          # rigid translation
            axis = np.random.randn(3)
            angle = (np.random.rand() - 0.5) * 2.0 * angle_max
            coords = Rotate_rigid_body(new_center, axis, coords, angle)

            mol.MOLCOORS = coords
            mol.MOLCENTER = coords.mean(axis=0)

        if not valid_molecules(child, -0.2):
            continue
        recompute_zmatrix(child)
        finalize_offspring(child, "Swarm")
        G["OFF_STRUC"].POPULATION.append(child)
        console_print(f"Structure {count}  generated by swarm mutation "
                      f"(attempt {attempt})")
        return

    console_print("failed to do Swarm mutation, switch to Random")
    from importlib import import_module as _im
    _im("gpevo.FunctionFolder.USPEX.310.Random_310").Random_310(count)
