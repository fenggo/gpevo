"""Python equivalent of Rotation_310.m (molecular rigid-body rotation operator).

Two moves are applied per molecule, matching the MATLAB ``SmartRotInertia``
behaviour:

1. **Principal-axis rigid rotation** -- the molecule is rotated about each of its
   three principal inertia axes by a random angle whose amplitude is scaled by
   the ratio ``Ref / b_loop`` (``Ref`` = smallest non-zero principal moment).
   This makes the lightest axis rotate by up to ``rotationAngleMax`` (the MATLAB
   default ``pi/2`` => 90 deg), instead of the previous fixed ``angle_max/3``.
2. **Flexible intramolecular dihedral twist** (only when the molecule template
   declares flexible dihedrals, ``STDMOL[k].flex_dihedral``) -- the substituent
   on one side of a rotatable bond is twisted around that bond by a progressive
   random angle (+-pi / +-pi/3 / +-pi/6), accepted only if the intramolecular
   connectivity is preserved. This is the molecule-internal "group rotation"
   (e.g. a nitro group about its C-N bond) and is functionally identical to the
   MATLAB Z-matrix dihedral edit, but done in Cartesian space so it reuses the
   existing ``Rotate_rigid_body`` / ``intramol_bonds`` helpers.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G
from gpevo.FunctionFolder.USPEX.src.variation_common import (
    copy_individual, finalize_offspring, pick_parent, recompute_zmatrix, valid_molecules,
)
from gpevo.FunctionFolder.USPEX.src.Rotate_rigid_body import Rotate_rigid_body
from gpevo.FunctionFolder.USPEX.src.PrincipleAxis import PrincipleAxis
from gpevo.FunctionFolder.sys.mol_bond_table import intramol_bonds
from gpevo.log_utils import console_print


# Successful intramolecular group (rotor) twists, tallied once per offspring
# structure that carries at least one successful twist.  EA_310 resets the
# per-generation counter before each round of variation operators.
_group_rot_gen = 0
_group_rot_total = 0


def reset_group_rotation_counter():
    """Reset the per-generation group-rotation tally (EA_310 calls this)."""
    global _group_rot_gen
    _group_rot_gen = 0


def group_rotation_counts():
    """Return ``(this generation, cumulative over the whole run)``."""
    return _group_rot_gen, _group_rot_total


def _bond_graph(coords, type_idxs):
    """Adjacency list of a molecule from its intramolecular bonds."""
    bonds = intramol_bonds(coords, type_idxs)
    n = coords.shape[0]
    adj = [set() for _ in range(n)]
    for i, j in bonds:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _twist_dihedrals(coords, flex, type_idxs, ref_bonds, max_iter=200):
    """Twist each flexible dihedral, keeping the intramolecular connectivity.

    ``flex`` is a list of ``(i1, i2, outer)`` tuples: the rotatable bond is
    ``(i1, i2)`` and ``outer`` (with its connected subtree) is the part that
    moves.  Returns ``(coords, moved)`` where ``moved`` is True when at least
    one rotatable group was actually twisted (a real "group rotation").
    """
    coords = np.asarray(coords, dtype=float).copy()
    n = coords.shape[0]
    if n < 4 or not flex:
        return coords, False
    ref = set(tuple(sorted(p)) for p in (ref_bonds or intramol_bonds(coords, type_idxs)))
    any_moved = False
    for i1, i2, outer in flex:
        if not (0 <= i1 < n and 0 <= i2 < n and 0 <= outer < n):
            continue
        adj = _bond_graph(coords, type_idxs)
        adj[i1].discard(i2)
        adj[i2].discard(i1)
        comp = set()
        stack = [outer]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(adj[x] - comp)
        if not comp:
            continue
        comp_idx = list(comp)
        original = coords.copy()
        moved = False
        for it in range(max_iter):
            if it < max_iter / 3:
                step = np.pi * 0.5          # early: +-90 deg
            elif it < 2 * max_iter / 3:
                step = np.pi / 3      # mid:   +-60 deg
            else:
                step = np.pi / 6      # late:  +-30 deg
            angle = (np.random.rand() - 0.5) * 2 * step
            axis = coords[i2] - coords[i1]
            cand = coords.copy()
            cand[comp_idx] = Rotate_rigid_body(
                coords[i1], axis, cand[comp_idx], angle
            )
            if set(tuple(sorted(p)) for p in intramol_bonds(cand, type_idxs)) == ref:
                coords = cand
                moved = True
                break
        if not moved:
            # MATLAB keeps only the principal-axis rotation on failure.
            coords = original
        else:
            any_moved = True
    return coords, any_moved


def Rotation_310(count):
    global _group_rot_gen, _group_rot_total
    _ = count
    POP = G["POP_STRUC"]
    angle_max = float(getattr(G["ORG_STRUC"], "rotationAngleMax", np.pi / 2))
    trans_max = float(getattr(G["ORG_STRUC"], "translationMax", 0.5))
    max_dih = int(getattr(G["ORG_STRUC"], "maxDihedralIter", 200) or 200)
    ORG = G["ORG_STRUC"]
    std_mols = getattr(ORG, "STDMOL", None)
    ref_bonds = getattr(ORG, "molRefBonds", None)
    for attempt in range(1, 51):
        idx = pick_parent()
        if idx is None or idx >= len(POP.POPULATION):
            continue
        child = copy_individual(POP.POPULATION[idx])
        # Breathing lattice (Rotation_310.m:38-45): the cell is expanded and the
        # molecules move outward with it.  Only the molecular centres are
        # scaled -- scaling MOLCOORS would stretch every intramolecular bond by
        # 10-20%, which a reactive force field then relaxes (or breaks) instead
        # of just re-packing the structure.
        breath_factor = 1.10 + np.random.rand() * 0.10
        breath_lattice = np.asarray(child.LATTICE, dtype=float) * breath_factor
        child.LATTICE = breath_lattice
        for mol in child.MOLECULES:
            coords = np.asarray(mol.MOLCOORS, dtype=float)
            center = np.mean(coords, axis=0)
            mol.MOLCOORS = coords + (breath_factor - 1.0) * center
        group_rotated = False
        for mi, mol in enumerate(child.MOLECULES):
            coords = np.asarray(mol.MOLCOORS, dtype=float)
            center = np.mean(coords, axis=0)
            # --- principal-axis rigid rotation (SmartRotInertia) ---
            a, b = PrincipleAxis(coords - center)
            moments = np.sort(np.abs(b))
            ref = moments[0] if moments[0] > 1e-4 else (
                moments[1] if moments.size > 1 else 1.0)
            for loop in range(3):
                if abs(b[loop]) <= 1e-4:
                    continue
                angle = (np.random.rand() - 0.5) * 2 * angle_max * ref / b[loop]
                coords = Rotate_rigid_body(center, a[:, loop], coords, angle)
            coords = coords + (np.random.rand(3) - 0.5) * 2 * trans_max
            # --- flexible intramolecular dihedral twist (flex_dihedral) ---
            if std_mols is not None and mi < len(std_mols):
                std = std_mols[mi]
                flex = getattr(std, "flex_dihedral", None)
                if flex:
                    types = [int(t) for t in std.types]
                    rb = ref_bonds[mi] if (ref_bonds is not None and mi < len(ref_bonds)) else None
                    coords, moved = _twist_dihedrals(coords, flex, types, rb, max_dih)
                    if moved:
                        group_rotated = True
            mol.MOLCOORS = coords
            mol.MOLCENTER = np.mean(coords, axis=0)
        if not valid_molecules(child, -0.2):
            continue
        recompute_zmatrix(child)
        finalize_offspring(child, "Rotation")
        G["OFF_STRUC"].POPULATION.append(child)
        if group_rotated:
            _group_rot_gen += 1
            _group_rot_total += 1
        console_print(f"Structure {count}  generated by rotmutation"
                      f"{' [group rotation]' if group_rotated else ''}")
        return
    console_print("failed to do Rotation, switch to Random")
    from importlib import import_module as _im
    _im("gpevo.FunctionFolder.USPEX.310.Random_310").Random_310(count)
