"""Python port of a molecule-aware rigid-body swarm move (PSO-like).

This is *not* a port of ``uspex-gp``'s PSO module: that module places every
atom independently (``Random_PSO.m`` does ``candidate = rand(sum(numIons),3)``)
and mutates atoms along soft modes, which tears molecules apart.  Here the
swarm update is applied to whole molecules:

    v <- w*v + c1*U(0,1)*(pbest - x) + U(0, swarmStep)*(g - x)
    x <- x + v                                  per molecule centre
    R <- inertia(R_prev) * pbest(R) * slerp(I, R(parent -> g), t)
                                                per molecule orientation

where ``g`` is the matching molecule of the best structure of the generation,
``R(parent -> g)`` is the best-fit rigid rotation between the two centred
molecules (Kabsch), and ``t = U(0, 1)`` is capped so the applied rotation never
exceeds ``swarmAngle``.
Every operation is a rigid-body transform of ``MOLCOORS``, so intramolecular
geometry is preserved by construction -- the move only re-packs and
re-orients molecules, and the relaxation afterwards repairs the details.

The velocity v, the last rotation R_prev and the personal best live on the
individual as ind.swarm_state (plain numpy arrays / floats, so picklable and
deepcopy-safe; see src/swarm_state.py).  pbest is refreshed once per generation
from the evaluated fitness, and the state travels with the lineage through
Current_POP.pkl restarts.  w = swarmInertia and c1 = swarmCognitive default to
0, which reduces the move to the pure social term; set them > 0 to enable
momentum / personal best.  The random U(0, step) still supplies the social
stochastic part.

``g`` lives in the best structure's cell, which need not match the child's
(heredity / latmutation mix cells).  ``_min_image`` therefore maps ``g`` into
the child's reference frame via fractional coordinates before folding the
displacement; otherwise the two cells are mixed (docs/swarm_operator.md 5.2 B).
Orientations need no such mapping: internal bond vectors are physical vectors
in the shared Cartesian basis, not lattice-relative positions.  Mapping them
through the cell affine transform would deform bond lengths, so a Kabsch
best-fit pure rotation is used instead (5.2 C).
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
from gpevo.FunctionFolder.USPEX.src.swarm_state import ensure_swarm_state
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


_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def _kabsch_rotation(P, Q):
    """Best-fit proper rotation R with P @ R ~= Q (row vectors).

    P and Q are centred (n, 3) arrays of the same molecule type, so the
    optimal map is a pure rotation: the sign of det(U @ Vt) is folded into
    the last singular direction to forbid reflections.
    """
    H = np.asarray(Q, dtype=float).T @ np.asarray(P, dtype=float)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _quat_from_matrix(rotation):
    """Row-vector rotation matrix -> unit quaternion (w, x, y, z)."""
    M = np.asarray(rotation, dtype=float).reshape(3, 3).T   # column form
    trace = np.trace(M)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (M[2, 1] - M[1, 2]) / s,
                      (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s])
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2.0
        q = np.array([(M[2, 1] - M[1, 2]) / s, 0.25 * s,
                      (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s])
    elif M[1, 1] > M[2, 2]:
        s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2.0
        q = np.array([(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s,
                      0.25 * s, (M[1, 2] + M[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2.0
        q = np.array([(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s,
                      (M[1, 2] + M[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def _quat_slerp(q0, q1, t):
    """Shortest-path unit-quaternion interpolation (slerp).

    q and -q encode the same rotation; flipping the sign keeps the arc at or
    below 180 degrees.  A degenerate arc (sin(theta) ~ 0) returns q0.
    """
    q0 = np.asarray(q0, dtype=float)
    q0 = q0 / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=float)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    if sin_theta < 1e-12:
        return q0
    return (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / sin_theta


def _quat_to_matrix(q):
    """Unit quaternion -> row-vector rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    column = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return column.T                    # row-vector form used by the 310 operators


def SwarmMutation_310(count):
    """Produce one offspring by moving the parent's molecules toward the best."""
    POP = G["POP_STRUC"]
    ORG = G["ORG_STRUC"]

    step = float(getattr(ORG, "swarmStep", 0.5) or 0.0)
    angle_max = float(getattr(ORG, "swarmAngle", 30.0) or 0.0) * np.pi / 180.0
    inertia = float(getattr(ORG, "swarmInertia", 0.0) or 0.0)
    cognitive = float(getattr(ORG, "swarmCognitive", 0.0) or 0.0)

    ranking = valid_ranking(POP)
    if not ranking or ranking[0] >= len(POP.POPULATION):
        console_print("failed to do Swarm mutation (no best structure), "
                      "switch to Random")
        from importlib import import_module as _im
        _im("gpevo.FunctionFolder.USPEX.310.Random_310").Random_310(count)
        return
    best = POP.POPULATION[ranking[0]]
    best_lattice = np.asarray(best.LATTICE, dtype=float).reshape(3, 3)

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

        lattice = np.asarray(child.LATTICE, dtype=float).reshape(3, 3)
        targets = _centers(best)
        state = ensure_swarm_state(child)
        for position, mol in enumerate(child.MOLECULES):
            coords = np.asarray(mol.MOLCOORS, dtype=float)
            center = coords.mean(axis=0)

            # Translational PSO: inertia + cognitive (pbest) + social (g).
            # pbest centres are stored fractionally, so map them into this
            # child's cell before taking the min image.
            d_g = _min_image(targets[position], center, lattice, best_lattice)
            p_cart = state.pbest_frac[position] @ lattice
            d_p = _min_image(p_cart, center, lattice)
            r1 = np.random.rand()
            r2 = np.random.rand()
            velocity = (inertia * state.v_cart[position]
                        + cognitive * r1 * d_p
                        + (step * r2) * d_g)
            # Textbook PSO v_max: at most half a lattice vector per generation.
            # Independent of swarmStep so step=0 can still carry momentum.
            v_max = 0.5 * np.linalg.norm(lattice, axis=1)
            velocity = np.clip(velocity, -v_max, v_max)
            new_center = center + velocity
            state.v_cart[position] = velocity

            coords = coords + (new_center - center)          # rigid translation
            # Rotational PSO: compose inertia, pbest and g rotation steps.
            # Bond vectors live in the shared Cartesian basis, so no lattice
            # mapping is needed here (unlike the centre); an affine map would
            # deform bond lengths, so Kabsch extracts a best-fit pure rotation.
            P = coords - new_center
            Q_g = np.asarray(best.MOLECULES[position].MOLCOORS, dtype=float)
            Q_g = Q_g - Q_g.mean(axis=0)
            Q_p = np.asarray(state.pbest_coords[position], dtype=float)
            if (P.shape == Q_g.shape == Q_p.shape
                    and np.linalg.norm(P) > 1e-12
                    and np.linalg.norm(Q_g) > 1e-12
                    and np.linalg.norm(Q_p) > 1e-12):
                q_g = _quat_from_matrix(_kabsch_rotation(P, Q_g))
                theta_g = 2.0 * np.arccos(min(1.0, abs(float(q_g[0]))))
                t_g = (np.random.rand()
                       * (min(1.0, angle_max / theta_g) if theta_g > 1e-12 else 0.0))
                R_social = _quat_to_matrix(_quat_slerp(_IDENTITY_QUAT, q_g, t_g))

                # A fresh state has pbest == current; Kabsch on rank-deficient
                # data could otherwise invent a twist, so short-circuit it.
                if np.allclose(P, Q_p, atol=1e-10):
                    q_p = _IDENTITY_QUAT.copy()
                else:
                    q_p = _quat_from_matrix(_kabsch_rotation(P, Q_p))
                theta_p = 2.0 * np.arccos(min(1.0, abs(float(q_p[0]))))
                t_p = (cognitive * np.random.rand()
                       * (min(1.0, angle_max / theta_p) if theta_p > 1e-12 else 0.0))
                R_cognitive = _quat_to_matrix(_quat_slerp(_IDENTITY_QUAT, q_p, t_p))

                R_inertia = _quat_to_matrix(
                    _quat_slerp(_IDENTITY_QUAT, state.q_v[position], inertia))

                R_net = R_inertia @ R_cognitive @ R_social
                q_net = _quat_from_matrix(R_net)
                theta_net = 2.0 * np.arccos(min(1.0, abs(float(q_net[0]))))
                t_cap = min(1.0, angle_max / theta_net) if theta_net > 1e-12 else 0.0
                q_applied = _quat_slerp(_IDENTITY_QUAT, q_net, t_cap)
                coords = new_center + P @ _quat_to_matrix(q_applied)
                state.q_v[position] = q_applied

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
