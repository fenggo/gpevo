"""Personal-best / velocity state carried by each individual for the swarm move.

The state lives ON the individual as ind.swarm_state: an AttrDict holding
plain numpy arrays and floats.  That makes it survive both copy.deepcopy
(offspring / elites) and Current_POP.pkl (pickle) restarts -- the reason
the textbook-PSO terms were originally left out (docs/swarm_operator.md 6.4).

Conventions
-----------
* pbest_frac   -- molecule centres in FRACTIONAL coordinates.  Lattice
  independent, so a lineage's personal best stays meaningful when latmutation
  changes the cell.
* v_cart       -- translational velocity in Cartesian Angstrom, the same
  units as the min-image displacement used by the operator.
* q_v          -- last applied orientation step as a unit quaternion per
  molecule: the rotational analogue of velocity.
* pbest_coords -- centred (n_atoms, 3) Cartesian coordinates at the
  personal best, used to recover the personal-best orientation via Kabsch.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import AttrDict

STATE_KEY = "swarm_state"
_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _centres(ind):
    return np.array([np.asarray(m.MOLCOORS, dtype=float).mean(axis=0)
                     for m in ind.MOLECULES], dtype=float)


def fractional_centres(ind):
    """(n_mols, 3) molecule centres in fractional coordinates."""
    cell = np.asarray(ind.LATTICE, dtype=float).reshape(3, 3)
    return _centres(ind) @ np.linalg.inv(cell)


def centred_coords(ind):
    """List of centred Cartesian coordinate arrays, one per molecule."""
    out = []
    for mol in ind.MOLECULES:
        c = np.asarray(mol.MOLCOORS, dtype=float)
        out.append(c - c.mean(axis=0))
    return out


def _fresh_state(ind):
    n = len(ind.MOLECULES)
    try:
        pfrac = np.asarray(fractional_centres(ind), dtype=float)
    except np.linalg.LinAlgError:
        pfrac = np.zeros((n, 3), dtype=float)
    return AttrDict(
        v_cart=np.zeros((n, 3), dtype=float),
        q_v=np.tile(_IDENTITY, (n, 1)),
        pbest_frac=pfrac,
        pbest_coords=centred_coords(ind),
        pbest_fitness=None,
    )


def _valid(state, n):
    """Shape check: rejects missing, stale and old-checkpoint states."""
    if not isinstance(state, dict):
        return False
    try:
        return (np.asarray(state["v_cart"]).shape == (n, 3)
                and np.asarray(state["q_v"]).shape == (n, 4)
                and np.asarray(state["pbest_frac"]).shape == (n, 3)
                and len(state["pbest_coords"]) == n)
    except (KeyError, TypeError, ValueError):
        return False


def ensure_swarm_state(ind):
    """Return a well-formed state, creating one for fresh / legacy individuals."""
    n = len(ind.MOLECULES)
    state = getattr(ind, STATE_KEY, None)
    if not _valid(state, n):
        state = _fresh_state(ind)
        setattr(ind, STATE_KEY, state)
    return state


def update_pbest(POP, fitness):
    """Refresh each individual's personal best from fresh (lower-is-better) fitness.

    Failed structures (non-finite or >= 99999) are ignored.  Missing / stale
    states are reinitialised first, so old checkpoints and seed structures are
    safe.  Never raises: swarm bookkeeping must not break fitness evaluation.
    """
    f = np.asarray(fitness, dtype=float).reshape(-1)
    for i, ind in enumerate(POP.POPULATION):
        if i >= f.size:
            break
        val = float(f[i])
        if not np.isfinite(val) or val >= 99999.0:
            continue
        try:
            state = ensure_swarm_state(ind)
            best = state.pbest_fitness
            if best is None or not np.isfinite(float(best)) or val < float(best):
                state.pbest_fitness = val
                state.pbest_frac = np.asarray(fractional_centres(ind), dtype=float)
                state.pbest_coords = centred_coords(ind)
        except Exception:
            continue
