"""Python equivalent of CalcFitness_310.m (optType 1/11/12/13/14 subset)."""
from __future__ import annotations

import math
import os
import numpy as np

from gpevo.runtime import G
from gpevo.log_utils import console_print
from gpevo.matrix import det
from gpevo.FunctionFolder.sys.elementMass import calc_density
from gpevo.FunctionFolder.USPEX.src.gp_ml import gp_active_learning, gp_enabled
from .use_gp_density import use_gp_density
from .use_gp_energy import use_gp_energy


def _usable(value):
    """A finite number, i.e. something optType 13 can use as a distance."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


# optType 14: fingerprints of the fixed target structure(s).  Keyed by
# (path, mtime_ns, RmaxFing, sigmaFing, deltaFing, dimension, atomType,
#  bond_cut, molecule).  Recomputed only when the file or the fingerprint
# parameters change.
_TARGET_FP_CACHE = {}


def _target_bond_cut(ORG):
    """Cutoff separating intra- from inter-molecular pairs (Angstrom)."""
    cut = float(getattr(ORG, "targetBondCut", 0.0) or 0.0)
    if cut > 0:
        return cut
    # The IonDistances block holds the shortest allowed separations; its
    # maximum is a safe covalent-bond cutoff (1.9 A for the TNT system).
    raw = getattr(ORG, "minDistMatrice", None)
    md = np.asarray(raw if raw is not None else [], dtype=float)
    return float(md.max()) if md.size else 1.9


def _resolve_targets(ORG):
    """Absolute paths of every targetStructure, relative to the run home."""
    raw = getattr(ORG, "targetStructure", None)
    if raw is None:
        raw = []
    if isinstance(raw, str):
        raw = raw.split(",")
    paths = [str(p).strip() for p in raw if str(p).strip()]
    if not paths:
        raise RuntimeError(
            "optType 14 needs `targetStructure : <POSCAR/CIF/.traj[,..]>` in INPUT.txt"
        )
    home = getattr(ORG, "homePath", None) or os.getcwd()
    out = []
    for path in paths:
        full = path if os.path.isabs(path) else os.path.join(home, path)
        if not (os.path.isfile(full) and os.access(full, os.R_OK)):
            raise RuntimeError(
                f"optType 14: target structure is missing or unreadable: {full} "
                f"(home={home})"
            )
        out.append(os.path.abspath(full))
    return out


def _composition_scale(num_ions, expected):
    """k when ``num_ions == k * expected`` element-wise, else None.

    The fingerprint is a normalised RDF, so a target that is an integer (or
    rational) multiple of the run's cell is comparable: volume and every
    N_i N_j scale together.  A different *ratio* of species is not.
    """
    a = np.asarray(num_ions, dtype=float)
    b = np.asarray(expected, dtype=float)
    if a.shape != b.shape or a.size == 0 or np.any(b <= 0):
        return None
    ratios = a / b
    k = float(ratios[0])
    if k <= 0 or not np.allclose(ratios, k, rtol=1e-6, atol=1e-9):
        return None
    # A cell can only hold an integer number of primitive cells, so k
    # (target larger) or 1/k (target smaller) must be an integer.  This
    # rejects e.g. a 1.5x cell, which is not a valid supercell.
    bigger = k if k >= 1.0 else 1.0 / k
    if abs(bigger - round(bigger)) > 1e-6:
        return None
    return k


def _molecule_ids(D, cut):
    """Connected components of the D < cut bond graph, one id per molecule."""
    n = D.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.where(np.triu(D < cut, k=1))
    for a, b in zip(ii, jj):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    return np.unique([find(i) for i in range(n)], return_inverse=True)[1]


def _fingerprint_from_atoms(atoms, ORG, cut):
    """Fingerprint one external structure with the population's convention.

    ``compute_structure_metrics`` (fingerprint_utils.py:492) stores fractional
    coordinates, species-blocked in ORG.atomType order, and drops zero-shift
    intra-molecular pairs.  Rebuild exactly that here so the cosine distance
    to a population FINGERPRINT is meaningful for any input atom order.
    """
    from gpevo.FunctionFolder.sys.fingerprint_utils import (
        compute_full_atom_fingerprint,
    )

    nums = np.asarray(atoms.get_atomic_numbers())
    species = [int(z) for z in ORG.atomType]  # ORG.atomType holds ATOMIC NUMBERS
    if sorted(set(nums.tolist())) != sorted(set(species)):
        raise RuntimeError(
            f"optType 14: target species {sorted(set(nums.tolist()))} do not match "
            f"ORG.atomType {sorted(set(species))}"
        )
    perm = np.concatenate([np.where(nums == z)[0] for z in species])
    num_ions = [int((nums == z).sum()) for z in species]

    expected = None
    if int(getattr(ORG, "molecule", 0) or 0) == 1 and getattr(ORG, "numMols", None) is not None:
        try:
            from gpevo.FunctionFolder.USPEX.src.GetPOP_MOL import GetPOP_MOL
            expected = [int(v) for v in GetPOP_MOL(list(ORG.numMols))[2]]
        except Exception:
            expected = None
    elif getattr(ORG, "numIons", None) is not None:
        expected = [int(v) for v in ORG.numIons]
    if expected is not None:
        k = _composition_scale(num_ions, expected)
        if k is None:
            raise RuntimeError(
                f"optType 14: target composition {num_ions} is not a uniform "
                f"multiple of the run's {expected}. The species RATIO must be "
                "identical (e.g. target 8:8 vs run 4:4 is OK, 8:4 is not)."
            )
        if abs(k - 1.0) > 1e-6:
            # Printed once per target file: _target_fingerprints caches the
            # result, and a changed file re-reads (and re-warns) once.
            console_print(
                f"  optType 14: target has {k:g}x the run's molecules "
                f"({num_ions} vs {expected}). The RDF fingerprint is "
                "size-intensive, but different cells leave a small systematic "
                "residual (~0.02 measured). A hit means SIMILAR PACKING, not "
                "an identical structure; use targetTolerance accordingly."
            )

    cell = np.asarray(atoms.cell.array, dtype=float).reshape(3, 3)
    # Fractional coordinates, as in compute_structure_metrics:526.  Do NOT
    # convert to Cartesian (pickAntiSeeds.py does that and is not comparable).
    coords = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)[perm]

    intra_map = None
    if int(getattr(ORG, "molecule", 0) or 0) == 1:
        D = atoms.get_all_distances(mic=True)[np.ix_(perm, perm)]
        mol = _molecule_ids(D, cut)
        # 1 = inter-molecular (keep), 0 = intra-molecular (drop), matching
        # intra_mol_dist / compute_full_atom_fingerprint:362-369.
        intra_map = (mol[:, None] != mol[None, :]).astype(np.int8)

    (_, fing, _), _ = compute_full_atom_fingerprint(
        cell, coords, num_ions,
        rmax=float(getattr(ORG, "RmaxFing", 10.0)),
        sigma=float(getattr(ORG, "sigmaFing", 0.03)),
        delta=float(getattr(ORG, "deltaFing", 0.08)),
        dimension=int(getattr(ORG, "dimension", 3)),
        intra_map=intra_map,
    )
    return np.asarray(fing, dtype=float)


def _target_fingerprints(ORG):
    """Cached fingerprints of every frame of every targetStructure."""
    from ase.io import read

    cut = _target_bond_cut(ORG)
    pars = (
        float(getattr(ORG, "RmaxFing", 10.0)),
        float(getattr(ORG, "sigmaFing", 0.03)),
        float(getattr(ORG, "deltaFing", 0.08)),
        int(getattr(ORG, "dimension", 3)),
        tuple(int(z) for z in ORG.atomType),
        cut,
        int(getattr(ORG, "molecule", 0) or 0),
    )
    out = []
    for path in _resolve_targets(ORG):
        key = (path, os.stat(path).st_mtime_ns) + pars
        cached = _TARGET_FP_CACHE.get(key)
        if cached is None:
            try:
                images = read(path, index=":")
            except Exception as exc:
                raise RuntimeError(f"optType 14: cannot read target {path}: {exc}") from exc
            cached = [_fingerprint_from_atoms(a, ORG, cut) for a in images]
            _TARGET_FP_CACHE[key] = cached
        out.extend(cached)
    if not out:
        raise RuntimeError("optType 14: no target structures were read")
    return out


def _fp_distance(fing, target, ORG):
    """Weighted fingerprint cosine distance, as in KeepBestStructures._distance."""
    from gpevo.FunctionFolder.sys.fingerprint_utils import _cosine_distance

    a = np.asarray(fing, dtype=float)
    b = np.asarray(target, dtype=float)
    if a.shape != b.shape:
        raise RuntimeError(
            f"optType 14: fingerprint shape {a.shape} != target {b.shape}; "
            "the target must use the same atomType/RmaxFing/deltaFing as the run"
        )
    w = np.asarray(getattr(ORG, "weight", 1.0), dtype=float).reshape(-1)
    if w.size != a.shape[0]:
        w = np.ones(a.shape[0], dtype=float)
    return float(_cosine_distance(a, b, w))


def CalcFitness_310():
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    fitness = np.zeros(len(POP.POPULATION), dtype=float)
    missing_distance = 0
    missing_fp = 0
    # optType 14 needs the target fingerprint(s) once per generation, not
    # once per individual.  Raises early with a clear message on bad config.
    target_fps = (_target_fingerprints(ORG)
                  if int(getattr(ORG, "optType", 1) or 1) == 14 else None)
    for i, s in enumerate(POP.POPULATION):
        enthalpy = float(np.asarray(s.Enthalpies)[-1])
        if enthalpy >= 9999:
            fitness[i] = 100000.0
            continue
        if ORG.optType == 1:
            factor = sum(map(int, s.numMols)) / max(1, sum(map(int, ORG.numMols)))
            if factor <= 0:
                factor = 1.0
            # optType 1 minimises the enthalpy, so with the GP enabled the
            # fitness is the GP-predicted enthalpy (the same quantity EI ranks
            # on), not the force-field value left in Enthalpies by Reading.
            # optType 11 has the equivalent use_gp_density in its own branch.
            # Enthalpies itself stays untouched: it is still used for the
            # failed-structure check below.
            if gp_enabled(ORG):
                enthalpy = use_gp_energy(s, enthalpy)
            fitness[i] = enthalpy / factor
        elif ORG.optType == 11:                              # density
            volume = det(np.asarray(s.LATTICE).reshape(3, 3))
            density = calc_density(s.numIons, ORG.atomType, volume)
            density = use_gp_density(s, density)
            fitness[i] = -1.0 * float(density if density is not None else 0.0)
        elif ORG.optType == 12:                              # 0.1*enthalpy/factor - density
            volume = det(np.asarray(s.LATTICE).reshape(3, 3))
            density = calc_density(s.numIons, ORG.atomType, volume)
            # With `1 : gp` both terms are the GP predictions (energy_gp and
            # density_gp); with the GP off they stay the force-field enthalpy
            # and the geometric density.
            if gp_enabled(ORG):
                enthalpy = use_gp_energy(s, enthalpy)
                density = use_gp_density(s, density)
            factor = (sum(map(int, s.numMols)) / max(1, sum(map(int, ORG.numMols)))) * sum(ORG.numMols)
            fitness[i] = 0.1 * enthalpy / max(1e-12, factor) - float(density if density is not None else 0.0)
        elif ORG.optType == 13:                              # distance to dataset
            # CalcFitness_310.m:46-53 uses POPULATION(i).distance, which is the
            # GP residual (Hilbert-space distance to the training set).  It is
            # stored by apply_gp_metrics / apply_pred_metrics, so read the
            # dedicated field first and fall back to the MATLAB name.
            dist = getattr(s, "residual", None)
            if not _usable(dist):
                dist = getattr(s, "distance", None)
            if _usable(dist):
                fitness[i] = float(dist)
            else:
                fitness[i] = float("nan")
                missing_distance += 1
        elif ORG.optType == 14:                              # fingerprint distance to target
            # Lower is better: 0 means the structure IS one of the targets.
            fing = getattr(s, "FINGERPRINT", None)
            if fing is None or np.asarray(fing).size == 0:
                fitness[i] = 1.0
                missing_fp += 1
            else:
                fitness[i] = min(_fp_distance(fing, t, ORG) for t in target_fps)
        else:
            # Fallback to enthalpy for not-yet-ported optTypes.
            factor = sum(map(int, s.numMols)) / max(1, sum(map(int, ORG.numMols)))
            fitness[i] = enthalpy / max(1e-12, factor)

    if ORG.optType == 14:
        if missing_fp:
            console_print(
                f"  WARNING: optType 14 — {missing_fp}/{len(POP.POPULATION)} structure(s) "
                "have no FINGERPRINT; assigned the maximum distance 1.0."
            )
        # Raw distances, before opt_sign.  A same-cell target reaches 0; a
        # different-size cell of the same packing has a small residual, so
        # 0.03 (not 0) is the default hit threshold.  Set `stopFitness` to
        # the same value to halt on the first hit.
        tol = float(getattr(ORG, "targetTolerance", 0.03) or 0.03)
        hits = int(np.count_nonzero(fitness <= tol))
        if hits:
            console_print(
                f"  optType 14: {hits}/{len(fitness)} structure(s) within "
                f"targetTolerance={tol:g} (best distance={float(fitness.min()):.6f})"
            )

    if ORG.optType == 13 and missing_distance:
        # .m:53 sets NaN here too, but silently.  Say why, and how to fix it:
        # the distance IS the GP residual, so the GP step has to run.
        console_print(
            f"  ERROR: optType 13 (distance) has no value for "
            f"{missing_distance}/{len(POP.POPULATION)} structure(s) in "
            f"generation {int(getattr(POP, 'generation', 0))}; fitness is NaN."
        )
        if not gp_enabled(ORG):
            console_print("         the distance is the GP residual — set `1 : gp` "
                          "in INPUT.txt to produce it.")
        else:
            console_print("         the GP step must complete for these structures "
                          "before their distance exists.")

    if ORG.opt_sign is None:
        ORG.opt_sign = 1
    fitness = ORG.opt_sign * fitness

    # Numerical stabilization: keep failed structures at 100000
    for i, s in enumerate(POP.POPULATION):
        if float(np.asarray(s.Enthalpies)[-1]) > 99999:
            fitness[i] = 100000.0
        if POP.generation > 1 and ORG.optType == 12 and np.isfinite(fitness[i]):
            pass  # original tolerance logic omitted in minimal port

    # CalcFitness_310.m:114-422 — GP active learning (EI top-K → DFT → pred).
    # No-op when `1 : gp` is off or no structure carries GP metrics.
    fitness = gp_active_learning(ORG, POP, fitness)

    POP.fitness = fitness
    G["POP_STRUC"] = POP

    # Swarm operator (docs/swarm_operator.md 6.4): refresh each individual's
    # personal best whenever a generation's fitness is evaluated.  The state
    # is stored on the individual, so it travels with the lineage through
    # Current_POP.pkl restarts.  Never raises.
    from gpevo.FunctionFolder.USPEX.src.swarm_state import update_pbest
    update_pbest(POP, fitness)

    # Native port of `uspexkit denevo`: draw the GP status figure from memory
    # every generation that actually carries predictions from ``gp()``.  The
    # figure is opt-in (`1 : gpplot`); with the default `0`, this call still
    # writes gp_status_history.csv and prints the terminal status report.
    # Before the training set is seeded there is nothing to draw and
    # plot_gp_generation says so and returns.
    if int(getattr(POP, "generation", 0)) >= 1 and gp_enabled(ORG):
        try:
            from gpevo.FunctionFolder.USPEX.src.gp_plot import plot_gp_generation
            plot_gp_generation(ORG, POP, fitness)
        except Exception as exc:            # a plot must never kill the run
            console_print(f"Warning: GP status plot failed: {exc}")

    return fitness
