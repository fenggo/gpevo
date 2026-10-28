"""Python equivalent of CalcFitness_310.m (optType 1/11/12/13 subset)."""
from __future__ import annotations

import math
import numpy as np

from gpevo.runtime import G
from gpevo.log_utils import console_print
from gpevo.matrix import det
from gpevo.FunctionFolder.sys.elementMass import calc_density
from gpevo.FunctionFolder.USPEX.src.gp_ml import gp_active_learning, gp_enabled
from .use_gp_density import use_gp_density


def _usable(value):
    """A finite number, i.e. something optType 13 can use as a distance."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


def CalcFitness_310():
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    fitness = np.zeros(len(POP.POPULATION), dtype=float)
    missing_distance = 0
    for i, s in enumerate(POP.POPULATION):
        enthalpy = float(np.asarray(s.Enthalpies)[-1])
        if enthalpy >= 9999:
            fitness[i] = 100000.0
            continue
        if ORG.optType == 1:
            factor = sum(map(int, s.numMols)) / max(1, sum(map(int, ORG.numMols)))
            if factor <= 0:
                factor = 1.0
            fitness[i] = enthalpy / factor
        elif ORG.optType == 11:                              # density
            volume = det(np.asarray(s.LATTICE).reshape(3, 3))
            density = calc_density(s.numIons, ORG.atomType, volume)
            density = use_gp_density(s, density)
            fitness[i] = -1.0 * float(density if density is not None else 0.0)
        elif ORG.optType == 12:                              # density combined with entropy
            volume = det(np.asarray(s.LATTICE).reshape(3, 3))
            density = calc_density(s.numIons, ORG.atomType, volume)
            density = use_gp_density(s, density)
            factor = (sum(map(int, s.numMols)) / max(1, sum(map(int, ORG.numMols)))) * sum(ORG.numMols)
            fitness[i] = enthalpy / max(1e-12, factor) - 0.5 * float(density if density is not None else 0.0)
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
        else:
            # Fallback to enthalpy for not-yet-ported optTypes.
            factor = sum(map(int, s.numMols)) / max(1, sum(map(int, ORG.numMols)))
            fitness[i] = enthalpy / max(1e-12, factor)

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

    # Native port of `uspexkit denevo`: draw the GP status figure from memory
    # every generation that actually carries predictions from ``gp()``.  Before
    # the training set is seeded there is nothing to draw and plot_gp_generation
    # says so and returns.
    if int(getattr(POP, "generation", 0)) >= 1 and gp_enabled(ORG):
        try:
            from gpevo.FunctionFolder.USPEX.src.gp_plot import plot_gp_generation
            plot_gp_generation(ORG, POP, fitness)
        except Exception as exc:            # a plot must never kill the run
            console_print(f"Warning: GP status plot failed: {exc}")

    return fitness
