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


def CalcFitness_310():
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    fitness = np.zeros(len(POP.POPULATION), dtype=float)
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
        elif ORG.optType == 13:
            dist = getattr(s, "distance", None)
            if isinstance(dist, (int, float)) and not math.isnan(dist) and not math.isinf(dist):
                fitness[i] = float(dist)
            else:
                fitness[i] = float("nan")
        else:
            # Fallback to enthalpy for not-yet-ported optTypes.
            factor = sum(map(int, s.numMols)) / max(1, sum(map(int, ORG.numMols)))
            fitness[i] = enthalpy / max(1e-12, factor)

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

    # Native port of `uspexkit denevo`: after the first generation has been
    # evaluated (every structure carries the dict uspexkit.gp() returned, the
    # model having just been trained on the database), draw the GP status
    # figure from memory.  To plot every generation instead, drop the
    # `== 1` condition.
    if int(getattr(POP, "generation", 0)) == 1 and gp_enabled(ORG):
        try:
            from gpevo.FunctionFolder.USPEX.src.gp_plot import plot_gp_generation
            plot_gp_generation(ORG, POP, fitness)
        except Exception as exc:            # a plot must never kill the run
            console_print(f"Warning: GP status plot failed: {exc}")

    return fitness
