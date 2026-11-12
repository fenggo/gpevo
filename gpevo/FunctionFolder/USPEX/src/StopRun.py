"""Python port of StopRun.m."""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G


def StopRun(fitness, generation, num_generations, stop_crit=0, stop_fitness=None):
    values = np.asarray(fitness, dtype=float).reshape(-1)

    if generation >= num_generations:
        return 1, f"Maximum Generation number achieved: {num_generations}"

    if stop_fitness is not None:
        if values.size and float(np.min(values)) <= float(stop_fitness):
            return 1, f"Target fitness achieved: {stop_fitness}"
    elif generation >= stop_crit:
        from gpevo.FunctionFolder.USPEX.src.SameBest import SameBest

        if SameBest() == stop_crit - 1:
            return 1, f"Halting criteria achieved: {stop_crit}"

    return 0, ""
