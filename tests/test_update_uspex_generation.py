"""Fitness must follow the population index, not the completion rank.

CalcFitness_310 keys fitness[i] by POP.POPULATION[i].  When structures finish
out of order, update_USPEX_GENERATION walks them in DoneOrder (completion)
order -- but must still read fitness by the *population* index.  The MATLAB
reference does exactly this: fitness(ranking(i)), not fitness(i).  A previous
revision used fitness[position] and attached each fitness to the wrong
structure whenever completion order differed from population order.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.USPEX.src.update_USPEX_GENERATION import (
    update_USPEX_GENERATION,
)


def _setup(done_order, n):
    """POP of n individuals whose Numbers are 1..n.

    update_USPEX_GENERATION writes to USPEX.POPULATION[Number - 1], so the
    cumulative record for structure k sits at index k-1.  POP.POPULATION
    indexes are the fitness keys.
    """
    POP = AttrDict()
    POP.generation = 1
    POP.POPULATION = [AttrDict(Number=i + 1, numIons=[1]) for i in range(n)]
    POP.DoneOrder = list(done_order)
    POP.convex_hull = []
    G["POP_STRUC"] = POP
    USPEX = AttrDict(POPULATION=[AttrDict(Number=i + 1) for i in range(n)],
                     GENERATION=[])
    G["USPEX_STRUC"] = USPEX
    return POP, USPEX


@pytest.fixture(autouse=True)
def _clean_globals():
    yield
    G["POP_STRUC"] = None
    G["USPEX_STRUC"] = None


def _fitness_by_number():
    return {r.Number: r.Fitness for r in G["USPEX_STRUC"].POPULATION
            if hasattr(r, "Number") and hasattr(r, "Fitness")}


def test_fitness_follows_population_index_when_completion_is_out_of_order():
    # POP index 0 finished last, index 1 first, index 2 second.
    _setup(done_order=[3, 1, 2], n=3)
    # fitness[0] belongs to POP index 0 (Number 1), fitness[1] to Number 2,
    # fitness[2] to Number 3 -- regardless of completion order.
    update_USPEX_GENERATION([1], np.array([-10.0, -20.0, -30.0]), 1)

    assert _fitness_by_number() == {
        1: pytest.approx(-10.0),       # old fitness[position] gave -30
        2: pytest.approx(-20.0),       # old gave -10
        3: pytest.approx(-30.0),       # old gave -20
    }


def test_identity_completion_order_keeps_fitness_aligned():
    _setup(done_order=[1, 2, 3], n=3)
    update_USPEX_GENERATION([1], np.array([-1.0, -2.0, -3.0]), 1)
    assert _fitness_by_number() == {
        1: pytest.approx(-1.0), 2: pytest.approx(-2.0), 3: pytest.approx(-3.0)}


def test_unfinished_structures_are_not_written():
    # DoneOrder 0 means the structure never finished: keep its prior fitness.
    _setup(done_order=[0, 1, 2], n=3)
    for r in G["USPEX_STRUC"].POPULATION:
        if hasattr(r, "Number"):
            r.Fitness = -999.0

    update_USPEX_GENERATION([2], np.array([-10.0, -20.0, -30.0]), 1)

    by_number = _fitness_by_number()
    assert by_number[1] == -999.0                   # never finished -> untouched
    assert by_number[2] == pytest.approx(-20.0)     # fitness keyed by POP index
    assert by_number[3] == pytest.approx(-30.0)


def test_done_order_length_mismatch_uses_population_order():
    # Mismatched DoneOrder falls back to range(len(POPULATION)); with the
    # identity fallback, fitness still lands on the same population index.
    _setup(done_order=[5, 6], n=2)
    update_USPEX_GENERATION([1], np.array([-7.0, -8.0]), 1)
    assert _fitness_by_number() == {
        1: pytest.approx(-7.0), 2: pytest.approx(-8.0)}
