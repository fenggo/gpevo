"""P3: a freshly assembled offspring population must not inherit the parent ranking.

``POP.ranking`` indexes the *parent* generation.  Copying it onto ``OFF`` (whose
POPULATION is empty at that point, then filled with new offspring) leaves stale
indices; ``valid_ranking`` silently drops the out-of-range ones and can return a
short, wrongly ordered list until FitnessRankingCorrection republishes it.
"""
from __future__ import annotations

import ast
import inspect
from importlib import import_module

from gpevo.runtime import struct


def test_ea310_starts_offspring_with_an_empty_ranking():
    module = import_module("gpevo.FunctionFolder.USPEX.310.EA_310")
    tree = ast.parse(inspect.getsource(module.EA_310))

    values = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Attribute)
                    and target.attr == "ranking"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "OFF"):
                values.append(node.value)

    assert len(values) == 1, "expected exactly one OFF.ranking assignment"
    assert isinstance(values[0], ast.List) and not values[0].elts, (
        "OFF.ranking must start as an empty list, not list(POP.ranking)")


def test_valid_ranking_falls_back_to_natural_order_for_a_new_population():
    from gpevo.FunctionFolder.USPEX.src.variation_common import valid_ranking

    pop = struct(POPULATION=[struct() for _ in range(5)], ranking=[])
    assert valid_ranking(pop) == [0, 1, 2, 3, 4]

    pop.ranking = 0                      # what WriteGenerationStart writes
    assert valid_ranking(pop) == [0, 1, 2, 3, 4]

    pop.ranking = [4, 3, 2, 1, 0]       # a real, in-range ranking
    assert valid_ranking(pop) == [4, 3, 2, 1, 0]


def test_stale_parent_ranking_is_exactly_what_we_want_to_avoid():
    from gpevo.FunctionFolder.USPEX.src.variation_common import valid_ranking

    pop = struct(POPULATION=[struct() for _ in range(5)],
                 ranking=[10, 3, 99, 1, 7])
    # Out-of-range indices are dropped rather than falling back to natural
    # order, so the caller sees only two wrongly ordered survivors.
    assert valid_ranking(pop) == [3, 1]
