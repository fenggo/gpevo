"""Tests for USPEX tournament parent selection.

build_tournament() is the Python port of update_STUFF.m:103-106: rank position
p (0-based, best first) gets weight (N-p)^2, stored as a reverse cumulative
sum so that find(tournament > r) in MATLAB becomes `tournament > draw` here.
pick_parent() is Heredity_310.m:20-32: draw uniformly in [0, max], take the
last entry above the draw, and map it through POP.ranking.

These tests pin the exact table, the rank probabilities implied by it (best
individual weight N^2, worst 1), and the draw -> ranking mapping.  No test
previously covered either function; other operator tests merely stub them.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.USPEX.src.variation_common import (
    build_tournament, update_tournament, pick_parent, valid_ranking,
)


def _pop(ranking, generation=1, bad_rank=0, population=None):
    """POP fixture: valid_ranking() needs POPULATION sized to hold the ranking."""
    if population is None:
        population = [None] * (max(ranking) + 1 if ranking else 1)
    return AttrDict(ranking=ranking, generation=generation, bad_rank=bad_rank,
                    POPULATION=population)


# --------------------------------------------------------------------------
# build_tournament
# --------------------------------------------------------------------------

def test_build_tournament_matches_matlab_table():
    """update_STUFF.m:103-106 with howManyProliferate = 5."""
    # MATLAB: t(5)=1; t(4)=1+2^2=5; t(3)=5+3^2=14; t(2)=14+4^2=30;
    #         t(1)=30+5^2=55
    assert build_tournament(5) == [55.0, 30.0, 14.0, 5.0, 1.0]


def test_build_tournament_single_and_degenerate_counts():
    assert build_tournament(0) == []
    assert build_tournament(1) == [1.0]
    assert build_tournament(-3) == []


def test_build_tournament_is_strictly_decreasing():
    table = build_tournament(8)
    assert len(table) == 8
    assert all(a > b for a, b in zip(table, table[1:]))
    assert table[-1] == 1.0                     # worst rank weight is 1


def test_build_tournament_weights_are_squared_rank_gaps():
    """Consecutive differences are N^2, (N-1)^2, ... , 1^2."""
    table = np.asarray(build_tournament(4), dtype=float)
    diffs = table[:-1] - table[1:]              # [t0-t1, t1-t2, t2-t3]
    assert diffs.tolist() == [16.0, 9.0, 4.0]   # 4^2, 3^2, 2^2 (last=1)


def test_build_tournament_best_rank_has_max_weight():
    """The first cumulative entry equals sum of all squared weights."""
    table = np.asarray(build_tournament(5), dtype=float)
    assert table[0] == sum(k ** 2 for k in range(1, 6))  # 55


# --------------------------------------------------------------------------
# update_tournament
# --------------------------------------------------------------------------

def test_update_tournament_uses_bestfrac_and_bad_rank(monkeypatch):
    ORG = AttrDict(bestFrac=0.5, initialPopSize=10, tournament=None)
    POP = _pop([0, 1, 2, 3, 4, 5, 6, 7], bad_rank=2)
    monkeypatch.setitem(G, "ORG_STRUC", ORG)
    monkeypatch.setitem(G, "POP_STRUC", POP)

    table = update_tournament()

    # N_pro = round(8*0.5) = 4; howMany = min(4, 8-2) = 6 -> 4
    assert table == build_tournament(4)
    assert G["ORG_STRUC"].tournament == table


def test_update_tournament_generation_zero_caps_at_initial_pop(monkeypatch):
    ORG = AttrDict(bestFrac=0.9, initialPopSize=3, tournament=None)
    POP = _pop(list(range(20)), generation=0)
    monkeypatch.setitem(G, "ORG_STRUC", ORG)
    monkeypatch.setitem(G, "POP_STRUC", POP)

    table = update_tournament()

    assert table == build_tournament(3)         # capped at initialPopSize


# --------------------------------------------------------------------------
# pick_parent
# --------------------------------------------------------------------------

def test_pick_parent_maps_draw_to_rank(monkeypatch):
    """Heredity_310.m:20-22: last tournament entry above draw -> ranking[i]."""
    ranking = [7, 3, 9, 1]                     # POP indices, best first
    table = np.asarray(build_tournament(4), dtype=float)  # [30, 14, 5, 1]
    ORG = AttrDict(tournament=table, bestFrac=0.5)
    monkeypatch.setitem(G, "ORG_STRUC", ORG)
    monkeypatch.setitem(G, "POP_STRUC", _pop(ranking))

    # draw in (14, 30] -> only t[0]=30 > draw -> position 0 -> ranking[0] = 7
    monkeypatch.setattr(np.random, "uniform", lambda a, b: 20.0)
    assert pick_parent() == 7

    # draw in (5, 14] -> t[0], t[1] above -> last index 1 -> ranking[1] = 3
    monkeypatch.setattr(np.random, "uniform", lambda a, b: 10.0)
    assert pick_parent() == 3

    # draw in (1, 5] -> last index 2 -> ranking[2] = 9
    monkeypatch.setattr(np.random, "uniform", lambda a, b: 3.0)
    assert pick_parent() == 9

    # draw in [0, 1] -> all above -> last index 3 -> ranking[3] = 1
    monkeypatch.setattr(np.random, "uniform", lambda a, b: 0.5)
    assert pick_parent() == 1


def test_pick_parent_falls_back_to_uniform_over_ranking(monkeypatch):
    """No tournament table -> uniform pick from valid ranking."""
    ranking = [4, 5, 6]
    # tournament=[] (not None): np.asarray(None, dtype=float) is array([nan])
    # with size 1, which would skip the update_tournament() branch.
    ORG = AttrDict(tournament=[], bestFrac=0.5, initialPopSize=3)
    monkeypatch.setitem(G, "ORG_STRUC", ORG)
    monkeypatch.setitem(G, "POP_STRUC", _pop(ranking))
    monkeypatch.setattr(np.random, "randint", lambda low, high: 1)

    # update_tournament() builds a table from ranking, so to exercise the
    # uniform branch we stub it to return an empty table.
    monkeypatch.setattr(
        "gpevo.FunctionFolder.USPEX.src.variation_common.update_tournament",
        lambda: [])

    assert pick_parent() == ranking[1]         # 5


def test_pick_parent_none_when_no_ranking(monkeypatch):
    monkeypatch.setitem(G, "ORG_STRUC", AttrDict(tournament=[1.0]))
    # empty POPULATION makes valid_ranking() return [] (no fallback range)
    monkeypatch.setitem(G, "POP_STRUC", _pop([], population=[]))
    assert pick_parent() is None


def test_pick_parent_respects_bestfrac_subset(monkeypatch):
    """Only ranking entries covered by the tournament table can be drawn."""
    ranking = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    table = np.asarray(build_tournament(3), dtype=float)  # best 3 only
    monkeypatch.setitem(G, "ORG_STRUC", AttrDict(tournament=table, bestFrac=0.3))
    monkeypatch.setitem(G, "POP_STRUC", _pop(ranking))
    monkeypatch.setattr(np.random, "uniform", lambda a, b: 0.1)

    drawn = pick_parent()
    assert drawn in ranking[:3]

    seen = set()
    for _ in range(200):
        seen.add(pick_parent())
    assert seen <= set(ranking[:3])


# --------------------------------------------------------------------------
# valid_ranking
# --------------------------------------------------------------------------

def test_valid_ranking_filters_stale_indices(monkeypatch):
    POP = AttrDict(ranking=[1, 3, 99, -1, 2], POPULATION=[None] * 5)
    monkeypatch.setitem(G, "POP_STRUC", POP)
    assert valid_ranking(POP) == [1, 3, 2]


def test_valid_ranking_falls_back_to_all_indices():
    POP = AttrDict(ranking=[99], POPULATION=[None] * 3)
    assert valid_ranking(POP) == [0, 1, 2]
