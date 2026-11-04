"""RL_inner: let the bandit move one continuous amplitude and freeze the rest.

A single-parameter ablation (e.g. only the rotation angle) needs the other
amplitudes to stay at the value INPUT.txt gave them; without RL_inner the inner
sampler moves all four of them every generation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.USPEX.src.RL_agent import _inner_params, ANGLE_HI, ANGLE_LO

ALL = {(4, 0), (4, 1), (5, 0), (6, 0)}


def test_default_is_every_amplitude():
    assert _inner_params(AttrDict()) == ALL


@pytest.mark.parametrize("text", ["all", "ALL", "1", "on", ""])
def test_all_keeps_the_current_behaviour(text):
    assert _inner_params(AttrDict(RL_inner=text)) == ALL


def test_a_single_parameter_is_selected():
    assert _inner_params(AttrDict(RL_inner="angle")) == {(4, 0)}
    assert _inner_params(AttrDict(RL_inner="translation")) == {(4, 1)}
    assert _inner_params(AttrDict(RL_inner="mutationRate")) == {(5, 0)}
    assert _inner_params(AttrDict(RL_inner="softstep")) == {(6, 0)}


def test_an_arm_selects_both_of_its_amplitudes():
    assert _inner_params(AttrDict(RL_inner="rotate")) == {(4, 0), (4, 1)}


def test_a_list_selects_each_of_them():
    assert _inner_params(AttrDict(RL_inner="angle,translation")) == {(4, 0), (4, 1)}
    assert _inner_params(AttrDict(RL_inner="angle mutationRate")) == {(4, 0), (5, 0)}


@pytest.mark.parametrize("text", ["none", "off", "0"])
def test_nothing_selected(text):
    assert _inner_params(AttrDict(RL_inner=text)) == set()


def test_an_unknown_token_falls_back_to_everything(capsys):
    assert _inner_params(AttrDict(RL_inner="rotato")) == ALL
    assert "unknown" in capsys.readouterr().out


def _org(**kwargs):
    from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault

    org = createORGDefault()
    org.populationSize = 10
    org.bestFrac = 0.5
    org.RL_strategy = 1
    org.RL_autoExport = 0
    org.rotationAngleMax = math.radians(45.0)
    org.translationMax = 0.3
    org.mutationRate = 0.15
    org.softStepScale = 1.5
    org.update(kwargs)
    return org


def _pop():
    pop = AttrDict()
    pop.generation = 3
    pop.ranking = [0, 1]
    pop.bad_rank = 0
    pop.POPULATION = []
    for index, how_come in enumerate(("Rotation", "softmutate")):
        individual = AttrDict()
        individual.Enthalpies = np.array([-100.0 - index])
        individual.howCome = how_come
        pop.POPULATION.append(individual)
    return pop


def _run(org, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    G["ORG_STRUC"] = org
    G["POP_STRUC"] = _pop()
    try:
        from gpevo.FunctionFolder.USPEX.src.RL_agent import RL_agent

        for _ in range(4):
            RL_agent(10)
    finally:
        G["ORG_STRUC"] = None
        G["POP_STRUC"] = None


def test_only_the_selected_amplitude_moves(tmp_path, monkeypatch):
    """RL_inner : angle -- the other three keep the INPUT.txt value."""
    org = _org(RL_inner="angle")
    _run(org, tmp_path, monkeypatch)

    assert ANGLE_LO <= org.rotationAngleMax <= ANGLE_HI
    assert org.translationMax == pytest.approx(0.3)
    assert org.mutationRate == pytest.approx(0.15)
    assert org.softStepScale == pytest.approx(1.5)


def test_without_rl_inner_every_amplitude_moves(tmp_path, monkeypatch):
    """The default is unchanged: all four are sampled."""
    org = _org()
    del org["RL_inner"]
    _run(org, tmp_path, monkeypatch)

    moved = [org.rotationAngleMax != math.radians(45.0),
             org.translationMax != 0.3,
             org.mutationRate != 0.15,
             org.softStepScale != 1.5]
    assert all(moved), moved


def test_rl_inner_is_read_from_input(tmp_path):
    """Both sides of the colon are accepted, like the other GP options."""
    from gpevo.inputparser import InputParser

    path = tmp_path / "INPUT.txt"
    path.write_text("angle : RL_inner\n", encoding="utf-8")
    assert InputParser(str(path)).option("RL_inner") == "angle"

def _all_elite_pop():
    """Six offspring of one arm, all in the elite pool, with a spread of gains."""
    pop = AttrDict()
    pop.generation = 3
    pop.ranking = list(range(6))
    pop.bad_rank = 0
    pop.POPULATION = []
    for i in range(6):
        individual = AttrDict()
        individual.Enthalpies = np.array([-100.0 - i])
        individual.howCome = "Rotate"          # arm 4
        pop.POPULATION.append(individual)
    return pop


def test_reward_above_one_does_not_clamp_beta(tmp_path, monkeypatch):
    """wins > trials must not push beta to its floor (the arm would dominate)."""
    import pickle
    from gpevo.FunctionFolder.USPEX.src.RL_agent import RL_agent, STATE_FILE

    monkeypatch.chdir(tmp_path)
    G["ORG_STRUC"] = _org(bestFrac=1.0)       # every offspring is elite
    G["POP_STRUC"] = _all_elite_pop()
    try:
        RL_agent(6)
        with open(STATE_FILE, "rb") as fh:
            state = pickle.load(fh)
    finally:
        G["ORG_STRUC"] = None
        G["POP_STRUC"] = None

    beta = np.asarray(state["beta"], dtype=float)
    alpha = np.asarray(state["alpha"], dtype=float)
    # arm 4 has all 6 trials; the raw reward sum (~7.1) exceeds trials, which
    # used to make beta negative and clamp it to 1e-3.
    assert beta[3] > 0.5, beta
    assert alpha[3] > 1.0, alpha
    assert (beta > 1e-3 + 1e-9).all(), beta
    hist = state["history"][-1]
    assert (hist["successes"] <= hist["trials"]).all()

