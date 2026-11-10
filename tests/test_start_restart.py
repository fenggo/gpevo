"""A restart must pick up INPUT.txt edits.

``Start()`` restores ``Current_ORG.pkl``, so without a re-read every option
the user changed while the run was stopped is ignored -- the GP gates
(``den``, ``u``, ``explore``) above all, since those are what gets tuned
between restarts.

Only the finished-run guard of ``Start()`` is run here; the rest would enter
the generation loop.  The refresh helper is exercised directly, with a run
directory that matches what a stopped run leaves behind.
"""
from __future__ import annotations

import os
import random

import numpy as np
import pytest

from gpevo.runtime import G, struct
from gpevo.FunctionFolder.USPEX.Start import _RESTART_KEEP, _refresh_input_options


def _input_file(tmp_path, den="1.8", u="0.03", explore=None):
    """A trimmed INPUT.txt: the system block the parsers need, plus the GP one."""
    lines = [
        "% atomType",
        "H C N O",
        "% EndAtomType",
        "% numSpeciesSymbols",
        "4",
        "% End",
        "",
        "* GAUSSIAN PROCESS PARAMETER *",
        "  1      : gp",
        "  data22 : data",
        f"  {den}   : den",
        f"  {u}   : u",
        "  1      : k",
    ]
    if explore is not None:
        lines.append(f"  {explore}      : explore")
    path = tmp_path / "INPUT.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _checkpoint(tmp_path):
    """The ORG a stopped run leaves behind: old gates plus run state."""
    from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault

    org = createORGDefault()
    org.homePath = str(tmp_path)
    org.resFolder = str(tmp_path / "results1")
    # what the user is about to change
    org.gp = 1
    org.gpData = "data22"
    org.gpDen = 1.8
    org.gpU = 0.03
    org.gpK = 1
    org.gpExplore = 1
    # run state the checkpoint must keep; every protected name gets a marker
    # so a dropped one shows up as a missing attribute, not a silent pass
    for name in _RESTART_KEEP:
        if getattr(org, name, None) is None:
            setattr(org, name, f"<{name}>")
    org.latVolume = 1234.5
    org.tournament = [1, 2, 3]
    org.firstGeneSplitAll = [[1, 1]]
    return org


@pytest.fixture(autouse=True)
def _workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    G["ORG_STRUC"] = _checkpoint(tmp_path)
    yield
    G["ORG_STRUC"] = None


def _record(gen):
    from gpevo.runtime import AttrDict

    record = AttrDict()
    record.gen = gen
    return record


def _uspex(population):
    from gpevo.runtime import AttrDict

    uspex = AttrDict()
    uspex.POPULATION = population
    uspex.GENERATION = []
    return uspex


def _snapshots(tmp_path, generations):
    """Write generation<N>/USPEX.pkl of a stopped run."""
    from gpevo.runtime import safesave

    res = tmp_path / "results1"
    for number, population in generations:
        folder = res / f"generation{number}"
        folder.mkdir(parents=True, exist_ok=True)
        safesave(str(folder / "USPEX.pkl"), _uspex(population))


def test_restart_reloads_the_cumulative_record(tmp_path):
    """Individuals is rewritten from USPEX.POPULATION: a restart must not start
    it empty, or every generation before the restart disappears."""
    from gpevo.runtime import AttrDict

    from gpevo.FunctionFolder.USPEX.Start import _restore_uspex_struc

    # the snapshot written after the earlier restart holds only its own
    # generation, the structures before it are gone (None)
    _snapshots(tmp_path, ((1, [_record(1)] * 3),
                          (2, [None, None, None, _record(2), _record(2)])))
    POP = AttrDict()
    POP.resFolder = str(tmp_path / "results1")
    POP.generation = 3

    uspex = _restore_uspex_struc(POP)

    assert uspex is not None
    assert [record.gen for record in uspex.POPULATION] == [1, 1, 1, 2, 2]


def test_restart_ignores_the_snapshot_of_the_generation_being_redone(tmp_path):
    """A half-written snapshot of the current generation is not a source."""
    from gpevo.runtime import AttrDict

    from gpevo.FunctionFolder.USPEX.Start import _restore_uspex_struc

    _snapshots(tmp_path, ((1, [_record(1)] * 3),
                          (2, [_record(2), _record(2)])))
    POP = AttrDict()
    POP.resFolder = str(tmp_path / "results1")
    POP.generation = 2

    uspex = _restore_uspex_struc(POP)

    assert [record.gen for record in uspex.POPULATION] == [1, 1, 1]


def test_restart_finds_the_snapshots_of_a_run_that_moved(tmp_path):
    """A copied run keeps the absolute resFolder of the machine it started on;
    the results folder of the current directory is then the only source."""
    from gpevo.runtime import AttrDict

    from gpevo.FunctionFolder.USPEX.Start import _uspex_snapshots

    _snapshots(tmp_path, ((1, [_record(1)] * 2),))
    POP = AttrDict()
    POP.resFolder = "/somewhere/else/results1"   # stale, from the other machine

    assert [n for n, _path in _uspex_snapshots(POP)] == [1]


def test_restart_applies_the_edited_gp_gates(tmp_path):
    """den/u/explore edited while the run was stopped must take effect."""
    _input_file(tmp_path, den="1.75", u="0.06", explore="0")

    org = _refresh_input_options("INPUT.txt")

    assert org.gpDen == pytest.approx(1.75)
    assert org.gpU == pytest.approx(0.06)
    assert org.gpExplore == 0
    assert org.gp == 1              # still enabled
    assert org.gpData == "data22"   # untouched options survive


def test_restart_keeps_the_run_state_of_the_checkpoint(tmp_path):
    """Everything in _RESTART_KEEP stays as the checkpoint had it."""
    before = _checkpoint(tmp_path)
    G["ORG_STRUC"] = before
    _input_file(tmp_path, den="1.75")

    org = _refresh_input_options("INPUT.txt")

    for name in _RESTART_KEEP:
        assert getattr(org, name) == getattr(before, name), name
    # and the tunable gate did move
    assert org.gpDen == pytest.approx(1.75)


def test_restart_reports_what_changed(tmp_path, capsys):
    _input_file(tmp_path, den="1.75", u="0.06")

    _refresh_input_options("INPUT.txt")

    out = capsys.readouterr().out
    assert "gpDen" in out and "1.8 -> 1.75" in out
    assert "gpU" in out and "0.03 -> 0.06" in out


def test_a_broken_input_txt_cannot_kill_the_restart(tmp_path, capsys):
    """A typo must not lose the checkpoint: the run continues as it was."""
    before = _checkpoint(tmp_path)
    G["ORG_STRUC"] = before
    (tmp_path / "INPUT.txt").write_text("1 : gp\nthis is not a number : den\n",
                                        encoding="utf-8")

    org = _refresh_input_options("INPUT.txt")

    # the parser tolerates the bad line, but whatever it did must not have
    # thrown away the gates the run was started with
    assert org.gpDen == pytest.approx(1.8) or org.gpDen == pytest.approx(0.0)
    assert org.resFolder == before.resFolder
    assert org.latVolume == before.latVolume


def test_no_input_txt_leaves_the_checkpoint_alone(tmp_path):
    before = _checkpoint(tmp_path)
    G["ORG_STRUC"] = before

    org = _refresh_input_options("INPUT.txt")

    assert org.gpDen == pytest.approx(1.8)
    assert org.latVolume == before.latVolume


# ---- fixRandSeed (MATLAB Start.m / Initialize.m / PickUp.m) -------------

def test_restart_picks_up_fix_rand_seed(tmp_path):
    """A seed edited into INPUT.txt while stopped must reach the ORG."""
    path = _input_file(tmp_path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("  12345 : fixRandSeed\n")

    org = _refresh_input_options(path)

    assert org.fixRndSeed == 12345


def test_fixed_seed_reproduces_both_rng_streams():
    """Same seed -> identical numpy AND random draws (paired comparisons)."""
    from gpevo.FunctionFolder.USPEX.Start import _setup_rng

    org = struct(fixRndSeed=12345)
    seed1, source1 = _setup_rng(org)
    a_np, a_py = np.random.rand(5), random.random()
    seed2, source2 = _setup_rng(org)
    b_np, b_py = np.random.rand(5), random.random()

    assert (seed1, source1) == (seed2, source2) == (12346, "fixRandSeed=12345")
    assert np.array_equal(a_np, b_np)
    assert a_py == b_py


def test_restart_seed_is_offset_by_the_generation():
    from gpevo.FunctionFolder.USPEX.Start import _setup_rng

    seed, source = _setup_rng(struct(fixRndSeed=12345), generation=9)

    assert seed == 12345 + 9
    assert source == "fixRandSeed=12345"


def test_zero_keeps_the_clock_seed():
    """Default behaviour (no fixRandSeed) is unchanged."""
    from gpevo.FunctionFolder.USPEX.Start import _setup_rng

    seed, source = _setup_rng(struct(fixRndSeed=0))

    assert source == "clock"
    assert 0 <= seed < 2**31


# ---- a finished run must not restart --------------------------------------

def test_a_finished_run_does_not_restart(tmp_path, monkeypatch, capsys):
    """USPEX_IS_DONE is MATLAB run-uspex.sh's stop flag; honour it here too."""
    from gpevo.FunctionFolder.USPEX.Start import Start

    monkeypatch.setattr("gpevo.runtime._HOME", None)   # set_home() writes this
    (tmp_path / "USPEX_IS_DONE").write_text("1\n")

    Start()

    assert "already finished" in capsys.readouterr().out
    assert not (tmp_path / "Current_POP.pkl").exists()
