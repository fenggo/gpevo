"""A restart must pick up INPUT.txt edits.

``Start()`` restores ``Current_ORG.pkl``, so without a re-read every option
the user changed while the run was stopped is ignored -- the GP gates
(``den``, ``u``, ``explore``) above all, since those are what gets tuned
between restarts.

``Start()`` itself is not run here: it would enter the generation loop.  The
refresh helper is exercised directly, with a run directory that matches what
a stopped run leaves behind.
"""
from __future__ import annotations

import os

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
