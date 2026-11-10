"""Tests for the finishing block Finish() appends to OUTPUT.txt.

MATLAB's Finish.m prints the relaxation count and a ``Job Finished at``
timestamp with ``fprintf(fp, ...)``; the reference line is 64 characters wide
because the stamp sits in a ``%30s`` field.  Nothing here touches a live run
directory; everything uses ``tmp_path``.
"""
from __future__ import annotations

import re

import pytest

from gpevo.runtime import G, struct
from gpevo.log_utils import configure_log, close_log
from gpevo.FunctionFolder.USPEX.src.Finish import Finish, _job_finished_stamp


_RELAXATIONS = "This calculation runs  610 structural relaxations"
_STAMP = re.compile(r"^\d{2}-[A-Z][a-z]{2}-\d{4} \d{2}:\d{2}:\d{2}$")


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Set up the minimal ORG/POP pair Finish() needs and capture OUTPUT.txt."""
    monkeypatch.chdir(tmp_path)          # Finish() also writes ./USPEX_IS_DONE
    log = tmp_path / "OUTPUT.txt"
    try:
        configure_log(str(log))         # patches builtins.print -> restore below

        ORG = struct()
        ORG.resFolder = str(tmp_path)
        ORG.numGenerations = 50
        POP = struct()
        POP.bodyCount = 610
        POP.generation = 50
        ind = struct()
        ind.Enthalpies = [-1131.8]
        POP.POPULATION = [ind]
        G["ORG_STRUC"] = ORG
        G["POP_STRUC"] = POP
        yield log
    finally:
        close_log()


def test_stamp_uses_english_month_whatever_the_locale():
    """Python's %b follows LC_TIME ("10月"), MATLAB's datestr does not."""
    assert _STAMP.match(_job_finished_stamp()), _job_finished_stamp()


def test_finish_writes_relaxation_count(run):
    with pytest.raises(SystemExit):
        Finish()
    assert _RELAXATIONS in run.read_text()


def test_finish_writes_job_finished_line(run):
    with pytest.raises(SystemExit):
        Finish()
    line = [l for l in run.read_text().splitlines() if "Job Finished" in l]
    assert len(line) == 1
    # Finish.m:20 formats the stamp with %30s, so the line is 34 + 30 = 64 wide.
    assert line[0].startswith("            Job Finished at")
    assert len(line[0]) == 64
    assert _STAMP.match(line[0][-20:])


def test_finished_block_follows_the_relaxation_count(run):
    with pytest.raises(SystemExit):
        Finish()
    lines = run.read_text().splitlines()
    assert _RELAXATIONS in lines[-3]                 # blank line in between
    assert lines[-1].strip().startswith("Job Finished at")


def test_finish_parks_the_generation_beyond_the_limit(run, tmp_path):
    """Finish.m saves generation = numGenerations+1 so a restart quits at once."""
    from gpevo.runtime import safeload

    with pytest.raises(SystemExit):
        Finish()

    assert safeload("Current_POP.pkl").generation == 51
    assert (tmp_path / "USPEX_IS_DONE").exists()
