"""results<N>/Parameters.txt is the INPUT.txt snapshot (MATLAB Start.m:66).

MATLAB copies INPUT.txt into the fresh result folder once, at start-up, and
ignores the shell return code.  A restart must NOT overwrite it: the file
records the settings the run STARTED with.  This matters more in gpevo, whose
restart branch re-reads INPUT.txt and can therefore run with edited options.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from gpevo.runtime import G, struct
from gpevo.FunctionFolder.USPEX import Start as S


@pytest.fixture
def isolated_G():
    saved = dict(G)
    yield
    G.clear()
    G.update(saved)


def _org(tmp_path, res):
    return struct(
        homePath=str(tmp_path), resFolder=str(res), log_file="OUTPUT.txt",
        dimension=3, molecule=1, varcomp=0, repeatForStatistics=1,
        fixRndSeed=0, numParallelCalcs=1, specificFolder="Specific",
        tournament=np.array([1.0]), bestFrac=0.5, populationSize=4,
    )


def _patch_common(monkeypatch):
    """Neutralise everything after the branch that would run a real search."""
    import gpevo.runtime as runtime

    monkeypatch.setattr(S, "set_home", lambda path: None)
    monkeypatch.setattr(S, "configure_log", lambda path: None)
    monkeypatch.setattr(S, "CreateCalcFolder", lambda *a: None)
    monkeypatch.setattr(S, "_setup_rng", lambda *a, **kw: (0, "test"))
    monkeypatch.setattr(S, "_im",
                        lambda name: types.SimpleNamespace(EA_310=lambda: None))
    monkeypatch.setattr(runtime, "safesave", lambda *a, **kw: None)
    monkeypatch.setattr(
        "gpevo.FunctionFolder.USPEX.src.pickAntiSeeds.pickAntiSeeds",
        lambda: None)


# --------------------------------------------------------------------------
#  the helper itself
# --------------------------------------------------------------------------

def test_copy_is_byte_for_byte(tmp_path):
    text = "1 : optType  % 中文注释保留\n\n**********\n0 : fixRandSeed\n"
    (tmp_path / "INPUT.txt").write_text(text, encoding="utf-8")
    S._save_parameters(str(tmp_path / "INPUT.txt"), str(tmp_path))
    assert (tmp_path / "Parameters.txt").read_bytes() == \
           (tmp_path / "INPUT.txt").read_bytes()


def test_existing_snapshot_is_overwritten_like_cp(tmp_path):
    (tmp_path / "INPUT.txt").write_text("NEW\n", encoding="utf-8")
    (tmp_path / "Parameters.txt").write_text("OLD\n", encoding="utf-8")
    S._save_parameters(str(tmp_path / "INPUT.txt"), str(tmp_path))
    assert (tmp_path / "Parameters.txt").read_text(encoding="utf-8") == "NEW\n"


def test_missing_input_only_warns(tmp_path, capsys):
    S._save_parameters(str(tmp_path / "missing.txt"), str(tmp_path))
    assert "could not save Parameters.txt" in capsys.readouterr().out
    assert not (tmp_path / "Parameters.txt").exists()


def test_empty_res_folder_is_a_silent_noop(tmp_path, capsys):
    S._save_parameters(str(tmp_path / "missing.txt"), "")
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
#  Start(): fresh run writes it once, restart leaves it alone
# --------------------------------------------------------------------------

def test_fresh_run_writes_the_snapshot_once(tmp_path, monkeypatch, isolated_G):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "INPUT.txt").write_text("11 : optType\n", encoding="utf-8")
    res = tmp_path / "results1"
    res.mkdir()
    org = _org(tmp_path, res)
    monkeypatch.setattr(S, "createORGStruc", lambda f: org)

    def fake_initialize():
        G["POP_STRUC"] = struct()

    monkeypatch.setattr(S, "Initialize", fake_initialize)
    _patch_common(monkeypatch)
    calls = []
    original = S._save_parameters

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(S, "_save_parameters", spy)

    S.Start()

    assert calls == [("INPUT.txt", str(res))]
    assert (res / "Parameters.txt").read_bytes() == \
           (tmp_path / "INPUT.txt").read_bytes()


def test_restart_does_not_overwrite_the_snapshot(tmp_path, monkeypatch,
                                                 isolated_G):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "INPUT.txt").write_text("CHANGED INPUT\n", encoding="utf-8")
    res = tmp_path / "results1"
    res.mkdir()
    (res / "Parameters.txt").write_text("ORIGINAL SNAPSHOT", encoding="utf-8")
    (tmp_path / "Current_POP.pkl").write_bytes(b"fake")
    (tmp_path / "Current_ORG.pkl").write_bytes(b"fake")

    org = _org(tmp_path, res)
    pop = struct(resFolder=str(res), generation=3, POPULATION=[],
                 ranking=[], bad_rank=0, DoneOrder=[])
    import gpevo.runtime as runtime
    monkeypatch.setattr(
        runtime, "safeload",
        lambda path: pop if "Current_POP" in str(path) else org)
    monkeypatch.setattr(S, "_refresh_input_options", lambda f: org)
    monkeypatch.setattr(S, "_restore_uspex_struc", lambda p: None)
    _patch_common(monkeypatch)
    calls = []
    monkeypatch.setattr(S, "_save_parameters",
                        lambda *a, **kw: calls.append(a))

    S.Start()

    assert calls == []
    assert (res / "Parameters.txt").read_text(encoding="utf-8") == \
           "ORIGINAL SNAPSHOT"
