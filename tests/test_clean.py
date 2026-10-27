"""Tests for ``gpevo clean``.

Everything is built inside pytest's ``tmp_path``: the command deletes things,
and the real run directories carry checkpoints that must never be touched from
a test.
"""
from __future__ import annotations

import os

import pytest

from gpevo.tools.clean import clean


def _touch(path, content=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fd:
        fd.write(content)
    return path


@pytest.fixture
def run_dir(tmp_path):
    """A run directory with scratch, state files and assets side by side."""
    root = str(tmp_path)
    # assets -- must survive every invocation
    _touch(os.path.join(root, "INPUT.txt"))
    _touch(os.path.join(root, "MOL_1"))
    _touch(os.path.join(root, "Specific", "ffield"))
    _touch(os.path.join(root, "Seeds", "POSCAR"))
    _touch(os.path.join(root, "results1", "gatheredPOSCARS"))
    _touch(os.path.join(root, "results1", "Individuals.traj"))
    _touch(os.path.join(root, "results1", "generation1", "POP_STRUC.pkl"))
    _touch(os.path.join(root, "results1", "AuxiliaryFiles", "keep.txt"))
    _touch(os.path.join(root, "data", "feature.csv"))
    # scratch -- what a plain clean removes
    _touch(os.path.join(root, "CalcFold1", "input"))
    _touch(os.path.join(root, "CalcFoldTemp", "input"))
    _touch(os.path.join(root, "AntiSeeds", "POSCAR"))
    _touch(os.path.join(root, "Current_ORG.pkl"))
    _touch(os.path.join(root, "Current_POP.pkl"))
    _touch(os.path.join(root, "leftover.mat"))
    _touch(os.path.join(root, "old.backup"))
    _touch(os.path.join(root, "still_running"))
    _touch(os.path.join(root, "USPEX_IS_DONE"))
    _touch(os.path.join(root, "Warnings"))
    # per-structure scratch of the reference (labelling) runs
    _touch(os.path.join(root, "results1", "7", "siesta.out"))
    _touch(os.path.join(root, "results1", "12", "POSCAR.12"))
    return root


def _exists(root, *parts):
    return os.path.exists(os.path.join(root, *parts))


def test_clean_removes_scratch_and_keeps_assets(run_dir, capsys):
    clean(path=run_dir)

    for gone in ("CalcFold1", "CalcFoldTemp", "AntiSeeds", "Current_ORG.pkl",
                 "Current_POP.pkl", "leftover.mat", "old.backup",
                 "still_running", "USPEX_IS_DONE", "Warnings"):
        assert not _exists(run_dir, gone), gone
    # the per-structure labelling folders go, everything else in results* stays
    assert not _exists(run_dir, "results1", "7")
    assert not _exists(run_dir, "results1", "12")
    for kept in ("INPUT.txt", "MOL_1", "Specific", "Seeds", "data", "results1"):
        assert _exists(run_dir, kept), kept
    assert _exists(run_dir, "results1", "gatheredPOSCARS")
    assert _exists(run_dir, "results1", "Individuals.traj")
    assert _exists(run_dir, "results1", "generation1", "POP_STRUC.pkl")
    assert _exists(run_dir, "results1", "AuxiliaryFiles", "keep.txt")
    assert "removed" in capsys.readouterr().out


def test_all_also_removes_results(run_dir):
    clean(path=run_dir, all_=True)

    assert not _exists(run_dir, "results1")
    assert _exists(run_dir, "INPUT.txt")
    assert _exists(run_dir, "Specific")
    assert _exists(run_dir, "data")


def test_dry_run_touches_nothing(run_dir, capsys):
    removed = clean(path=run_dir, dry_run=True)

    assert removed > 0
    assert _exists(run_dir, "CalcFold1")
    assert _exists(run_dir, "Current_ORG.pkl")
    out = capsys.readouterr().out
    assert "would remove" in out
    assert "would be removed" in out


def test_refuses_a_directory_that_is_not_a_run(tmp_path):
    with pytest.raises(SystemExit) as exc:
        clean(path=str(tmp_path))
    assert "does not look like a USPEX run directory" in str(exc.value)


def test_force_cleans_a_directory_without_the_usual_markers(tmp_path):
    _touch(os.path.join(str(tmp_path), "Current_POP.pkl"))

    clean(path=str(tmp_path), force=True)

    assert not os.path.exists(os.path.join(str(tmp_path), "Current_POP.pkl"))


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(SystemExit) as exc:
        clean(path=str(tmp_path / "nope"))
    assert "not a directory" in str(exc.value)


def test_cli_dispatches_clean(monkeypatch, tmp_path):
    from gpevo import cli

    calls = []

    def fake_clean(**kwargs):
        calls.append(kwargs)

    import gpevo.tools.clean as clean_module
    monkeypatch.setattr(clean_module, "clean", fake_clean)

    cli.main(["clean", "--all", "--dry-run"])

    assert calls == [{"path": ".", "all_": True, "force": False,
                      "dry_run": True}]


def test_cli_without_arguments_starts_the_search(monkeypatch):
    from gpevo import cli
    from gpevo.FunctionFolder.USPEX import Start as start_module

    started = []
    monkeypatch.setattr(start_module, "Start", lambda: started.append(True))

    cli.main([])
    cli.main(["run"])

    assert started == [True, True]
