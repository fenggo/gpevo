"""Fresh GP runs keep Specific/ffield.json in sync with the data copy."""
from __future__ import annotations

import json

import pytest

from gpevo.FunctionFolder.USPEX.src.gp_ml import sync_ffield_json
from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault


DATA_A = {"p": {"bo1_C-O": 1.0}, "m": [1, 2]}
DATA_B = {"p": {"bo1_C-O": 2.0}, "m": [1, 2]}


def _org(tmp_path, gp=1, gp_data="data22"):
    org = createORGDefault()
    org.homePath = str(tmp_path)
    org.gp = gp
    org.gpData = gp_data
    org.specificFolder = "Specific"
    return org


def _write(path, obj, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, **kw), encoding="utf-8")


def test_disabled_gp_does_nothing(tmp_path):
    _write(tmp_path / "data22" / "ffield.json", DATA_B)
    _write(tmp_path / "Specific" / "ffield.json", DATA_A)
    org = _org(tmp_path, gp=0)
    assert sync_ffield_json(org) == "disabled"
    assert json.loads((tmp_path / "Specific" / "ffield.json").read_text()) == DATA_A
    assert not (tmp_path / "Specific" / "ffield.json.bak").exists()


def test_equal_json_different_format_is_noop(tmp_path):
    # Same content, different key order/indent -> semantic equality, no backup.
    _write(tmp_path / "data22" / "ffield.json", DATA_A)
    _write(tmp_path / "Specific" / "ffield.json", dict(reversed(list(DATA_A.items()))),
           indent=4)
    before = (tmp_path / "Specific" / "ffield.json").read_text()
    assert sync_ffield_json(_org(tmp_path)) == "equal"
    assert (tmp_path / "Specific" / "ffield.json").read_text() == before
    assert not (tmp_path / "Specific" / "ffield.json.bak").exists()


def test_differing_specific_is_backed_up_and_replaced(tmp_path):
    _write(tmp_path / "data22" / "ffield.json", DATA_B)
    _write(tmp_path / "Specific" / "ffield.json", DATA_A)
    assert sync_ffield_json(_org(tmp_path)) == "synced"
    assert json.loads((tmp_path / "Specific" / "ffield.json").read_text()) == DATA_B
    assert json.loads((tmp_path / "Specific" / "ffield.json.bak").read_text()) == DATA_A


def test_missing_specific_copy_is_skipped(tmp_path):
    _write(tmp_path / "data22" / "ffield.json", DATA_B)
    assert sync_ffield_json(_org(tmp_path)) == "no-specific"


def test_missing_data_copy_is_skipped(tmp_path):
    _write(tmp_path / "Specific" / "ffield.json", DATA_A)
    assert sync_ffield_json(_org(tmp_path)) == "no-data"


def test_invalid_json_is_skipped_without_touching_files(tmp_path):
    _write(tmp_path / "data22" / "ffield.json", DATA_B)
    dst = tmp_path / "Specific" / "ffield.json"
    _write(dst, DATA_A)
    dst.write_text("{not valid json")
    assert sync_ffield_json(_org(tmp_path)) == "error"
    assert dst.read_text() == "{not valid json"
    assert not (tmp_path / "Specific" / "ffield.json.bak").exists()
