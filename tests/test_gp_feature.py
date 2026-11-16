"""INPUT.txt `f` / `feature`: feature-vector selector for the GP step.

Only optType 1/11/12 with `1 : gp` honour the option -- those rank on a GP
prediction (energy / density).  optType 13 ranks on the column-weighted 10-D
distance and is pinned to f=1, so changing f cannot break its fitness.
"""
from __future__ import annotations

import pytest

from gpevo.runtime import G
from gpevo.FunctionFolder.USPEX.src import gp_ml
from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault


def _org_from_input(tmp_path, text):
    from gpevo.FunctionFolder.USPEX.src.createORG_AbinitCode import (
        createORG_AbinitCode)

    (tmp_path / "INPUT.txt").write_text(text)
    G["ORG_STRUC"] = createORGDefault()
    createORG_AbinitCode(str(tmp_path / "INPUT.txt"))
    return G["ORG_STRUC"]


# ── INPUT.txt → ORG ─────────────────────────────────────────────────────

@pytest.mark.parametrize("line", ["2 : f", "2 : feature", "f : 2", "feature : 2"])
def test_f_and_feature_are_both_accepted(tmp_path, line):
    org = _org_from_input(tmp_path, "1 : gp\n" + line + "\n")
    assert org.gpFeature == 2
    assert gp_ml.gp_feature(org) == 2


def test_f_defaults_to_1(tmp_path):
    org = _org_from_input(tmp_path, "1 : gp\n")
    assert org.gpFeature == 1
    assert gp_ml.gp_feature(org) == 1


def test_f_non_numeric_falls_back_to_1(tmp_path):
    org = _org_from_input(tmp_path, "1 : gp\nbogus : f\n")
    assert org.gpFeature == 1


# ── optType gate ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("opt_type", [1, 11, 12])
def test_prediction_opt_types_can_select_f(opt_type):
    org = createORGDefault()
    org.gp = 1
    org.optType = opt_type
    org.gpFeature = 2
    assert gp_ml.gp_feature(org) == 2


@pytest.mark.parametrize("opt_type", [13, 14, 2])
def test_other_opt_types_are_pinned_to_f1(opt_type):
    org = createORGDefault()
    org.gp = 1
    org.optType = opt_type
    org.gpFeature = 2
    assert gp_ml.gp_feature(org) == 1


def test_gp_off_pins_f_to_1():
    org = createORGDefault()
    org.gp = 0
    org.optType = 11
    org.gpFeature = 2
    assert gp_ml.gp_feature(org) == 1


# ── threaded into gp() / pred() / scheduler CLI ──────────────────────────

def test_run_gp_passes_f(monkeypatch):
    org = createORGDefault()
    org.gp = 1
    org.optType = 11
    org.gpFeature = 2
    captured = {}

    def fake_call(name, **kwargs):
        captured.update(kwargs)
        return True, {"density_gp": 1.9}

    monkeypatch.setattr(gp_ml, "_call", fake_call)
    monkeypatch.setattr(gp_ml, "_backend_ready", lambda backend: True)
    gp_ml.run_gp(org)
    assert captured["f"] == 2


def test_run_gp_forces_f1_for_opt_type_13(monkeypatch):
    org = createORGDefault()
    org.gp = 1
    org.optType = 13
    org.gpFeature = 2
    captured = {}

    def fake_call(name, **kwargs):
        captured.update(kwargs)
        return True, {"density_gp": 1.9}

    monkeypatch.setattr(gp_ml, "_call", fake_call)
    monkeypatch.setattr(gp_ml, "_backend_ready", lambda backend: True)
    gp_ml.run_gp(org)
    assert captured["f"] == 1


def test_parse_gp_params_carries_f():
    org = createORGDefault()
    org.gp = 1
    org.optType = 11
    org.gpFeature = 2
    assert gp_ml.parse_gp_params(org)["f"] == 2


def test_code_exec_line_passes_f(monkeypatch):
    from gpevo.FunctionFolder.Submission import submitJob

    monkeypatch.setattr(gp_ml, "is_gp_step", lambda ORG, step: True)
    monkeypatch.setattr(gp_ml, "_ncpu", lambda ORG: 24)
    org = createORGDefault()
    org.gp = 1
    org.optType = 11
    org.gpFeature = 2
    line = submitJob._code_exec_line(org, None, 1)
    assert "--f=2" in line


def test_uspexkit_cli_exposes_f():
    from gpevo.uspexkit.cli import _build_parser

    args = _build_parser().parse_args(["gp", "--f=2"])
    assert args.f == 2
