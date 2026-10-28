"""Tests for storing the GP residual so optType 13 (distance) can use it.

The distance optType 13 optimises is the residual ``gp()``/``pred()`` return
(``CalcFitness_310.m:46-53`` reads ``POPULATION(i).distance``).  It is stored
on the individual under both names, and must survive the re-prediction that
follows a training-set update.

Nothing here touches a live run directory; everything uses ``tmp_path``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gpevo.runtime import G, struct
from gpevo.FunctionFolder.USPEX.src.gp_ml import (
    GP_OUTPUT_HEADER, apply_gp_metrics, apply_pred_metrics,
)
from gpevo.FunctionFolder.USPEX.src.WriteGenerationOutput_310 import _gp_columns
from gpevo.FunctionFolder.USPEX.src.createORGStruc import check_optType_13

# `310` is not a valid identifier, so the module cannot be imported statically
# (inside the package EA_310.py reaches it with a relative import).
from importlib import import_module

CalcFitness_310 = import_module(
    "gpevo.FunctionFolder.USPEX.310.CalcFitness_310").CalcFitness_310


# --------------------------------------------------------------------------
# storing the residual
# --------------------------------------------------------------------------

def test_gp_metrics_store_residual_under_both_names():
    s = struct()
    ok = apply_gp_metrics(s, {
        "residual": 0.42, "density_gp": 1.8, "uncertainty": 0.05,
        "energy_pred": -10.0, "uncertainty_eng": 0.2,
    })
    assert ok
    assert s.residual == pytest.approx(0.42)     # the new dedicated field
    assert s.distance == pytest.approx(0.42)     # the name CalcFitness reads


def test_pred_metrics_store_residual():
    """pred() returns 'residual' too; it used to be dropped."""
    s = struct()
    apply_pred_metrics(s, {
        "residual": 1.7, "density_gp": 1.9, "uncertainty": 0.02,
        "energy_pred": -11.0, "uncertainty_eng": 0.3,
    })
    assert s.residual == pytest.approx(1.7)
    assert s.distance == pytest.approx(1.7)


def test_pred_metrics_without_residual_keeps_the_old_one():
    """A missing residual must not clobber the value the GP step stored."""
    s = struct()
    apply_gp_metrics(s, {"residual": 0.9, "density_gp": 1.8})
    apply_pred_metrics(s, {"density_gp": 1.85, "residual": float("nan")})
    assert s.residual == pytest.approx(0.9)
    assert s.density_gp == pytest.approx(1.85)   # the prediction still updates


def test_non_dict_payload_is_rejected():
    assert apply_gp_metrics(struct(), None) is False
    assert apply_pred_metrics(struct(), "nope") is False


# --------------------------------------------------------------------------
# optType 13
# --------------------------------------------------------------------------

def _org(opt_type=13, gp=1, **extra):
    fields = dict(optType=opt_type, gp=gp, opt_sign=None, bestFrac=0.7,
                  varcomp=0, weight=1.0, numMols=[1], atomType=[1],
                  antiSeedsMax=0.0, antiSeedsSigma=0.001,
                  antiSeedsActivation=5000, homePath=".", resFolder=".")
    fields.update(extra)
    return struct(**fields)


def _individual(residual=None, distance=None, enthalpy=-10.0):
    s = struct(Enthalpies=[enthalpy], numIons=[2], numMols=[1])
    if residual is not None:
        s.residual = residual
    if distance is not None:
        s.distance = distance
    return s


def test_opt_type_13_uses_the_residual(tmp_path, monkeypatch):
    monkeypatch.setitem(G, "ORG_STRUC", _org())
    monkeypatch.setitem(G, "POP_STRUC", struct(
        generation=1, POPULATION=[_individual(residual=2.5),
                                  _individual(residual=0.5)]))

    out = CalcFitness_310()

    assert out[0] == pytest.approx(2.5)
    assert out[1] == pytest.approx(0.5)


def test_opt_type_13_falls_back_to_distance(tmp_path, monkeypatch):
    """Structures evaluated before the residual field existed still work."""
    monkeypatch.setitem(G, "ORG_STRUC", _org())
    monkeypatch.setitem(G, "POP_STRUC", struct(
        generation=1, POPULATION=[_individual(distance=1.25)]))

    assert CalcFitness_310()[0] == pytest.approx(1.25)


def test_opt_type_13_warns_instead_of_silently_nan(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(G, "ORG_STRUC", _org(gp=0))
    monkeypatch.setitem(G, "POP_STRUC", struct(
        generation=3, POPULATION=[_individual(), _individual()]))

    out = CalcFitness_310()

    assert all(math.isnan(v) for v in out)      # .m:53 is NaN as well
    text = capsys.readouterr().out
    assert "optType 13" in text
    assert "`1 : gp`" in text                   # and it says how to fix it


def test_opt_type_13_requires_gp(tmp_path):
    with pytest.raises(RuntimeError, match="1 : gp"):
        check_optType_13(_org(opt_type=13, gp=0))


def test_opt_type_13_with_gp_is_allowed(tmp_path):
    check_optType_13(_org(opt_type=13, gp=1))     # must not raise


def test_other_opt_types_are_not_checked(tmp_path):
    check_optType_13(_org(opt_type=1, gp=0))      # must not raise
    check_optType_13(_org(opt_type=11, gp=0))


# --------------------------------------------------------------------------
# output column
# --------------------------------------------------------------------------

def test_gp_columns_include_residual_and_stay_aligned():
    s = struct(gp={"energy_pred": -10.0, "uncertainty_eng": 0.2,
                   "density_gp": 1.8, "uncertainty": 0.05})
    s.residual = 0.42
    columns = _gp_columns(s)
    assert len(columns) == len(GP_OUTPUT_HEADER)      # header and data agree
    assert "0.4200" in columns


def test_gp_columns_blank_block_has_the_same_width():
    assert _gp_columns(struct()) == " " * len(GP_OUTPUT_HEADER)
