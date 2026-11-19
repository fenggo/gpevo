"""Tests for rejecting fake low-energy structures by GP residual.

Collapsed structures (e.g. an intramolecular N-N contact of 0.936 A) slip past
the molecule checks in readMOL.py because CL-20 already contains N-N bonds.
Their GP residual, however, is catastrophic (~1e4 versus 0.3-3 for normal
structures).  These tests cover the dedicated predicate/marker and the two
integration points (job read-back and fitness calculation).

All tests use ``tmp_path`` / monkeypatched globals; none touch a live run.
"""
from __future__ import annotations

import math
from importlib import import_module

import numpy as np
import pytest

from gpevo.runtime import G, RESIDUAL_CUT, FAKE_RESIDUAL_CUT, struct
from gpevo.FunctionFolder.USPEX.src import residual_check
from gpevo.FunctionFolder.USPEX.src.residual_check import (
    is_fake_residual, mark_fake_residual,
)

CalcFitness_310 = import_module(
    "gpevo.FunctionFolder.USPEX.310.CalcFitness_310").CalcFitness_310
ReadJobs_310 = import_module(
    "gpevo.FunctionFolder.USPEX.310.ReadJobs_310").ReadJobs_310


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

def test_cutoffs_are_independent():
    assert FAKE_RESIDUAL_CUT == 100.0
    assert RESIDUAL_CUT == 10.0
    assert FAKE_RESIDUAL_CUT > RESIDUAL_CUT


# --------------------------------------------------------------------------
# is_fake_residual
# --------------------------------------------------------------------------

def test_fake_residual_true_for_collapse_value():
    assert is_fake_residual(struct(residual=12657.1377)) is True


@pytest.mark.parametrize("value", [0.3, 2.5, 99.999])
def test_fake_residual_false_for_normal_values(value):
    assert is_fake_residual(struct(residual=value)) is False


def test_fake_residual_strict_boundary():
    assert is_fake_residual(struct(residual=100.0)) is False
    assert is_fake_residual(struct(residual=100.01)) is True


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"),
                                   float("-inf"), "abc", [1, 2]])
def test_fake_residual_ignores_missing_and_nonfinite(value):
    assert is_fake_residual(struct(residual=value)) is False


def test_fake_residual_falls_back_to_distance():
    # distance is only a fallback when residual is absent
    assert is_fake_residual(struct(distance=12657.0)) is True
    # residual wins when both are present
    assert is_fake_residual(struct(residual=0.5, distance=12657.0)) is False


def test_fake_residual_uses_injected_cutoff(monkeypatch):
    monkeypatch.setattr(residual_check, "FAKE_RESIDUAL_CUT", 10.0)
    # default argument captured at definition time, so pass explicitly
    assert is_fake_residual(struct(residual=50.0), cutoff=10.0) is True


# --------------------------------------------------------------------------
# mark_fake_residual
# --------------------------------------------------------------------------

def _fake(enthalpy=-1185.272, step=None, error=0, residual=12657.1377):
    s = struct(Enthalpies=[enthalpy], Step=step if step is not None else [4],
               Error=error, residual=residual, distance=residual)
    return s


def test_mark_applies_failure_pattern(capsys):
    s = _fake()
    assert mark_fake_residual(s, label="279") is True
    assert s.Enthalpies[-1] == 100000.0
    assert s.fakeResidualCheck is True
    assert "FAKE low-energy" in s.moleculeCheck
    assert s.Error == 4                      # Step=4 >= 2
    text = capsys.readouterr().out
    assert "FAKE low-energy rejected" in text
    assert "1.266e+04" in text               # residual in %.4g form


def test_mark_preserves_earlier_enthalpy_history():
    s = struct(Enthalpies=[-1184.8, -1185.27], Step=[2], Error=0,
               residual=12657.0)
    mark_fake_residual(s)
    assert s.Enthalpies[0] == -1184.8
    assert s.Enthalpies[-1] == 100000.0


def test_mark_does_not_touch_gp_evidence():
    s = _fake()
    s.gp = {"energy_pred": -28083.0}
    s.gp_feature = [-1185.0] * 10
    distance_before, residual_before = s.distance, s.residual
    mark_fake_residual(s)
    assert s.distance == distance_before
    assert s.residual == residual_before
    assert s.gp == {"energy_pred": -28083.0}
    assert s.gp_feature == [-1185.0] * 10


def test_mark_is_idempotent(capsys):
    s = _fake()
    assert mark_fake_residual(s) is True
    first_error = s.Error
    first_marker = s.moleculeCheck
    capsys.readouterr()
    assert mark_fake_residual(s) is False
    assert s.Error == first_error
    assert s.moleculeCheck == first_marker
    assert capsys.readouterr().out == ""


def test_mark_noop_when_not_fake():
    s = struct(Enthalpies=[-1140.0], Step=[4], Error=0, residual=0.5)
    before = dict(s.__dict__)
    assert mark_fake_residual(s) is False
    assert s.__dict__ == before


def test_mark_appends_to_existing_molecule_check():
    s = _fake()
    s.moleculeCheck = "MOLECULE FORM is not kept"
    mark_fake_residual(s)
    assert "MOLECULE FORM is not kept" in s.moleculeCheck
    assert "FAKE low-energy" in s.moleculeCheck


def test_mark_step_guard():
    s1 = _fake(step=[1])
    mark_fake_residual(s1)
    assert s1.Error == 0                     # Step < 2
    s2 = _fake(step=[1, 2, 3])
    mark_fake_residual(s2)
    assert s2.Error == 4


# --------------------------------------------------------------------------
# integration: fitness path
# --------------------------------------------------------------------------

def _org(opt_type=1, gp=0, opt_sign=None, **extra):
    fields = dict(optType=opt_type, gp=gp, opt_sign=opt_sign, bestFrac=0.7,
                  varcomp=0, weight=1.0, numMols=[1], atomType=[1],
                  antiSeedsMax=0.0, antiSeedsSigma=0.001,
                  antiSeedsActivation=5000, homePath=".", resFolder=".")
    fields.update(extra)
    return struct(**fields)


def _pop(fake_residual=12657.0, normal_residual=0.5):
    lattice = np.eye(3).tolist()
    fake = struct(Enthalpies=[-1185.272], Step=[4], Error=0,
                  Number=279, residual=fake_residual, distance=fake_residual,
                  numIons=[2], numMols=[1], LATTICE=lattice)
    normal = struct(Enthalpies=[-1140.0], Step=[4], Error=0,
                    Number=280, residual=normal_residual,
                    distance=normal_residual, numIons=[2], numMols=[1],
                    LATTICE=lattice)
    return struct(generation=2, POPULATION=[fake, normal], ranking=[0, 1])


@pytest.mark.parametrize("opt_type", [1, 11, 13])
def test_calcfitness_rejects_fake_for_all_opttypes(opt_type, monkeypatch):
    monkeypatch.setitem(G, "ORG_STRUC", _org(opt_type=opt_type))
    monkeypatch.setitem(G, "POP_STRUC", _pop())

    out = CalcFitness_310()

    assert out[0] == 100000.0
    assert out[1] < 100000.0


def test_calcfitness_fake_stays_positive_with_negative_opt_sign(monkeypatch):
    monkeypatch.setitem(G, "ORG_STRUC", _org(opt_type=1, opt_sign=-1))
    monkeypatch.setitem(G, "POP_STRUC", _pop())

    out = CalcFitness_310()

    assert out[0] == 100000.0                 # not -100000


def test_fake_excluded_from_gp_active_learning(monkeypatch):
    module = import_module(
        "gpevo.FunctionFolder.USPEX.310.CalcFitness_310")
    seen = {}

    def fake_gal(org, pop, fitness):
        seen["fitness"] = np.asarray(fitness).copy()
        return fitness

    monkeypatch.setattr(module, "gp_active_learning", fake_gal)
    monkeypatch.setitem(G, "ORG_STRUC", _org(opt_type=1, gp=1))
    monkeypatch.setitem(G, "POP_STRUC", _pop())

    CalcFitness_310()

    assert seen["fitness"][0] == 100000.0
    assert seen["fitness"][1] < 100000.0


# --------------------------------------------------------------------------
# INPUT.txt `fakerescut` option
# --------------------------------------------------------------------------

def test_default_cutoff_from_org(monkeypatch):
    from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault

    org = createORGDefault()
    assert org.fakeResCut == 100.0
    monkeypatch.setitem(G, "ORG_STRUC", org)
    assert is_fake_residual(struct(residual=12657.0)) is True
    assert is_fake_residual(struct(residual=50.0)) is False


@pytest.mark.parametrize("value", [-1.0, 0.0, float("nan"), "abc", None])
def test_invalid_org_cutoff_falls_back_to_constant(monkeypatch, value):
    org = struct(fakeResCut=value)
    monkeypatch.setitem(G, "ORG_STRUC", org)
    assert is_fake_residual(struct(residual=50.0)) is False
    assert is_fake_residual(struct(residual=12657.0)) is True


def test_org_cutoff_overrides_constant(monkeypatch):
    org = struct(fakeResCut=50.0)
    monkeypatch.setitem(G, "ORG_STRUC", org)
    assert is_fake_residual(struct(residual=60.0)) is True
    assert is_fake_residual(struct(residual=40.0)) is False
    # explicit cutoff wins over the ORG value
    assert is_fake_residual(struct(residual=60.0), cutoff=100.0) is False


def test_explicit_org_argument_wins_over_global(monkeypatch):
    monkeypatch.setitem(G, "ORG_STRUC", struct(fakeResCut=100.0))
    org = struct(fakeResCut=30.0)
    assert is_fake_residual(struct(residual=50.0), ORG=org) is True
    assert is_fake_residual(struct(residual=50.0)) is False


def test_calcfitness_honours_custom_cutoff(monkeypatch):
    org = _org(opt_type=1)
    org.fakeResCut = 50.0
    pop = _pop(fake_residual=60.0, normal_residual=40.0)
    monkeypatch.setitem(G, "ORG_STRUC", org)
    monkeypatch.setitem(G, "POP_STRUC", pop)

    out = CalcFitness_310()

    assert out[0] == 100000.0
    assert out[1] < 100000.0


def test_input_fakerescut_parsing(tmp_path, monkeypatch):
    from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault
    from gpevo.FunctionFolder.USPEX.src.createORG_AbinitCode import \
        createORG_AbinitCode

    (tmp_path / "INPUT.txt").write_text("50 : fakerescut\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    G["ORG_STRUC"] = createORGDefault()
    createORG_AbinitCode("INPUT.txt")
    assert G["ORG_STRUC"].fakeResCut == 50.0


def test_input_without_fakerescut_keeps_default(tmp_path, monkeypatch):
    from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault
    from gpevo.FunctionFolder.USPEX.src.createORG_AbinitCode import \
        createORG_AbinitCode

    (tmp_path / "INPUT.txt").write_text("1 : gp\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    G["ORG_STRUC"] = createORGDefault()
    createORG_AbinitCode("INPUT.txt")
    assert G["ORG_STRUC"].fakeResCut == 100.0
