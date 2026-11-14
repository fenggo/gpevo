"""optType 1 ranks on the GP-predicted enthalpy when the GP step is enabled.

Symmetric with ``use_gp_density`` for optType 11: CalcFitness used to take
``Enthalpies[-1]`` (the force-field value Reading carries into the GP step),
which left the fitness on the MLP/ReaxFF enthalpy for most generations.
"""
from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest

from gpevo.runtime import G, struct
from gpevo.FunctionFolder.USPEX.src.gp_ml import gp_enabled

_cf = import_module("gpevo.FunctionFolder.USPEX.310.CalcFitness_310")
CalcFitness_310 = _cf.CalcFitness_310
use_gp_energy = import_module(
    "gpevo.FunctionFolder.USPEX.310.use_gp_energy").use_gp_energy


def _org(gp=1, opt_type=1):
    return struct(optType=opt_type, gp=gp, opt_sign=1, bestFrac=0.7,
                  varcomp=0, weight=1.0, numMols=[1], atomType=[1],
                  antiSeedsMax=0.0, antiSeedsSigma=0.001,
                  antiSeedsActivation=5000, homePath=".", resFolder=".")


def _individual(enthalpy=-100.0, energy_gp=None):
    s = struct(Enthalpies=[enthalpy], numIons=[2], numMols=[1])
    if energy_gp is not None:
        s.energy_gp = energy_gp
    return s


def _run(monkeypatch, org, individuals):
    # Keep the unit under test on CalcFitness itself: the active-learning
    # wrapper and the generation plot are exercised by other suites.
    monkeypatch.setattr(_cf, "gp_active_learning", lambda ORG, POP, f: f)
    monkeypatch.setitem(G, "ORG_STRUC", org)
    monkeypatch.setitem(G, "POP_STRUC", struct(
        generation=0, POPULATION=individuals))
    return CalcFitness_310()


def test_use_gp_energy_prefers_the_gp_prediction():
    assert use_gp_energy(struct(energy_gp=-280.0), -100.0) == -280.0


def test_use_gp_energy_falls_back_on_bad_values():
    assert use_gp_energy(struct(), -100.0) == -100.0
    assert use_gp_energy(struct(energy_gp=float("nan")), -100.0) == -100.0
    assert use_gp_energy(struct(energy_gp="bad"), -100.0) == -100.0


def test_opt_type_1_uses_gp_enthalpy_when_gp_is_on(monkeypatch):
    assert gp_enabled(_org(gp=1))
    out = _run(monkeypatch, _org(gp=1),
               [_individual(enthalpy=-100.0, energy_gp=-280.0)])
    assert out[0] == pytest.approx(-280.0)


def test_opt_type_1_keeps_force_field_enthalpy_when_gp_is_off(monkeypatch):
    out = _run(monkeypatch, _org(gp=0),
               [_individual(enthalpy=-100.0, energy_gp=-280.0)])
    assert out[0] == pytest.approx(-100.0)


def test_opt_type_1_falls_back_when_no_gp_prediction(monkeypatch):
    out = _run(monkeypatch, _org(gp=1),
               [_individual(enthalpy=-100.0, energy_gp=float("nan")),
                _individual(enthalpy=-120.0)])
    assert out[0] == pytest.approx(-100.0)     # nan -> force field
    assert out[1] == pytest.approx(-120.0)     # missing field -> force field


def test_failed_structures_stay_at_100000(monkeypatch):
    """Enthalpies is untouched; the failed-structure gate still wins."""
    out = _run(monkeypatch, _org(gp=1),
               [_individual(enthalpy=100000.0, energy_gp=-280.0)])
    assert out[0] == pytest.approx(100000.0)


def test_opt_type_11_still_uses_gp_density(monkeypatch):
    s = _individual(enthalpy=-100.0, energy_gp=-280.0)
    s.density_gp = 1.9
    s.LATTICE = np.eye(3).tolist()
    out = _run(monkeypatch, _org(gp=1, opt_type=11), [s])
    assert out[0] == pytest.approx(-1.9)


def _geometric_density(*args, **kwargs):
    return 2.0


def _individual12(enthalpy=-100.0, energy_gp=None, density_gp=None):
    s = _individual(enthalpy=enthalpy, energy_gp=energy_gp)
    s.LATTICE = np.eye(3).tolist()
    if density_gp is not None:
        s.density_gp = density_gp
    return s


def test_opt_type_12_formula_with_both_gp_predictions(monkeypatch):
    """`1 : gp` -> 0.1*energy_gp/factor - density_gp (both GP predictions)."""
    monkeypatch.setattr(_cf, "calc_density", _geometric_density)
    s = _individual12(enthalpy=-100.0, energy_gp=-280.0,
                      density_gp=1.9)
    out = _run(monkeypatch, _org(gp=1, opt_type=12), [s])
    assert out[0] == pytest.approx(0.1 * (-280.0) / 1.0 - 1.9)


def test_opt_type_12_keeps_raw_values_when_gp_is_off(monkeypatch):
    """With the GP off, enthalpy is force-field and density is geometric."""
    monkeypatch.setattr(_cf, "calc_density", _geometric_density)
    s = _individual12(enthalpy=-100.0, energy_gp=-280.0,
                      density_gp=1.9)
    out = _run(monkeypatch, _org(gp=0, opt_type=12), [s])
    assert out[0] == pytest.approx(0.1 * (-100.0) / 1.0 - 2.0)


def test_opt_type_12_falls_back_when_gp_predictions_are_missing(monkeypatch):
    monkeypatch.setattr(_cf, "calc_density", _geometric_density)
    a = _individual12(enthalpy=-100.0, energy_gp=float("nan"),
                      density_gp=float("nan"))
    b = _individual12(enthalpy=-120.0)
    out = _run(monkeypatch, _org(gp=1, opt_type=12), [a, b])
    assert out[0] == pytest.approx(0.1 * (-100.0) - 2.0)
    assert out[1] == pytest.approx(0.1 * (-120.0) - 2.0)
