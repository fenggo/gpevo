"""Refresh a carried-over (kept-best) structure's fitness against the current GP.

A kept-best skips its GP step next generation, so its density_gp/energy_gp
would stay at the old model's prediction after the surrogate was retrained.
The 10-D gp_feature is geometry-only and already on the structure, so the
refresh is a scaler transform + two GP posterior evaluations -- no GULP/DFT.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from gpevo.runtime import G, AttrDict, struct
from gpevo.uspexkit import core
from gpevo.FunctionFolder.USPEX.src import gp_ml
from gpevo.FunctionFolder.USPEX.src.gp_ml import refresh_kept_best_gp_metrics

_HEADER = ", etot, ebond, eang, etor, evdw, ehb_cho,ehb_chn,ehb_chc, ecoul, density"


class _FakeGPR:
    """Picklable stand-in for a fitted GP: predict returns (mean, std)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def predict(self, X, return_std=True):
        n = X.shape[0]
        mean = np.full(n, self.mean)
        if return_std:
            return mean, np.full(n, self.std)
        return mean


def _write_models(data_dir, density=1.80, energy=-100.0, version=None):
    """Write version-matched cached models + a training CSV into data_dir."""
    os.makedirs(data_dir, exist_ok=True)
    version = core._KERNEL_VERSION if version is None else version
    for name, model in (("gpr_density.pkl", _FakeGPR(density, 0.01)),
                        ("gpr_energy.pkl", _FakeGPR(energy, 0.02))):
        with open(os.path.join(data_dir, name), "wb") as f:
            pickle.dump({"version": version, "model": model}, f)
    with open(os.path.join(data_dir, "feature_mlp.csv"), "w") as fd:
        fd.write(_HEADER + "\n")
        for i in range(4):
            fd.write(f"{i}, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.9\n")


def _org(data_dir, *, gp=1, backend="sklearn"):
    return struct(gp=gp, gpBackend=backend, homePath=os.path.dirname(data_dir),
                  gpData=os.path.basename(data_dir), optType=11)


# ── predict_from_feature (core) ────────────────────────────────────────

def test_predict_from_feature_uses_cached_models(tmp_path):
    _write_models(str(tmp_path), density=1.80, energy=-100.0)
    result = core.predict_from_feature([1.0] * 10, str(tmp_path),
                                       backend="sklearn")
    assert result["density_gp"] == pytest.approx(1.80)
    assert result["energy_pred"] == pytest.approx(-100.0)
    # 1.96 * std of the two fake posteriors
    assert result["uncertainty"] == pytest.approx(0.0196)
    assert result["uncertainty_eng"] == pytest.approx(0.0392)


def test_predict_from_feature_rejects_stale_model(tmp_path):
    _write_models(str(tmp_path), version="wrong-version")
    assert core.predict_from_feature([1.0] * 10, str(tmp_path),
                                     backend="sklearn") is None


def test_predict_from_feature_returns_none_when_no_models(tmp_path):
    assert core.predict_from_feature([1.0] * 10, str(tmp_path),
                                     backend="sklearn") is None


# ── refresh_kept_best_gp_metrics ───────────────────────────────────────

def test_refresh_updates_scalars_from_saved_feature(tmp_path):
    data_dir = str(tmp_path / "data")
    _write_models(data_dir, density=1.77, energy=-99.0)
    s = AttrDict(gp_feature=[1.0] * 10,
                 density_gp=1.50, energy_gp=-80.0,
                 uq=0.5, uq_eng=0.6, sigma_gp=0.25, sigma_eng=0.3)
    assert refresh_kept_best_gp_metrics(s, _org(data_dir)) is True
    assert s.density_gp == pytest.approx(1.77)
    assert s.energy_gp == pytest.approx(-99.0)
    assert s.uq == pytest.approx(0.0196)
    assert s.uq_eng == pytest.approx(0.0392)
    assert s.sigma_gp == pytest.approx(0.01)
    assert s.sigma_eng == pytest.approx(0.02)
    # gp_feature itself is untouched -- it is the geometry-only input.
    assert s.gp_feature == [1.0] * 10


def test_refresh_is_noop_when_gp_disabled(tmp_path):
    data_dir = str(tmp_path / "data")
    _write_models(data_dir)
    s = AttrDict(gp_feature=[1.0] * 10, density_gp=1.50)
    assert refresh_kept_best_gp_metrics(s, _org(data_dir, gp=0)) is False
    assert s.density_gp == 1.50


def test_refresh_is_noop_without_saved_feature(tmp_path):
    data_dir = str(tmp_path / "data")
    _write_models(data_dir)
    s = AttrDict(density_gp=1.50)
    assert refresh_kept_best_gp_metrics(s, _org(data_dir)) is False
    assert s.density_gp == 1.50


def test_refresh_is_noop_when_no_cached_model(tmp_path):
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    s = AttrDict(gp_feature=[1.0] * 10, density_gp=1.50)
    assert refresh_kept_best_gp_metrics(s, _org(data_dir)) is False
    assert s.density_gp == 1.50


def test_refresh_never_raises_on_malformed_model(tmp_path):
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    for name in ("gpr_density.pkl", "gpr_energy.pkl"):
        with open(os.path.join(data_dir, name), "wb") as f:
            f.write(b"not a pickle")
    s = AttrDict(gp_feature=[1.0] * 10, density_gp=1.50)
    assert refresh_kept_best_gp_metrics(s, _org(data_dir)) is False
    assert s.density_gp == 1.50


# ── KeepBestStructures integration ────────────────────────────────────

def test_kept_best_carries_refreshed_gp_metrics(monkeypatch, tmp_path):
    """The elitist step hands the next generation a fitness matching the new model."""
    from gpevo.FunctionFolder.USPEX.src import KeepBestStructures as KBS

    data_dir = str(tmp_path / "data")
    _write_models(data_dir, density=1.77, energy=-99.0)

    source = AttrDict(Number=5, Enthalpies=np.array([-110.0]), numIons=[2],
                      gp_feature=[1.0] * 10, density_gp=1.50, energy_gp=-80.0,
                      FINGERPRINT=None)
    ORG = struct(gp=1, gpBackend="sklearn", homePath=str(tmp_path),
                 gpData="data", optType=11, abinitioCode=[1, 1],
                 bestFrac=1.0, keepBestHM=1, dynamicalBestHM=0, reoptOld=0,
                 doFing=0, toleranceBestHM=0.02, dimension=3,
                 convex_hull=[], softMutOnly=[], weight=1.0, resFolder="results1")
    POP = struct(generation=1, ranking=[0], bad_rank=0, fitness=np.array([-1.50]),
                 POPULATION=[source], convex_hull=[])
    OFF = struct(POPULATION=[AttrDict()], generation=1, bodyCount=5)
    monkeypatch.setitem(G, "ORG_STRUC", ORG)
    monkeypatch.setitem(G, "POP_STRUC", POP)
    monkeypatch.setitem(G, "OFF_STRUC", OFF)

    KBS.KeepBestStructures()

    child = OFF.POPULATION[0]
    assert child.density_gp == pytest.approx(1.77)
    assert child.energy_gp == pytest.approx(-99.0)
    assert child.gp_feature == [1.0] * 10
    assert child.howCome == "keptBest"
