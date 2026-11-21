"""``gpevo rft``: hold-out evaluation of RandomForest models."""
from __future__ import annotations

import os

import numpy as np
import pytest
from ase import Atoms
from ase.io import Trajectory
from sklearn.preprocessing import StandardScaler

from gpevo.cli import _build_parser

_HEADER = ", etot, ebond, eang, etor, evdw, ehb_cho,ehb_chn,ehb_chc, ecoul, density"
_N_ROWS = 8


def _write_csv(path, rows):
    with open(path, "w") as fd:
        fd.write(_HEADER + "\n")
        for i, row in enumerate(rows):
            fd.write("{},{}\n".format(i, ",".join("%.6f" % v for v in row)))


@pytest.fixture
def run_dir(tmp_path):
    """A run directory with an 8-crystal database."""
    root = tmp_path / "run"
    root.mkdir()
    (root / "INPUT.txt").write_text("data : data\n")
    data = root / "data"
    data.mkdir()
    rng = np.random.default_rng(1)
    mlp = rng.normal(size=(_N_ROWS, 10))
    dft = rng.normal(size=(_N_ROWS, 10))
    dft[:, 0] = -28000.0 - np.arange(_N_ROWS)
    dft[:, 9] = 1.7 + 0.01 * np.arange(_N_ROWS)
    _write_csv(data / "feature_mlp.csv", mlp)
    _write_csv(data / "feature.csv", dft)
    with Trajectory(str(data / "structures.traj"), "w") as traj:
        for _ in range(_N_ROWS):
            traj.write(Atoms("H2O",
                             positions=[[0, 0, 0], [0.8, 0, 0], [0, 0.8, 0]],
                             cell=[6.0, 6.0, 6.0], pbc=True))
    return str(root), str(data), mlp, dft


class _FakeTree:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


class _FakeRF:
    """Fixed predictions so report/plot tests need no real forest."""
    def __init__(self, value, sigma, **kwargs):
        # two trees whose predictions differ by 2*sigma -> std == sigma
        self.estimators_ = [_FakeTree(value - sigma), _FakeTree(value + sigma)]
        self.kwargs = kwargs

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        return self


def _stub_forests(monkeypatch, rft_module, captured):
    """Patch _fit_rf so tests can inspect the training rows."""
    def fake_fit(X_train, y, *, n_estimators, random_state, n_jobs):
        captured.setdefault("X", []).append(X_train.copy())
        captured.setdefault("y", []).append(y.copy())
        captured["kwargs"] = dict(n_estimators=n_estimators,
                                  random_state=random_state, n_jobs=n_jobs)
        # first call is density, second is energy
        value, sigma = (1.8, 0.01) if len(captured["X"]) == 1 else (-28000.0, 0.3)
        return _FakeRF(value, sigma)

    monkeypatch.setattr(rft_module, "_fit_rf", fake_fit)
    return captured


def test_rft_returns_records_and_report(run_dir, monkeypatch, capsys):
    from gpevo.tools import rft as rft_module
    root, _data, _mlp, dft = run_dir
    _stub_forests(monkeypatch, rft_module, {})

    records = rft_module.rft(path=root, n=3, seed=0, plot=False)

    assert len(records) == 3
    assert set(records[0]) == {
        "row", "dft_density", "dft_energy", "rf_density",
        "rf_density_sigma", "rf_energy", "rf_energy_sigma"}
    assert [r["row"] for r in records] == sorted(r["row"] for r in records)
    for rec in records:
        assert rec["dft_density"] == pytest.approx(dft[rec["row"], 9])
        assert rec["dft_energy"] == pytest.approx(dft[rec["row"], 0])
        assert rec["rf_density"] == pytest.approx(1.8)
        assert rec["rf_energy"] == pytest.approx(-28000.0)
        assert rec["rf_density_sigma"] == pytest.approx(0.01)
    out = capsys.readouterr().out
    assert "RF holdout test" in out
    assert "RMSE" in out and "MAE" in out
    assert "RF ensemble std" in out
    assert "backend=" not in out


def test_rft_sampling_is_seed_controlled(run_dir, monkeypatch):
    from gpevo.tools import rft as rft_module
    root, _data, _mlp, _dft = run_dir
    _stub_forests(monkeypatch, rft_module, {})

    a = [r["row"] for r in rft_module.rft(path=root, n=3, seed=7, plot=False)]
    b = [r["row"] for r in rft_module.rft(path=root, n=3, seed=7, plot=False)]
    c = [r["row"] for r in rft_module.rft(path=root, n=3, seed=9, plot=False)]
    assert a == b
    assert a != c


def test_rft_refits_without_heldout_rows(run_dir, monkeypatch):
    from gpevo.tools import rft as rft_module
    root, data, _mlp, dft = run_dir
    captured = _stub_forests(monkeypatch, rft_module, {})

    records = rft_module.rft(path=root, n=3, seed=0, plot=False)
    idx = [r["row"] for r in records]
    mask = np.ones(_N_ROWS, dtype=bool)
    mask[idx] = False

    mlp = np.loadtxt(os.path.join(data, "feature_mlp.csv"),
                     delimiter=",", skiprows=1)[:, 1:]
    expected = StandardScaler().fit(mlp[mask]).transform(mlp[mask])
    assert len(captured["X"]) == 2          # density + energy
    for X in captured["X"]:
        assert X.shape == (_N_ROWS - 3, 10)
        np.testing.assert_allclose(X, expected)
    np.testing.assert_allclose(captured["y"][0], dft[mask, 9])
    np.testing.assert_allclose(captured["y"][1], dft[mask, 0])
    assert captured["kwargs"]["n_estimators"] == 300
    assert captured["kwargs"]["random_state"] == 37


def test_rft_uses_same_split_as_gpt(run_dir, monkeypatch):
    """The whole point of rft(): identical holdout to gpt()."""
    from gpevo.tools.gpt import prepare_holdout
    from gpevo.tools import rft as rft_module
    root, _data, _mlp, _dft = run_dir

    prep = prepare_holdout(root, 3, seed=5)
    # rft() delegates to prepare_holdout with the same (path, n, seed).
    seen = {}

    def fake_fit(X_train, y, **kwargs):
        seen["X"] = X_train.copy()
        return _FakeRF(1.8, 0.01)

    monkeypatch.setattr(rft_module, "_fit_rf", fake_fit)
    rft_module.rft(path=root, n=3, seed=5, plot=False)
    np.testing.assert_allclose(seen["X"], prep["X_train"])


def test_rft_writes_parity_plots(run_dir, monkeypatch, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from gpevo.tools import rft as rft_module
    root, _data, _mlp, _dft = run_dir
    _stub_forests(monkeypatch, rft_module, {})

    monkeypatch.chdir(tmp_path)
    rft_module.rft(path=root, n=3, seed=0, plot=True)

    for name in ("test_rf_energy.svg", "test_rf_density.svg"):
        p = tmp_path / name
        assert p.exists() and p.stat().st_size > 0, name
        assert b"<svg" in p.read_bytes()[:400]
    out = capsys.readouterr().out
    assert "Energy parity" in out and "Density parity" in out
    assert "test_rf_energy.svg" in out


def test_rft_uses_green_scatter(run_dir, monkeypatch, tmp_path):
    """rft parity plots use the green #009E73 scatter colour."""
    from gpevo.tools import rft as rft_module
    root, _data, _mlp, _dft = run_dir
    _stub_forests(monkeypatch, rft_module, {})
    monkeypatch.chdir(tmp_path)

    seen = {}
    real = rft_module.save_parity_plots

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(rft_module, "save_parity_plots", spy)
    rft_module.rft(path=root, n=3, seed=0, plot=True)

    assert seen["scatter_color"] == "#009E73"
    for name in ("test_rf_energy.svg", "test_rf_density.svg"):
        svg = (tmp_path / name).read_text(encoding="utf-8").lower()
        assert "#009e73" in svg, name


@pytest.mark.parametrize("mutation", [
    "no_input", "no_csv", "n_zero", "n_too_large", "train_too_small",
    "count_mismatch",
])
def test_rft_validation_errors(run_dir, monkeypatch, mutation):
    from gpevo.tools import rft as rft_module
    root, data, _mlp, _dft = run_dir
    _stub_forests(monkeypatch, rft_module, {})
    n = 3
    if mutation == "no_input":
        os.remove(os.path.join(root, "INPUT.txt"))
    elif mutation == "no_csv":
        os.remove(os.path.join(data, "feature.csv"))
    elif mutation == "n_zero":
        n = 0
    elif mutation == "n_too_large":
        n = _N_ROWS
    elif mutation == "train_too_small":
        n = _N_ROWS - 1
    elif mutation == "count_mismatch":
        _write_csv(os.path.join(data, "feature.csv"), np.zeros((_N_ROWS + 1, 10)))

    with pytest.raises(RuntimeError):
        rft_module.rft(path=root, n=n, seed=0, plot=False)


def test_rft_real_fit_smoke(run_dir, monkeypatch, tmp_path):
    """One genuine forest fit: finite predictions and both SVGs."""
    pytest.importorskip("matplotlib")
    from gpevo.tools.rft import rft as rft_func
    root, _data, _mlp, _dft = run_dir

    monkeypatch.chdir(tmp_path)
    records = rft_func(path=root, n=2, seed=0, plot=True,
                       n_estimators=10, n_jobs=1)

    assert len(records) == 2
    for rec in records:
        assert np.isfinite(rec["rf_density"]) and np.isfinite(rec["rf_energy"])
        assert rec["rf_density_sigma"] >= 0
        assert rec["rf_energy_sigma"] >= 0
    for name in ("test_rf_energy.svg", "test_rf_density.svg"):
        assert (tmp_path / name).exists()


def test_cli_dispatches_rft(run_dir, monkeypatch):
    from gpevo import cli
    from gpevo.tools import rft as rft_module
    root, _data, _mlp, _dft = run_dir
    calls = []
    monkeypatch.setattr(rft_module, "rft", lambda **kw: calls.append(kw))

    cli.main(["rft", "-p", root, "--n", "2"])

    assert calls == [{"path": root, "n": 2,
                      "n_estimators": 300, "n_jobs": -1}]


def test_cli_rft_options():
    parser = _build_parser()
    sub = next(a for a in parser._actions
               if getattr(a, "dest", None) == "command")
    rft_parser = sub.choices["rft"]
    flags = {f for action in rft_parser._actions for f in action.option_strings}
    assert {"--n", "--path", "--n-estimators", "--n-jobs"} <= flags
