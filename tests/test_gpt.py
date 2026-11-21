"""``gpevo gpt``: hold-out evaluation of the GP models, no calculator runs."""
from __future__ import annotations

import os

import numpy as np
import pytest
from ase import Atoms
from ase.io import Trajectory
from sklearn.preprocessing import StandardScaler

_HEADER = ", etot, ebond, eang, etor, evdw, ehb_cho,ehb_chn,ehb_chc, ecoul, density"
_N_ROWS = 8


def _write_csv(path, rows):
    with open(path, "w") as fd:
        fd.write(_HEADER + "\n")
        for i, row in enumerate(rows):
            fd.write("{},{}\n".format(i, ",".join("%.6f" % v for v in row)))


def _atoms():
    return Atoms("H2O",
                 positions=[[0, 0, 0], [0.8, 0, 0], [0, 0.8, 0]],
                 cell=[6.0, 6.0, 6.0], pbc=True)


@pytest.fixture
def run_dir(tmp_path):
    """A run directory with an 8-crystal GP database."""
    root = tmp_path / "run"
    root.mkdir()
    (root / "INPUT.txt").write_text("data : data\n")
    data = root / "data"
    data.mkdir()

    rng = np.random.default_rng(1)
    mlp = rng.normal(size=(_N_ROWS, 10))
    dft = rng.normal(size=(_N_ROWS, 10))
    dft[:, 0] = -28000.0 - np.arange(_N_ROWS)   # column 1: DFT energy
    dft[:, 9] = 1.7 + 0.01 * np.arange(_N_ROWS)  # column 10: DFT density
    _write_csv(data / "feature_mlp.csv", mlp)
    _write_csv(data / "feature.csv", dft)
    with Trajectory(str(data / "structures.traj"), "w") as traj:
        for _ in range(_N_ROWS):
            traj.write(_atoms())
    return str(root), str(data), mlp, dft


class _FakeGPR:
    def __init__(self, value, sigma):
        self.value = value
        self.sigma = sigma

    def predict(self, X, return_std=False):
        mu = np.full(len(X), self.value)
        if return_std:
            return mu, np.full(len(X), self.sigma)
        return mu


def _stub_loader(monkeypatch, gpt_module, captured):
    """Fake GP models; record what was trained and leave a cwd sentinel."""
    def fake_loader(X, y, y_eng, **options):
        captured["X"] = X.copy()
        captured["y"] = y.copy()
        captured["y_eng"] = y_eng.copy()
        captured["options"] = options
        captured["cwd"] = os.getcwd()
        with open("gpt_sentinel.txt", "w") as fd:
            fd.write("x")
        # load_gaussian_process returns (energy, density)
        return _FakeGPR(-28000.0, 0.3), _FakeGPR(1.8, 0.01)

    monkeypatch.setattr(gpt_module, "load_gaussian_process", fake_loader)
    return captured


def test_gpt_returns_holdout_records_and_report(run_dir, monkeypatch, capsys):
    from gpevo.tools import gpt as gpt_module
    root, _data, _mlp, dft = run_dir
    captured = _stub_loader(monkeypatch, gpt_module, {})

    records = gpt_module.gpt(path=root, n=3, seed=0)

    assert len(records) == 3
    assert set(records[0]) == {
        "row", "dft_density", "dft_energy", "gp_density", "gp_density_sigma",
        "gp_energy", "gp_energy_sigma"}
    rows = [r["row"] for r in records]
    assert rows == sorted(rows)
    for rec in records:
        assert rec["dft_density"] == pytest.approx(dft[rec["row"], 9])
        assert rec["dft_energy"] == pytest.approx(dft[rec["row"], 0])
        assert rec["gp_density"] == pytest.approx(1.8)
        assert rec["gp_energy"] == pytest.approx(-28000.0)
        assert rec["gp_density_sigma"] == pytest.approx(0.01)
        assert rec["gp_energy_sigma"] == pytest.approx(0.3)
    out = capsys.readouterr().out
    assert "GP holdout test" in out
    assert "RMSE" in out and "MAE" in out
    assert "sd = GP posterior sigma" in out
    assert captured["cwd"] != root


def test_gpt_sampling_is_seed_controlled(run_dir, monkeypatch):
    from gpevo.tools import gpt as gpt_module
    root, _data, _mlp, _dft = run_dir
    _stub_loader(monkeypatch, gpt_module, {})

    a = [r["row"] for r in gpt_module.gpt(path=root, n=3, seed=7)]
    b = [r["row"] for r in gpt_module.gpt(path=root, n=3, seed=7)]
    c = [r["row"] for r in gpt_module.gpt(path=root, n=3, seed=9)]
    assert a == b
    assert a != c


def test_gpt_refits_without_heldout_rows(run_dir, monkeypatch):
    """The scaler and both GPRs must be fitted on the reduced training set."""
    from gpevo.tools import gpt as gpt_module
    root, data, _mlp, dft = run_dir
    captured = _stub_loader(monkeypatch, gpt_module, {})

    records = gpt_module.gpt(path=root, n=3, seed=0)
    idx = [r["row"] for r in records]
    mask = np.ones(_N_ROWS, dtype=bool)
    mask[idx] = False

    # Reload from disk: the fixture writes 6 decimals, so the in-memory array
    # is slightly more precise than what gpt() actually reads.
    mlp = np.loadtxt(os.path.join(data, "feature_mlp.csv"),
                     delimiter=",", skiprows=1)[:, 1:]
    assert captured["X"].shape == (_N_ROWS - 3, 10)
    np.testing.assert_allclose(captured["y"], dft[mask, 9])
    np.testing.assert_allclose(captured["y_eng"], dft[mask, 0])
    expected = StandardScaler().fit(mlp[mask]).transform(mlp[mask])
    np.testing.assert_allclose(captured["X"], expected)


def test_gpt_forwards_default_gp_options(run_dir, monkeypatch):
    """INPUT.txt without gpbackend still passes the same defaults as gp()."""
    from gpevo.tools import gpt as gpt_module
    root, _data, _mlp, _dft = run_dir
    captured = _stub_loader(monkeypatch, gpt_module, {})

    gpt_module.gpt(path=root, n=3, seed=0)

    assert captured["options"] == {
        "backend": "sklearn", "training_iter": 200, "learning_rate": 0.1}


def test_gpt_honours_gpbackend_from_input(run_dir, monkeypatch, capsys):
    """The holdout refit must use the backend/options production gp() uses."""
    from gpevo.tools import gpt as gpt_module
    root, _data, _mlp, _dft = run_dir
    with open(os.path.join(root, "INPUT.txt"), "w") as fd:
        fd.write("data : data\ngpytorch : gpbackend\n50 : gpiter\n0.05 : gplr\n")
    # gpytorch is an opt-in extra; pretend it is installed so we can assert
    # the options are forwarded without importing torch.
    monkeypatch.setattr(gpt_module, "_gpytorch_is_installed", lambda: True)
    captured = _stub_loader(monkeypatch, gpt_module, {})

    gpt_module.gpt(path=root, n=3, seed=0)

    assert captured["options"] == {
        "backend": "gpytorch", "training_iter": 50, "learning_rate": 0.05}
    out = capsys.readouterr().out
    assert "backend=gpytorch" in out
    assert "iter=50" in out and "lr=0.05" in out


def test_gpt_rejects_missing_gpytorch_backend(run_dir, monkeypatch):
    """Never silently score sklearn when gpytorch was requested."""
    from gpevo.tools import gpt as gpt_module
    root, _data, _mlp, _dft = run_dir
    with open(os.path.join(root, "INPUT.txt"), "w") as fd:
        fd.write("data : data\ngpytorch : gpbackend\n")
    monkeypatch.setattr(gpt_module, "_gpytorch_is_installed", lambda: False)
    _stub_loader(monkeypatch, gpt_module, {})

    with pytest.raises(RuntimeError, match="gpytorch is not installed"):
        gpt_module.gpt(path=root, n=3, seed=0)


def test_gpt_rejects_unknown_gpbackend(run_dir, monkeypatch):
    from gpevo.tools import gpt as gpt_module
    root, _data, _mlp, _dft = run_dir
    with open(os.path.join(root, "INPUT.txt"), "w") as fd:
        fd.write("data : data\ntorch : gpbackend\n")
    _stub_loader(monkeypatch, gpt_module, {})

    with pytest.raises(RuntimeError, match="unknown gpbackend"):
        gpt_module.gpt(path=root, n=3, seed=0)


def test_gpt_does_not_pollute_run_directory(run_dir, monkeypatch):
    from gpevo.tools import gpt as gpt_module
    root, data, _mlp, _dft = run_dir
    cwd_before = os.getcwd()
    captured = _stub_loader(monkeypatch, gpt_module, {})

    gpt_module.gpt(path=root, n=2, seed=0)

    assert os.getcwd() == cwd_before
    assert captured["cwd"] != cwd_before
    for name in ("gpr_density.pkl", "gpr_energy.pkl", "gpcsp.log",
                 "gpt_sentinel.txt"):
        assert not os.path.exists(os.path.join(root, name)), name
        assert not os.path.exists(os.path.join(data, name)), name
        assert not os.path.exists(os.path.join(cwd_before, name)), name


def test_cli_dispatches_gpt(run_dir, monkeypatch):
    from gpevo import cli
    from gpevo.tools import gpt as gpt_module
    root, _data, _mlp, _dft = run_dir
    calls = []
    monkeypatch.setattr(gpt_module, "gpt",
                        lambda **kw: calls.append(kw))

    cli.main(["gpt", "-p", root, "--n", "2"])

    assert calls == [{"path": root, "n": 2}]


def test_cli_has_no_ncpu_option():
    from gpevo.cli import _build_parser
    parser = _build_parser()
    sub = next(a for a in parser._actions
               if getattr(a, "dest", None) == "command")
    gpt_parser = sub.choices["gpt"]
    flags = {f for action in gpt_parser._actions for f in action.option_strings}
    assert "--n" in flags and "--path" in flags
    assert "--ncpu" not in flags


@pytest.mark.parametrize("mutation", [
    "no_input", "no_data_dir", "no_csv", "n_zero", "n_negative",
    "n_too_large", "train_too_small", "count_mismatch",
])
def test_gpt_validation_errors(run_dir, monkeypatch, mutation):
    from gpevo.tools import gpt as gpt_module
    root, data, _mlp, _dft = run_dir
    _stub_loader(monkeypatch, gpt_module, {})
    n = 3
    if mutation == "no_input":
        os.remove(os.path.join(root, "INPUT.txt"))
    elif mutation == "no_data_dir":
        with open(os.path.join(root, "INPUT.txt"), "w") as fd:
            fd.write("data : missing\n")
    elif mutation == "no_csv":
        os.remove(os.path.join(data, "feature.csv"))
    elif mutation == "n_zero":
        n = 0
    elif mutation == "n_negative":
        n = -1
    elif mutation == "n_too_large":
        n = _N_ROWS
    elif mutation == "train_too_small":
        n = _N_ROWS - 1
    elif mutation == "count_mismatch":
        _write_csv(os.path.join(data, "feature.csv"), np.zeros((_N_ROWS + 1, 10)))

    with pytest.raises(RuntimeError):
        gpt_module.gpt(path=root, n=n, seed=0)


def test_gpt_real_fit_smoke(run_dir):
    """One genuine refit: predictions are finite and the run dir stays clean."""
    from gpevo.tools.gpt import gpt as gpt_func
    root, _data, _mlp, _dft = run_dir

    records = gpt_func(path=root, n=2, seed=0)

    assert len(records) == 2
    for rec in records:
        assert np.isfinite(rec["gp_density"])
        assert np.isfinite(rec["gp_energy"])
        assert rec["gp_density_sigma"] >= 0
        assert rec["gp_energy_sigma"] >= 0
    for name in ("gpr_density.pkl", "gpr_energy.pkl", "gpcsp.log"):
        assert not os.path.exists(os.path.join(root, name)), name
