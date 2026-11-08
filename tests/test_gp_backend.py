"""Opt-in GPyTorch training backend: INPUT.txt wiring, routing and caches.

These tests need no GULP/SIESTA and (apart from the importorskip group) no
gpytorch -- the routing tests monkeypatch the model constructor, exactly like
the rest of tests/test_gp_training.py stubs out uspexkit._call.
"""
from __future__ import annotations

import importlib.util
import os
import pickle

import numpy as np
import pytest

from gpevo.FunctionFolder.USPEX.src import gp_ml
from gpevo.uspexkit import core


class _FakeSklearnGPR:
    """Picklable stand-in exposing the sklearn internals LOO needs."""

    def __init__(self, **kwargs):
        self.L_ = np.eye(2)
        self.alpha_ = np.zeros(2)

    def fit(self, X, y):
        return self


class _FakeGpytorchGPR:
    """Picklable stand-in without sklearn's Cholesky factor."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        return self


def _org_from_input(tmp_path, text):
    from gpevo.runtime import G
    from gpevo.FunctionFolder.USPEX.src.createORG_AbinitCode import (
        createORG_AbinitCode)
    from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault

    (tmp_path / "INPUT.txt").write_text(text)
    G["ORG_STRUC"] = createORGDefault()
    G["ORG_STRUC"].homePath = str(tmp_path)
    createORG_AbinitCode(str(tmp_path / "INPUT.txt"))
    return G["ORG_STRUC"]


# ── INPUT.txt → ORG → parse_gp_params ───────────────────────────────────

def test_backend_options_parsed_from_input(tmp_path):
    org = _org_from_input(tmp_path,
                          "1 : gp\ngpytorch : gpbackend\n50 : gpiter\n0.05 : gplr\n")
    params = gp_ml.parse_gp_params(org)

    assert org.gpBackend == "gpytorch"
    assert org.gpTrainingIter == 50
    assert org.gpLr == 0.05
    assert params["backend"] == "gpytorch"
    assert params["training_iter"] == 50
    assert params["learning_rate"] == 0.05


def test_backend_options_default_to_sklearn(tmp_path):
    org = _org_from_input(tmp_path, "1 : gp\n")
    params = gp_ml.parse_gp_params(org)

    assert org.gpBackend == "sklearn"
    assert org.gpTrainingIter == 200
    assert org.gpLr == 0.1
    assert params["backend"] == "sklearn"
    assert params["training_iter"] == 200
    assert params["learning_rate"] == 0.1


def test_unknown_backend_falls_back_to_sklearn(tmp_path, capsys):
    org = _org_from_input(tmp_path, "1 : gp\nfoobar : gpbackend\n")
    assert org.gpBackend == "sklearn"
    assert "unknown gpbackend" in capsys.readouterr().out
    assert gp_ml.parse_gp_params(org)["backend"] == "sklearn"


def test_run_gp_passes_backend_options(monkeypatch):
    from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault

    org = createORGDefault()
    org.gpBackend = "gpytorch"
    org.gpTrainingIter = 50
    org.gpLr = 0.05
    org.resFolder = "/tmp/results9"
    captured = {}

    def fake_call(name, **kwargs):
        captured.update(kwargs)
        return True, {"density_gp": 1.9}

    monkeypatch.setattr(gp_ml, "_call", fake_call)
    # gpytorch is installed on the dev machine; keep this test independent of
    # that by stubbing the availability check.
    monkeypatch.setattr(gp_ml, "_backend_ready", lambda backend: True)

    gp_ml.run_gp(org, id_=7)
    assert captured["backend"] == "gpytorch"
    assert captured["training_iter"] == 50
    assert captured["learning_rate"] == 0.05
    assert captured["id_"] == 7
    assert captured["log_dir"] == "/tmp/results9"


# ── fail-fast on a missing optional dependency ──────────────────────────

def test_missing_gpytorch_fails_fast(monkeypatch, capsys):
    assert gp_ml._backend_ready("sklearn") is True
    with monkeypatch.context() as m:
        m.setattr(importlib.util, "find_spec", lambda name: None)
        assert gp_ml._backend_ready("gpytorch") is False
    out = capsys.readouterr().out
    assert "gpytorch is not installed" in out
    assert "NOT falling back to sklearn silently" in out


def test_run_gp_skips_step_when_gpytorch_missing(monkeypatch):
    from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault

    org = createORGDefault()
    org.gpBackend = "gpytorch"
    called = []
    monkeypatch.setattr(gp_ml, "_call", lambda *a, **kw: called.append(kw))

    with monkeypatch.context() as m:
        m.setattr(importlib.util, "find_spec", lambda name: None)
        assert gp_ml.run_gp(org) is None
    assert called == []


# ── routing: one backend, two kernel families ───────────────────────────

class _StubModel:
    kernel_ = "kernel"

    def log_marginal_likelihood(self):
        return 0.0


def test_load_gaussian_process_dispatches_cores(monkeypatch, tmp_path):
    seen = []

    def fake(path, X, y, kernel, **kwargs):
        seen.append((path, kwargs.get("backend"), kwargs.get("core")))
        return _StubModel()

    monkeypatch.setattr(core, "_load_or_fit_gpr", fake)
    monkeypatch.chdir(tmp_path)
    core.load_gaussian_process(np.zeros((2, 2)), np.zeros(2), np.zeros(2),
                               backend="gpytorch", training_iter=7,
                               learning_rate=0.2)

    assert ("gpr_density.pkl", "gpytorch", "matern") in seen
    assert ("gpr_energy.pkl", "gpytorch", "rbf") in seen
    assert len(seen) == 2


def test_default_backend_still_uses_sklearn_tag(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core, "GaussianProcessRegressor", _FakeSklearnGPR)

    core._load_or_fit_gpr("d.pkl", np.zeros((2, 2)), np.zeros(2), None)
    with open("d.pkl", "rb") as fp:
        blob = pickle.load(fp)
    assert blob["version"] == core._KERNEL_VERSION == 2
    assert isinstance(blob["model"], _FakeSklearnGPR)
    assert core._GPYTORCH_VERSION == "gpytorch-2"


# ── cache isolation between backends ────────────────────────────────────

def test_switching_backends_always_refits(monkeypatch, tmp_path):
    import gpevo.uspexkit.gp as gp_backend

    calls = []

    def sk_fit(self, X, y):
        calls.append("sklearn")
        return self

    def gp_fit(self, X, y):
        calls.append("gpytorch")
        return self

    # Module-level classes so the fitted models can be pickled into the cache.
    monkeypatch.setattr(_FakeSklearnGPR, "fit", sk_fit)
    monkeypatch.setattr(_FakeGpytorchGPR, "fit", gp_fit)
    monkeypatch.setattr(core, "GaussianProcessRegressor", _FakeSklearnGPR)
    monkeypatch.setattr(gp_backend, "GPyTorchRegressor", _FakeGpytorchGPR)
    monkeypatch.chdir(tmp_path)

    X, y = np.zeros((2, 2)), np.zeros(2)
    core._load_or_fit_gpr("d.pkl", X, y, None, backend="sklearn")
    core._load_or_fit_gpr("d.pkl", X, y, None, backend="gpytorch")
    core._load_or_fit_gpr("d.pkl", X, y, None, backend="sklearn")

    # sklearn -> gpytorch -> sklearn: every switch sees a foreign version tag
    # and refits instead of reusing the other backend's pickle.
    assert calls == ["sklearn", "gpytorch", "sklearn"]


def test_cached_models_rejects_non_sklearn_model(tmp_path):
    for name in ("gpr_density.pkl", "gpr_energy.pkl"):
        with open(tmp_path / name, "wb") as fp:
            pickle.dump({"version": core._KERNEL_VERSION,
                         "model": _FakeGpytorchGPR()}, fp)

    assert gp_ml._cached_models(str(tmp_path)) is None


# ── scheduler command line is self-contained ────────────────────────────

def test_code_exec_line_uses_vendored_cli(monkeypatch):
    from gpevo.FunctionFolder.Submission import submitJob
    from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault

    monkeypatch.setattr(gp_ml, "is_gp_step", lambda ORG, step: True)
    monkeypatch.setattr(gp_ml, "_ncpu", lambda ORG: 24)
    org = createORGDefault()
    org.gpBackend = "gpytorch"
    org.gpTrainingIter = 50
    org.gpLr = 0.05
    org.resFolder = "/tmp/results9"

    line = submitJob._code_exec_line(org, None, 1)
    assert line.startswith("python -m gpevo.uspexkit.cli gp ")
    assert "--backend=gpytorch" in line
    assert "--iter=50" in line
    assert "--lr=0.05" in line
    assert "--res-folder=/tmp/results9" in line
    assert "uspexkit gp " not in line


def test_vendored_cli_parser_exposes_backend():
    from gpevo.uspexkit.cli import _build_parser

    args = _build_parser().parse_args(
        ["gp", "--backend=gpytorch", "--iter=50", "--lr=0.05",
         "--res-folder=/tmp/results2"])
    assert args.backend == "gpytorch"
    assert args.training_iter == 50
    assert args.learning_rate == 0.05
    assert args.res_folder == "/tmp/results2"
    assert args.command == "gp"


# ── real gpytorch (skipped when the optional extra is absent) ────────────

def test_gpytorch_regressor_roundtrip():
    pytest.importorskip("gpytorch")
    from gpevo.uspexkit.gp import GPyTorchRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3))
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=12)

    def fit(core_name):
        return GPyTorchRegressor(normalize_y=True, random_state=0,
                                 training_iter=5, n_restarts_optimizer=0,
                                 core=core_name).fit(X, y)

    m = fit("matern")
    mu, sd = m.predict(X[:3], return_std=True)
    assert mu.shape == (3,) and sd.shape == (3,)
    assert np.all(sd > 0)
    assert np.isfinite(m.log_marginal_likelihood())

    restored = pickle.loads(pickle.dumps(m))
    mu2, _ = restored.predict(X[:3], return_std=True)
    assert np.allclose(mu, mu2)

    # seeding makes repeated fits reproducible
    assert np.allclose(fit("matern").predict(X[:3]), mu)
    # the energy kernel family is a valid, distinct construction
    assert fit("rbf").predict(X[:3], return_std=True)[1].shape == (3,)


def test_gpytorch_fit_scores_restarts_with_the_prior_mll():
    """The restart score is log p(y|X,theta), not a posterior log-prob.

    Calling model(train_x) in eval mode returns the posterior (and trips
    GPInputWarning because input == train_inputs).  Feeding that to
    ExactMarginalLogLikelihood inflates the score used for best_state.
    """
    pytest.importorskip("gpytorch")
    import warnings
    import torch
    import gpytorch
    from gpytorch.utils.warnings import GPInputWarning
    from gpevo.uspexkit.gp import GPyTorchRegressor

    rng = np.random.default_rng(5)
    X = rng.normal(size=(10, 3))
    y = np.sin(X[:, 0]) + 0.05 * rng.normal(size=10)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = GPyTorchRegressor(normalize_y=True, random_state=0,
                              training_iter=5, n_restarts_optimizer=0).fit(X, y)
    assert not [w for w in caught if issubclass(w.category, GPInputWarning)]

    # Standard GPyTorch idiom: the MLL consumes the prior, so train mode.
    m._model.train()
    m._likelihood.train()
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(m.alpha):
        per_point = gpytorch.mlls.ExactMarginalLogLikelihood(
            m._likelihood, m._model)(m._model(m._X_train), m._y_train).item()
    m._model.eval()
    m._likelihood.eval()

    # ExactMLL divides by n; the public API reports the TOTAL, like sklearn.
    assert m.log_marginal_likelihood() == pytest.approx(per_point * len(y),
                                                        rel=1e-4, abs=1e-5)


def test_gpcsp_log_goes_to_the_results_folder(monkeypatch, tmp_path):
    """The GP report belongs to <resFolder>, not <data> or CalcFoldN."""
    res = tmp_path / "results2"
    monkeypatch.chdir(tmp_path)

    info = {
        'backend': 'gpytorch', 'core': 'matern',
        'n_train': 12, 'n_features': 3, 'n_restarts': 3, 'training_iter': 5,
        'restart_lmls': [-1.0, -0.9, -0.8], 'failed_restarts': 0,
        'best_lml': -0.8, 'best_evals': 21,
        'y_mean': 1.9, 'y_std': 0.03,
        'noise_sigma': 0.1, 'mean_constant': 0.0,
        'linear_outputscale': 0.01, 'core_outputscale': 0.12,
        'lengthscale': [1.0, 2.0, 3.0],
        'train_rmse': 0.02, 'train_r2': 0.98,
        'train_max_resid': 0.05, 'train_posterior_sigma_mean': 0.01,
    }

    class _StubGPR:
        kernel_ = "stub-kernel"
        _fit_info = info

        def log_marginal_likelihood(self):
            return -0.8

    monkeypatch.setattr(core, "_load_or_fit_gpr", lambda *a, **kw: _StubGPR())
    core.load_gaussian_process(np.zeros((12, 3)), np.zeros(12), np.zeros(12),
                               backend="gpytorch", log_dir=str(res))

    log = (res / "gpcsp.log").read_text(encoding="utf-8")
    assert "LML:" in log                      # original four lines survive
    assert "backend=gpytorch" in log
    assert "restart_LML=" in log
    assert "y_std=0.03" in log
    assert "train_RMSE=" in log
    assert "LOO_RMSE=n/a" in log
    assert not (tmp_path / "gpcsp.log").exists()


def test_uspexkit_gp_stays_callable_after_gpytorch_submodule_import():
    """The gp.py submodule must not shadow core.gp() on the package.

    Importing gpevo.uspexkit.gp (or unpickling a cached GPyTorchRegressor)
    sets pkg.gp to the module.  _uspexkit_func used to read that attribute and
    the *second* uspexkit.gp() call raised "'module' object is not callable".
    """
    import importlib
    from gpevo.uspexkit import core

    importlib.import_module("gpevo.uspexkit.gp")
    func = gp_ml._uspexkit_func("gp")
    assert func is core.gp
    assert callable(func)
    for name in ("pred", "calc"):
        assert gp_ml._uspexkit_func(name) is getattr(core, name)


def test_gpytorch_fit_diagnostics_survive_pickling():
    """A cache hit reads the pickled model, so the report must travel with it."""
    pytest.importorskip("gpytorch")
    from gpevo.uspexkit.gp import GPyTorchRegressor

    rng = np.random.default_rng(7)
    X = rng.normal(size=(10, 3))
    y = np.sin(X[:, 0]) + 0.05 * rng.normal(size=10)
    m = GPyTorchRegressor(normalize_y=True, random_state=0,
                          training_iter=5, n_restarts_optimizer=0).fit(X, y)

    info = m._fit_info
    assert info["backend"] == "gpytorch"
    assert info["n_train"] == 10 and info["n_features"] == 3
    assert info["n_restarts"] == 1 and info["failed_restarts"] == 0
    assert info["best_evals"] > 0
    assert len(info["restart_lmls"]) == 1
    assert info["noise_sigma"] > 0
    assert len(info["lengthscale"]) == 3
    assert np.isfinite(info["train_rmse"])
    assert np.isfinite(info["train_r2"])
    assert pickle.loads(pickle.dumps(m))._fit_info == info


def test_gpytorch_restarts_use_different_initialisations():
    """n_restarts_optimizer used to replay one fixed init N times."""
    pytest.importorskip("gpytorch")
    from gpevo.uspexkit.gp import (_build_kernel, _kernel_init, _sample_init,
                                   GPyTorchRegressor)

    init = _kernel_init(None, 3)
    rng = np.random.default_rng(0)
    a, b = _sample_init(init, rng), _sample_init(init, rng)
    assert a['linear_scale'] != b['linear_scale']
    assert a['core_scale'] != b['core_scale']
    assert not np.allclose(a['lengthscale'], b['lengthscale'])
    assert a['noise'] != b['noise']
    ka, kb = _build_kernel(3, init=a), _build_kernel(3, init=b)
    assert not np.allclose(
        ka.kernels[1].base_kernel.lengthscale.detach().numpy(),
        kb.kernels[1].base_kernel.lengthscale.detach().numpy())

    X = np.random.default_rng(1).normal(size=(12, 3))
    m = GPyTorchRegressor(normalize_y=True, random_state=0, training_iter=5,
                          n_restarts_optimizer=3).fit(X, np.sin(X[:, 0]))
    assert len(m._fit_info['restart_lmls']) == 4
    assert m._fit_info['best_lml'] == max(m._fit_info['restart_lmls'])


def test_gpytorch_fit_raises_when_every_restart_fails(monkeypatch):
    pytest.importorskip("gpytorch")
    import torch
    from gpevo.uspexkit.gp import GPyTorchRegressor

    def boom(self, closure):
        raise RuntimeError("cholesky failed")
    monkeypatch.setattr(torch.optim.LBFGS, "step", boom)

    X = np.random.default_rng(2).normal(size=(10, 3))
    with pytest.raises(RuntimeError, match="all 2 L-BFGS restart"):
        GPyTorchRegressor(normalize_y=True, training_iter=5,
                          n_restarts_optimizer=1).fit(X, np.sin(X[:, 0]))


def test_gpytorch_predict_accepts_a_single_1d_row():
    pytest.importorskip("gpytorch")
    from gpevo.uspexkit.gp import GPyTorchRegressor

    X = np.random.default_rng(3).normal(size=(10, 3))
    m = GPyTorchRegressor(normalize_y=True, random_state=0,
                          training_iter=5).fit(X, np.sin(X[:, 0]))
    mu2, sd2 = m.predict(X[:1], return_std=True)
    mu1, sd1 = m.predict(X[0], return_std=True)
    assert np.allclose(mu1, mu2) and np.allclose(sd1, sd2)


def test_fit_rejects_1d_X():
    pytest.importorskip("gpytorch")
    from gpevo.uspexkit.gp import GPyTorchRegressor
    with pytest.raises(ValueError, match="2-D"):
        GPyTorchRegressor().fit(np.zeros(3), np.zeros(3))


def test_log_marginal_likelihood_rejects_theta():
    pytest.importorskip("gpytorch")
    from gpevo.uspexkit.gp import GPyTorchRegressor

    X = np.random.default_rng(4).normal(size=(10, 3))
    m = GPyTorchRegressor(normalize_y=True, training_iter=5).fit(
        X, np.sin(X[:, 0]))
    with pytest.raises(NotImplementedError):
        m.log_marginal_likelihood(theta=np.zeros(3))


def test_kernel_init_and_noise_bounds_follow_the_sklearn_kernel():
    pytest.importorskip("gpytorch")
    import gpytorch
    from sklearn.gaussian_process.kernels import (DotProduct, Matern,
                                                  WhiteKernel)
    from gpevo.uspexkit.gp import (_kernel_init, _new_likelihood,
                                   GPyTorchRegressor)

    k = (0.00581 ** 2 * DotProduct(sigma_0=0.412)
         + 0.35 ** 2 * Matern(length_scale=[0.7, 0.7, 0.7], nu=2.5)
         + WhiteKernel(noise_level=0.02,
                       noise_level_bounds=(1e-6, 0.05)))
    init = _kernel_init(k, 3)
    assert np.allclose(init['lengthscale'], [0.7, 0.7, 0.7])
    assert init['noise'] == 0.02
    assert init['bounds']['noise'] == (1e-6, 0.05)
    assert isinstance(_new_likelihood(init['bounds']['noise'])
                      .noise_covar.raw_noise_constraint,
                      gpytorch.constraints.Interval)

    X = np.random.default_rng(5).normal(size=(10, 3))
    m = GPyTorchRegressor(kernel=k, normalize_y=True, random_state=0,
                          training_iter=5).fit(X, np.sin(X[:, 0]))
    assert m._noise_bounds == (1e-6, 0.05)
    assert float(m._likelihood.noise.item()) <= 0.05


def test_writeORG_labels_the_gp_step_gaussianprocess(tmp_path, monkeypatch):
    """`1 : gp` appends a GP step, not a third GULP relaxation."""
    from gpevo.runtime import G, struct
    from gpevo.FunctionFolder.USPEX.src.writeORG import writeORG

    cmd = "mpirun -n 16 gulp< input > output"

    def make_org(gp):
        n = 3 if gp else 2
        return struct(
            varcomp=0, molecule=0, numIons=[8], atomType=[6],
            latVolume=100.0, numGenerations=50, initialPopSize=30,
            populationSize=30, abinitioCode=[3] * n, Kresol=[0.0] * n,
            commandExecutable=[cmd] * (n - 1) + ([""] if gp else [cmd]),
            gp=gp, platform=0, numParallelCalcs=1, ExternalPressure=None,
            resFolder=str(tmp_path), log_file="OUTPUT.txt")

    monkeypatch.setitem(G, "ORG_STRUC", make_org(1))
    rows = [l for l in writeORG().splitlines()
            if l.startswith(("  1", "  2", "  3"))]
    assert len(rows) == 3
    assert "GULP" in rows[0] and "GULP" in rows[1]
    assert "GaussianProcess" in rows[2] and "GULP" not in rows[2]
    # The longer label must not push K-resolution out of alignment.
    assert rows[0].find("0.000") == rows[2].find("0.000")

    monkeypatch.setitem(G, "ORG_STRUC", make_org(0))
    assert "GaussianProcess" not in writeORG()


def _pop_with_one_gp_record():
    from gpevo.runtime import struct
    ind = struct(Number=1, howCome="", gp={
        "density_gp": 1.9, "uncertainty": 0.098, "residual": 1.0,
        "dft": False, "energy_pred": -100.0, "uncertainty_eng": 0.1})
    return struct(generation=2, POPULATION=[ind])


def test_gpplot_is_off_by_default_and_opt_in(tmp_path):
    org = _org_from_input(tmp_path, "1 : gp\n")
    assert org.gpPlot == 0
    assert gp_ml.gp_plot_enabled(org) is False

    org_on = _org_from_input(tmp_path, "1 : gp\n1 : gpplot\n")
    assert org_on.gpPlot == 1
    assert gp_ml.gp_plot_enabled(org_on) is True


def test_gpplot_off_skips_png_but_keeps_diagnostics(tmp_path, capsys):
    from gpevo.FunctionFolder.USPEX.src import gp_plot

    org = _org_from_input(tmp_path, "1 : gp\n")
    org.resFolder = str(tmp_path / "results1")
    os.makedirs(org.resFolder, exist_ok=True)

    assert gp_plot.plot_gp_generation(org, _pop_with_one_gp_record()) is None
    assert not list((tmp_path / "results1").glob("gp_gen*.png"))
    assert (tmp_path / "results1" / "gp_status_history.csv").is_file()
    assert "GP status" in capsys.readouterr().out


def test_gpplot_on_writes_the_png(tmp_path, monkeypatch):
    import matplotlib.figure
    from gpevo.FunctionFolder.USPEX.src import gp_plot

    def fake_savefig(self, *args, **kwargs):
        with open(args[0], "w", encoding="utf-8") as fh:
            fh.write("png")
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fake_savefig)

    org = _org_from_input(tmp_path, "1 : gp\n1 : gpplot\n")
    org.resFolder = str(tmp_path / "results1")
    os.makedirs(org.resFolder, exist_ok=True)

    out = gp_plot.plot_gp_generation(org, _pop_with_one_gp_record())
    assert out and out.endswith("gp_gen2.png")
    assert os.path.isfile(out)
