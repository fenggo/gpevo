"""Tests for the GP training-set bootstrap / per-generation growth loop.

uspexkit is never executed here: it would need GULP, ReaxFF libraries and
SIESTA.  Everything goes through ``gp_ml._call``, which is the single entry
point to uspexkit, so replacing it exercises the real selection, database and
cache-invalidation logic.

Every fixture writes into pytest's ``tmp_path``.  The live run directories
(``/home/feng/tnt`` and friends) are never touched: their checkpoints carry
absolute paths and would be scribbled on.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from gpevo.FunctionFolder.USPEX.src import gp_ml
from gpevo.FunctionFolder.USPEX.src.gp_ml import (
    _invalidate_pkl, _PKL_NAMES, _prune_stale_id_dirs, gp_active_learning,
    gp_db_ready,
)
from gpevo.runtime import struct

_HEADER = ", etot, ebond, eang, etor, evdw, ehb_cho,ehb_chn,ehb_chc, ecoul, density"


def _write_db(data_dir, n_rows):
    """A training set of *n_rows* fake labelled rows."""
    os.makedirs(data_dir, exist_ok=True)
    for name in ("feature_mlp.csv", "feature.csv"):
        with open(os.path.join(data_dir, name), "w") as fd:
            fd.write(_HEADER + "\n")
            for i in range(n_rows):
                fd.write("{}, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.9\n".format(i))
    for name in ("structures.traj", "structures_mlp.traj"):
        open(os.path.join(data_dir, name), "w").close()


def _add_rows(data_dir, n_rows, start=0):
    """Append *n_rows* rows to an existing database."""
    for name in ("feature_mlp.csv", "feature.csv"):
        with open(os.path.join(data_dir, name), "a") as fd:
            for i in range(start, start + n_rows):
                fd.write("{}, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.9\n".format(i))


def test_ei_log_is_written_into_the_results_folder(tmp_path):
    """It belongs to the run that produced it, so not to the home directory."""
    home = tmp_path / "home"
    res = home / "results3"
    res.mkdir(parents=True)
    ORG = struct(homePath=str(home), resFolder=str(res))
    gp_ml._write_ei_log(ORG, 4, [17, 25])
    assert (res / "ei.log").read_text(encoding="utf-8") == "gen 4:17 25\n"
    assert not (home / "ei.log").exists()


def test_ei_log_falls_back_to_the_home_directory(tmp_path):
    """resFolder is empty until createORGStruc runs."""
    home = tmp_path / "home"
    home.mkdir()
    gp_ml._write_ei_log(struct(homePath=str(home), resFolder=""), 1, [7])
    assert (home / "ei.log").read_text(encoding="utf-8") == "gen 1:7\n"


@pytest.fixture
def run(tmp_path):
    """A throwaway run directory: home/results1/<data> plus a trajectory."""
    home = str(tmp_path)
    res_folder = str(tmp_path / "results1")
    data_dir = str(tmp_path / "data")
    os.makedirs(res_folder, exist_ok=True)
    open(os.path.join(res_folder, "Individuals.traj"), "w").close()
    return home, res_folder, data_dir


def _individual(number, residual=1.0, density_gp=None, sigma=0.05,
                energy_gp=-100.0, sigma_eng=0.1):
    """A structure ready for the learning loop; no GP prediction by default.

    ``sigma`` is the posterior std EI consumes; ``uq`` is the reported 95%
    half-width that OUTPUT.txt and the ``u`` gate use.
    """
    return struct(Number=number, Enthalpies=np.array([-100.0]),
                  numMols=[2], numIons=[2], distance=residual,
                  density_gp=density_gp, sigma_gp=sigma,
                  uq=sigma * gp_ml.UNCERT_SCALE, energy_gp=energy_gp,
                  sigma_eng=sigma_eng, uq_eng=sigma_eng * gp_ml.UNCERT_SCALE)


def _org(home, res_folder, *, u=0.04, k=2, mode="ei", nboot=2, den=1.8,
         opt_type=11):
    return struct(gp=1, gpData="data", gpDen=den, gpNcpu=4, gpU=u, gpK=k,
                  gpMode=mode, gpNboot=nboot, homePath=home, resFolder=res_folder,
                  optType=opt_type, opt_sign=1, numMols=[2],
                  specificFolder="Specific")


def _population(numbers):
    return struct(generation=1, POPULATION=[_individual(n) for n in numbers])


class _FakeCalls:
    """Stand-in for uspexkit: records calls and adds rows on demand."""

    def __init__(self, rows_to_add=2, pkl_writer=True):
        self.calls = []
        self.rows_to_add = rows_to_add
        self._added = 0

    def __call__(self, name, cwd=None, **kwargs):
        self.calls.append((name, cwd, kwargs))
        if name == "calc":
            if self.rows_to_add and self._added < self.rows_to_add:
                ids = [int(i) for i in kwargs["ids"].split()]
                data_dir = os.path.join(os.path.dirname(str(cwd)), kwargs["dat"])
                os.makedirs(data_dir, exist_ok=True)
                # uspexkit creates the three files on the very first row
                if not os.path.exists(os.path.join(data_dir, "structures.traj")):
                    for name_ in ("feature_mlp.csv", "feature.csv"):
                        with open(os.path.join(data_dir, name_), "w") as fd:
                            fd.write(_HEADER + "\n")
                    open(os.path.join(data_dir, "structures.traj"), "w").close()
                _add_rows(data_dir, len(ids))
                self._added += len(ids)
            return True, None
        if name == "pred":
            ids = [int(i) for i in kwargs["ids"].split()]
            return True, {i: {"density_gp": 1.75, "uncertainty": 0.05,
                              "energy_pred": -99.0, "uncertainty_eng": 0.1,
                              "residual": 1.0} for i in ids}
        raise AssertionError("unexpected uspexkit call: " + name)

    def names(self):
        return [call[0] for call in self.calls]

    def ids_of(self, name):
        return [call[2]["ids"] for call in self.calls if call[0] == name]


# ────────────────────────────────────────────────────────────────────
#  bootstrap
# ────────────────────────────────────────────────────────────────────

def test_bootstrap_seeds_an_empty_database(run, monkeypatch, capsys):
    home, res_folder, data_dir = run
    # cached models left behind by an earlier generation
    for directory in (res_folder, os.path.join(home, "CalcFold1")):
        os.makedirs(directory, exist_ok=True)
        for name in _PKL_NAMES:
            open(os.path.join(directory, name), "w").close()
    population = _population([1, 2, 3])
    # fitness is -density: id 3 is the best, then 1, then 2.
    fitness = np.array([-1.90, -1.80, -1.95])
    org = _org(home, res_folder, nboot=2)
    fake = _FakeCalls()
    monkeypatch.setattr(gp_ml, "_call", fake)

    assert not gp_db_ready(org)
    out = gp_active_learning(org, population, fitness)

    assert fake.names() == ["calc", "pred"]
    assert fake.ids_of("calc") == ["3 1"]          # best two by fitness
    assert os.path.isdir(data_dir)
    assert gp_ml._db_rows(data_dir) == 2
    # every cached model, in every directory uspexkit writes them
    for directory in (res_folder, os.path.join(home, "CalcFold1")):
        for name in _PKL_NAMES:
            assert not os.path.isfile(os.path.join(directory, name))
    text = capsys.readouterr().out
    assert "seeding it with the best structures" in text
    assert "GP training set: 0 -> 2 rows" in text
    # the re-predicted density replaced the fitness (optType 11: -density)
    assert out[0] == pytest.approx(-1.75)
    assert out[2] == pytest.approx(-1.75)
    assert out[1] == pytest.approx(-1.80)          # not selected, untouched


def test_bootstrap_then_no_second_seed_without_reference_labels(run, monkeypatch):
    home, res_folder, data_dir = run
    population = _population([1, 2, 3])
    fitness = np.array([-1.9, -1.8, -1.95])
    org = _org(home, res_folder)
    # the reference calculation produced no row at all (e.g. siesta missing)
    monkeypatch.setattr(gp_ml, "_call", _FakeCalls(rows_to_add=0))

    out = gp_active_learning(org, population, fitness)

    assert np.array_equal(out, fitness)
    assert not os.path.isdir(data_dir) or gp_ml._db_rows(data_dir) == 0


# ────────────────────────────────────────────────────────────────────
#  EI selection, once the database exists
# ────────────────────────────────────────────────────────────────────

def _population_with_predictions(numbers, mus, sigma=0.05):
    pop = struct(generation=2, POPULATION=[])
    for number, mu in zip(numbers, mus):
        pop.POPULATION.append(_individual(number, density_gp=mu, sigma=sigma))
    return pop


def test_ei_top_k_labels_the_best_crystals(run, monkeypatch):
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)
    ids = [11, 12, 13]
    population = _population_with_predictions(ids, [1.85, 1.95, 1.90])
    fitness = np.array([-1.0, -1.0, -1.0])
    org = _org(home, res_folder, k=2, mode="ei")
    fake = _FakeCalls()
    monkeypatch.setattr(gp_ml, "_call", fake)

    gp_active_learning(org, population, fitness)

    # same sigma: EI grows with mu, hence the two highest densities
    assert fake.ids_of("calc") == ["12 13"]
    assert gp_ml._db_rows(data_dir) == 6


def test_uncertainty_mode_ranks_by_sigma(run, monkeypatch):
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)
    population = struct(generation=2, POPULATION=[
        _individual(11, density_gp=1.90, sigma=0.05),
        _individual(12, density_gp=1.85, sigma=0.50),
        _individual(13, density_gp=1.88, sigma=0.10),
    ])
    org = _org(home, res_folder, k=1, mode="uncertainty")
    fake = _FakeCalls()
    monkeypatch.setattr(gp_ml, "_call", fake)

    gp_active_learning(org, population, np.array([-1.0, -1.0, -1.0]))

    assert fake.ids_of("calc") == ["12"]


def test_no_selection_below_the_uncertainty_threshold(run, monkeypatch, capsys):
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)
    population = _population_with_predictions([11, 12], [1.9, 1.95], sigma=0.001)
    org = _org(home, res_folder, u=0.04)
    fake = _FakeCalls()
    monkeypatch.setattr(gp_ml, "_call", fake)

    gp_active_learning(org, population, np.array([-1.0, -1.0]))

    assert fake.calls == []                       # nothing labelled, nothing refit
    assert "no crystal above the uncertainty threshold" in capsys.readouterr().out


def test_density_gate_stops_the_selection(run, monkeypatch, capsys):
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)
    population = _population_with_predictions([11, 12], [1.5, 1.6])
    org = _org(home, res_folder, den=1.8)
    fake = _FakeCalls()
    monkeypatch.setattr(gp_ml, "_call", fake)

    gp_active_learning(org, population, np.array([-1.0, -1.0]))

    assert fake.calls == []
    assert "density threshold gate" in capsys.readouterr().out


def test_optType1_ranks_by_gp_energy(run, monkeypatch, capsys):
    """optType 1 minimises the enthalpy: EI has to run on the GP energy."""
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)
    # identical densities: a density-driven EI could not tell them apart,
    # and the lowest energy is the one with the *highest* EI.
    population = struct(generation=2, POPULATION=[
        _individual(11, density_gp=1.90, sigma=0.05, energy_gp=-5.0, sigma_eng=0.10),
        _individual(12, density_gp=1.90, sigma=0.05, energy_gp=-6.0, sigma_eng=0.10),
        _individual(13, density_gp=1.90, sigma=0.05, energy_gp=-5.5, sigma_eng=0.10),
    ])
    org = _org(home, res_folder, k=1, mode="ei", opt_type=1)
    fake = _FakeCalls()
    monkeypatch.setattr(gp_ml, "_call", fake)

    gp_active_learning(org, population, np.array([-1.0, -1.0, -1.0]))

    assert fake.ids_of("calc") == ["12"]        # lowest enthalpy
    assert "eng_gp" in capsys.readouterr().out


def test_optType1_ranks_uncertainty_by_the_energy_one(run, monkeypatch):
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)
    population = struct(generation=2, POPULATION=[
        _individual(11, density_gp=1.90, sigma=0.50, energy_gp=-5.0, sigma_eng=0.10),
        _individual(12, density_gp=1.90, sigma=0.05, energy_gp=-6.0, sigma_eng=0.40),
    ])
    org = _org(home, res_folder, k=1, mode="uncertainty", opt_type=1)
    fake = _FakeCalls()
    monkeypatch.setattr(gp_ml, "_call", fake)

    gp_active_learning(org, population, np.array([-1.0, -1.0]))

    # sigma_eng, not the density uncertainty, decides
    assert fake.ids_of("calc") == ["12"]


def test_negative_optType1_maximises_the_energy(run, monkeypatch):
    """`-1 : optType` maximises the enthalpy: the sign has to flip."""
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)
    population = struct(generation=2, POPULATION=[
        _individual(11, density_gp=1.90, sigma=0.05, energy_gp=-5.0, sigma_eng=0.10),
        _individual(12, density_gp=1.90, sigma=0.05, energy_gp=-6.0, sigma_eng=0.10),
    ])
    org = _org(home, res_folder, k=1, mode="ei", opt_type=1)
    org.opt_sign = -1
    fake = _FakeCalls()
    monkeypatch.setattr(gp_ml, "_call", fake)

    gp_active_learning(org, population, np.array([-1.0, -1.0]))

    assert fake.ids_of("calc") == ["11"]        # highest enthalpy


def test_optType1_falls_back_to_density_without_energy(run, monkeypatch, capsys):
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)
    population = struct(generation=2, POPULATION=[
        _individual(11, density_gp=1.90, sigma=0.05, energy_gp=float("nan")),
        _individual(12, density_gp=1.95, sigma=0.05, energy_gp=float("nan")),
    ])
    org = _org(home, res_folder, k=1, mode="ei", opt_type=1)
    fake = _FakeCalls()
    monkeypatch.setattr(gp_ml, "_call", fake)

    gp_active_learning(org, population, np.array([-1.0, -1.0]))

    assert fake.ids_of("calc") == ["12"]
    assert "falls back to the density" in capsys.readouterr().out


def test_missing_reference_run_never_retrains(run, monkeypatch, capsys):
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)
    population = _population_with_predictions([11, 12], [1.9, 1.95])
    org = _org(home, res_folder)
    open(os.path.join(res_folder, "gpr_density.pkl"), "w").close()
    monkeypatch.setattr(gp_ml, "_call", _FakeCalls(rows_to_add=0))

    gp_active_learning(org, population, np.array([-1.0, -1.0]))

    assert os.path.isfile(os.path.join(res_folder, "gpr_density.pkl"))
    assert "training set is still 4 row(s)" in capsys.readouterr().out


# ────────────────────────────────────────────────────────────────────
#  learning curve
# ────────────────────────────────────────────────────────────────────

def test_pred_receives_the_features_the_gp_step_computed(run, monkeypatch):
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)
    population = struct(generation=2, POPULATION=[
        _individual(11, density_gp=1.90, sigma=0.05),
        _individual(12, density_gp=1.95, sigma=0.05),
    ])
    # what submitJob stores after each structure's GP step
    for number, s in zip([11, 12], population.POPULATION):
        gp_ml.apply_gp_metrics(s, {"residual": 1.0, "density_gp": s.density_gp,
                                   "uncertainty": 0.05, "energy_pred": -100.0,
                                   "uncertainty_eng": 0.1,
                                   "feature": [float(number)] * 10})
    org = _org(home, res_folder, k=1)
    fake = _FakeCalls()
    monkeypatch.setattr(gp_ml, "_call", fake)

    gp_active_learning(org, population, np.array([-1.0, -1.0]))

    pred_calls = [c for c in fake.calls if c[0] == "pred"]
    features = pred_calls[0][2]["features"]
    assert sorted(features) == [11, 12]
    assert features[11] == [11.0] * 10


def test_feature_cache_is_switched_off_by_input(run, monkeypatch):
    """`0 : cache` in INPUT.txt must fall back to recomputing."""
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)
    population = struct(generation=2, POPULATION=[
        _individual(11, density_gp=1.90, sigma=0.05),
        _individual(12, density_gp=1.95, sigma=0.05),
    ])
    for s in population.POPULATION:
        gp_ml.apply_gp_metrics(s, {"residual": 1.0, "density_gp": s.density_gp,
                                   "uncertainty": 0.05, "energy_pred": -100.0,
                                   "uncertainty_eng": 0.1,
                                   "feature": [1.0] * 10})
    org = _org(home, res_folder, k=1)
    org.gpCache = 0
    fake = _FakeCalls()
    monkeypatch.setattr(gp_ml, "_call", fake)

    gp_active_learning(org, population, np.array([-1.0, -1.0]))

    pred_calls = [c for c in fake.calls if c[0] == "pred"]
    assert pred_calls[0][2]["features"] is None


def test_loo_rmses_reads_the_models_uspexkit_cached(tmp_path):
    """End to end: fit like uspexkit does, then score that very pickle."""
    from gpevo.uspexkit import core

    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(3)
    features = rng.normal(size=(8, 10))
    density = 1.8 + 0.05 * rng.normal(size=8)
    energy = -28000.0 + 3.0 * rng.normal(size=8)
    for name, mat in (("feature_mlp.csv", np.column_stack([np.arange(8), features])),
                      ("feature.csv", np.column_stack(
                          [np.arange(8), energy, features[:, 1:9], density]))):
        np.savetxt(os.path.join(data_dir, name), mat, delimiter=",",
                   header=_HEADER, comments="", fmt="%.6f")

    previous = os.getcwd()
    try:
        os.chdir(data_dir)
        core.load_gaussian_process(features, density, energy)
    finally:
        os.chdir(previous)

    rmses = gp_ml._loo_rmses(data_dir)
    assert rmses is not None
    assert rmses[0] == pytest.approx(0.05, abs=0.05)     # density, g/cm^3
    assert rmses[1] == pytest.approx(3.0, abs=3.0)       # energy, eV
    # a directory without the models reports nothing instead of raising
    assert gp_ml._loo_rmses(str(tmp_path)) is None


def test_learning_curve_reports_the_row_delta(run, monkeypatch, capsys):
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)
    population = _population_with_predictions([11, 12], [1.9, 1.95])
    org = _org(home, res_folder, k=1)
    monkeypatch.setattr(gp_ml, "_call", _FakeCalls())

    gp_active_learning(org, population, np.array([-1.0, -1.0]))

    out = capsys.readouterr().out
    assert "GP learning curve (gen 2): rows 4 -> 5 (+1)" in out
    assert "LOO RMSE" in out


def test_learning_curve_is_printed_when_a_gate_blocks(run, monkeypatch, capsys):
    """A generation that labels nothing must still say so: rows 4 -> 4 (+0)."""
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)
    population = _population_with_predictions([11, 12], [1.5, 1.6])
    org = _org(home, res_folder, den=1.8)
    fake = _FakeCalls()
    monkeypatch.setattr(gp_ml, "_call", fake)

    gp_active_learning(org, population, np.array([-1.0, -1.0]))

    assert fake.calls == []
    assert "rows 4 -> 4 (+0)" in capsys.readouterr().out


def _bruteforce_loo(X, y, normalize_y):
    """Refit without every point in turn: the slow reference for _loo_rmse."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    kernel = 1.0 * RBF(length_scale=[1.0] * X.shape[1]) + WhiteKernel(noise_level=1e-3)

    # optimizer=None pins the kernel: the identity holds for a fixed kernel,
    # and re-optimising each fold would measure the optimiser, not the formula.
    def fit(X_, y_):
        return GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y,
                                        optimizer=None, random_state=0).fit(X_, y_)

    errors = []
    for i in range(len(y)):
        keep = np.ones(len(y), dtype=bool)
        keep[i] = False
        errors.append(y[i] - float(fit(X[keep], y[keep]).predict(X[i:i + 1])[0]))
    return float(np.sqrt(np.mean(np.square(errors))))


def test_loo_rmse_matches_a_bruteforce_leave_one_out():
    """The hat-matrix shortcut must equal refitting without each point."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 3))
    y = 2.0 * np.sin(X[:, 0]) + 0.3 * X[:, 1]
    kernel = 1.0 * RBF(length_scale=[1.0] * 3) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                   optimizer=None, random_state=0).fit(X, y)

    assert gp_ml._loo_rmse(gpr) == pytest.approx(
        _bruteforce_loo(X, y, normalize_y=False), rel=1e-6)


def test_loo_rmse_of_a_normalised_model_is_close_to_the_refit_reference():
    """Production models use normalize_y=True; the two then differ slightly.

    Refitting each fold re-centres the target with that fold's own mean, so it
    scores a different model than the one fitted on all rows.  The shortcut
    keeps the full-set centring, which is what is being reported.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 3))
    y = 2.0 * np.sin(X[:, 0]) + 0.3 * X[:, 1]
    kernel = 1.0 * RBF(length_scale=[1.0] * 3) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   optimizer=None, random_state=0).fit(X, y)

    reference = _bruteforce_loo(X, y, normalize_y=True)
    assert gp_ml._loo_rmse(gpr) == pytest.approx(reference, rel=0.15)
    assert gp_ml._loo_rmse(gpr) != pytest.approx(reference, rel=1e-3)


# ────────────────────────────────────────────────────────────────────
#  helpers
# ────────────────────────────────────────────────────────────────────

def test_prune_stale_id_dirs_keeps_finished_runs(tmp_path):
    stale = tmp_path / "7"
    stale.mkdir()
    done = tmp_path / "8"
    done.mkdir()
    (done / "POSCAR.8").write_text("EA8")

    pruned = _prune_stale_id_dirs(str(tmp_path), [7, 8, 9])

    assert [os.path.basename(p) for p in pruned] == ["7"]
    assert not stale.exists()
    assert done.exists()


def test_invalidate_pkl_covers_every_cache_location(tmp_path):
    """The pickles gp() and pred() leave in the run tree, all of them."""
    home = tmp_path
    directories = [str(tmp_path / "data"), str(tmp_path / "results1")]
    for extra in ("CalcFold1", "CalcFoldTemp"):
        directories.append(str(tmp_path / extra))
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        for name in _PKL_NAMES:
            with open(os.path.join(directory, name), "w") as fd:
                fd.write("stale")

    removed = _invalidate_pkl(str(home), str(tmp_path / "results1"),
                              str(tmp_path / "data"))

    assert len(removed) == 4 * len(_PKL_NAMES)
    leftover = [str(p) for p in tmp_path.rglob("*.pkl")]
    assert leftover == []


def test_gp_db_ready_needs_two_rows(run):
    home, res_folder, data_dir = run
    org = _org(home, res_folder)
    assert not gp_db_ready(org)
    _write_db(data_dir, 1)
    assert not gp_db_ready(org)          # a single row breaks uspexkit's data[:, 1:]
    _add_rows(data_dir, 1, start=1)
    assert gp_db_ready(org)


def test_explore_off_drops_only_the_sigma_term():
    """EI with and without sigma*phi(Z); the improvement term is untouched."""
    mu, sigma, f_best = 1.82, 0.50, 1.90
    diff = mu - f_best
    z = diff / sigma
    phi_z = 0.5 * math.erfc(-z / math.sqrt(2.0))
    pdf_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    assert gp_ml._expected_improvement(mu, sigma, f_best) == pytest.approx(
        diff * phi_z + sigma * pdf_z)
    assert gp_ml._expected_improvement(mu, sigma, f_best, explore=False) \
        == pytest.approx(diff * phi_z)


def test_exploration_term_lets_a_bad_crystal_win(run, monkeypatch):
    """The reason `explore : 0` exists.

    Crystal 11 is predicted *worse* than the observed incumbent (1.82 < 1.9)
    but its posterior is ten times wider: sigma*phi(Z) = 0.197 against an
    improvement term of -0.035, so it outranks a crystal that is genuinely
    better.  That is an expensive reference calculation spent on junk.
    """
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)                  # observed incumbent density = 1.9
    population = struct(generation=2, POPULATION=[
        _individual(11, density_gp=1.82, sigma=0.50),
        _individual(12, density_gp=1.95, sigma=0.05),
    ])
    fake = _FakeCalls()
    monkeypatch.setattr(gp_ml, "_call", fake)

    gp_active_learning(_org(home, res_folder, k=1, mode="ei"), population,
                       np.array([-1.0, -1.0]))

    assert fake.ids_of("calc") == ["11"]


def test_explore_off_ranks_by_the_improvement_term(run, monkeypatch):
    """`explore : 0`: the wide-but-worse crystal loses to the better one."""
    home, res_folder, data_dir = run
    _write_db(data_dir, 4)
    population = struct(generation=2, POPULATION=[
        _individual(11, density_gp=1.82, sigma=0.50),
        _individual(12, density_gp=1.95, sigma=0.05),
    ])
    org = _org(home, res_folder, k=1, mode="ei")
    org.gpExplore = 0
    fake = _FakeCalls()
    monkeypatch.setattr(gp_ml, "_call", fake)

    gp_active_learning(org, population, np.array([-1.0, -1.0]))

    assert fake.ids_of("calc") == ["12"]


def test_explore_is_parsed_from_input(tmp_path):
    from gpevo.runtime import G
    from gpevo.FunctionFolder.USPEX.src.createORG_AbinitCode import createORG_AbinitCode
    from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault

    input_file = tmp_path / "INPUT.txt"
    input_file.write_text("1 : gp\n0 : explore\n")
    G["ORG_STRUC"] = createORGDefault()
    G["ORG_STRUC"].homePath = str(tmp_path)

    createORG_AbinitCode(str(input_file))

    assert G["ORG_STRUC"].gpExplore == 0
    assert gp_ml.parse_gp_params(G["ORG_STRUC"])["explore"] is False


def test_explore_defaults_to_on(tmp_path):
    from gpevo.runtime import G
    from gpevo.FunctionFolder.USPEX.src.createORG_AbinitCode import createORG_AbinitCode
    from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault

    input_file = tmp_path / "INPUT.txt"
    input_file.write_text("1 : gp\n")
    G["ORG_STRUC"] = createORGDefault()
    G["ORG_STRUC"].homePath = str(tmp_path)

    createORG_AbinitCode(str(input_file))

    assert G["ORG_STRUC"].gpExplore == 1
    assert gp_ml.parse_gp_params(G["ORG_STRUC"])["explore"] is True


def test_nboot_is_parsed_from_input(tmp_path):
    from gpevo.runtime import G
    from gpevo.FunctionFolder.USPEX.src.createORG_AbinitCode import createORG_AbinitCode
    from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault

    input_file = tmp_path / "INPUT.txt"
    input_file.write_text("1 : gp\n5 : nboot\n")
    G["ORG_STRUC"] = createORGDefault()
    G["ORG_STRUC"].homePath = str(tmp_path)

    createORG_AbinitCode(str(input_file))

    assert G["ORG_STRUC"].gp == 1
    assert G["ORG_STRUC"].gpNboot == 5


def test_nboot_defaults_to_two(tmp_path):
    from gpevo.runtime import G
    from gpevo.FunctionFolder.USPEX.src.createORG_AbinitCode import createORG_AbinitCode
    from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault

    input_file = tmp_path / "INPUT.txt"
    input_file.write_text("1 : gp\n")
    G["ORG_STRUC"] = createORGDefault()
    G["ORG_STRUC"].homePath = str(tmp_path)

    createORG_AbinitCode(str(input_file))

    assert G["ORG_STRUC"].gpNboot == 2

# ────────────────────────────────────────────────────────────────────
#  sigma semantics: EI uses posterior sigma, the u gate uses 1.96*sigma
# ────────────────────────────────────────────────────────────────────

def test_gp_metrics_keep_both_sigma_and_95pct_width():
    s = struct()
    gp_ml.apply_gp_metrics(s, {"residual": 1.0, "density_gp": 1.8,
                               "uncertainty": 0.098, "energy_pred": -10.0,
                               "uncertainty_eng": 0.196})
    assert s.uq == pytest.approx(0.098)                    # OUTPUT.txt / u gate
    assert s.sigma_gp == pytest.approx(0.098 / 1.96)      # EI
    assert s.uq_eng == pytest.approx(0.196)
    assert s.sigma_eng == pytest.approx(0.196 / 1.96)


def test_pred_metrics_convert_the_uncertainty_too():
    s = struct()
    gp_ml.apply_pred_metrics(s, {"residual": 1.0, "density_gp": 1.9,
                                 "uncertainty": 0.049, "energy_pred": -11.0,
                                 "uncertainty_eng": 0.098})
    assert s.sigma_gp == pytest.approx(0.025)
    assert s.sigma_eng == pytest.approx(0.05)


def test_metrics_row_prefers_the_posterior_sigma_and_falls_back():
    s = _individual(1, density_gp=1.9, sigma=0.03)
    assert gp_ml._metrics_row(s)[gp_ml._UNCERT] == pytest.approx(0.03)
    assert gp_ml._metrics_row(s)[gp_ml._UNCERT_ENG] == pytest.approx(0.1)
    # an older checkpoint without sigma_gp must still yield sigma
    del s.sigma_gp
    assert gp_ml._metrics_row(s)[gp_ml._UNCERT] == pytest.approx(0.03)


def test_read_gp_csv_converts_the_95pct_columns(tmp_path):
    path = tmp_path / "gp.csv"
    path.write_text(
        "index, nearest, residual, density_min, density_rf, density_gp, "
        "uncertainty, energy_min, energy_pred, uncertainty_eng\n"
        "11, 0, 1.0, 1.7, 1.75, 1.80, 0.098, -10.0, -10.1, 0.196\n",
        encoding="utf-8")
    table = gp_ml._read_gp_csv(str(path))
    assert table[11][gp_ml._UNCERT] == pytest.approx(0.05)
    assert table[11][gp_ml._UNCERT_ENG] == pytest.approx(0.10)


def _ei_table(sigma):
    nan = float("nan")
    return {11: [nan, 1.0, nan, nan, 1.90, sigma, nan, -10.0, sigma]}


def test_ei_uses_posterior_sigma_but_u_gate_uses_the_95pct_width():
    params = gp_ml.parse_gp_params(_org(".", ".", u=0.04, k=1, den=1.8))
    selected, _label, reason, _target = gp_ml._select_by_ei(
        [11], _ei_table(0.03), params)
    assert reason is None
    assert selected[0][1] == pytest.approx(0.03)           # posterior sigma
    # 1.96*0.019 = 0.0372 < u=0.04: the gate stays on the reported scale.
    # (If u had been switched to sigma, 0.019 would also fail, but 0.03 would
    # have been the one to check: 0.03 < 0.04 would have been rejected.)
    selected, _label, reason, _target = gp_ml._select_by_ei(
        [11], _ei_table(0.019), params)
    assert reason is not None and "uncertainty threshold" in reason


def test_ei_preview_uses_the_posterior_std():
    from gpevo.FunctionFolder.USPEX.src import gp_plot

    params = gp_ml.parse_gp_params(_org(".", ".", u=0.04, k=1, den=1.8))
    records = [{"number": 11, "dgp": 1.90, "sigma": 0.098, "sd": 0.05,
                "residual": 1.0}]
    selected, _f_best, _top = gp_plot._ei_preview(records, params)
    assert len(selected) == 1
    assert selected[0][0] == pytest.approx(
        gp_ml._expected_improvement(1.90, 0.05, 1.90))

# ────────────────────────────────────────────────────────────────────
#  f_best: same incumbent in selection and report, plus calibration
# ────────────────────────────────────────────────────────────────────

def test_observed_density_incumbent_reads_the_dft_best(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "feature.csv").write_text(
        _HEADER + "\n"
        "0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.80\n"
        "1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.8991\n"
        "2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.85\n",
        encoding="utf-8")
    org = _org(str(tmp_path), str(tmp_path / "results1"), opt_type=11)
    assert gp_ml.observed_density_incumbent(org) == pytest.approx(1.8991)


def test_ei_preview_uses_the_observed_incumbent():
    from gpevo.FunctionFolder.USPEX.src import gp_plot

    params = gp_ml.parse_gp_params(_org(".", ".", u=0.001, k=1, den=1.0))
    records = [{"number": 11, "dgp": 1.8032, "sigma": 0.098, "sd": 0.05,
                "residual": 1.0}]
    _, f_best_fallback, _ = gp_plot._ei_preview(records, params)
    _, f_best_observed, _ = gp_plot._ei_preview(records, params,
                                                 incumbent=1.8991)
    assert f_best_fallback == pytest.approx(1.8032)
    assert f_best_observed == pytest.approx(1.8991)


def test_select_by_ei_reports_the_incumbent_gap(run, capsys):
    _home, _res_folder, data_dir = run
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "feature.csv"), "w",
              encoding="utf-8") as fd:
        fd.write(_HEADER + "\n"
                 "0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.8991\n")
    params = gp_ml.parse_gp_params(
        _org(str(_home), _res_folder, k=1, den=1.0, u=0.001))
    nan = float("nan")
    table = {11: [nan, 1.0, nan, nan, 1.8032, 0.05, nan, -10.0, 0.05]}
    gp_ml._select_by_ei([11], table, params, data_dir)

    out = capsys.readouterr().out
    assert "max predicted this generation" in out
    assert "EI ~ exploration" in out


def test_gp_calibration_regression_prints_after_three_labels(tmp_path, capsys):
    res = str(tmp_path / "results1")
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, "density.log"), "w", encoding="utf-8") as fd:
        fd.write("# Crystal_id Density Energy\n")
        for cid, y in ((1, 1.80), (2, 1.85), (3, 1.90)):
            fd.write(f"{cid} {y:.6f} -100.0\n")
    gp_ml._CALIBRATION.clear()
    gp_ml._CALIBRATION_DIR = None

    gp_ml._record_gp_calibration(res, {1, 2, 3},
                                 {1: 1.75, 2: 1.80, 3: 1.85})

    out = capsys.readouterr().out
    assert "GP calibration (n=3 labelled)" in out
    assert "mean(mu-y) = -0.0500" in out

# ────────────────────────────────────────────────────────────────────
#  §2.3 shared residual cutoff (value restored to 10.0)
# ────────────────────────────────────────────────────────────────────

def test_residual_cut_is_a_single_shared_constant():
    from gpevo.runtime import RESIDUAL_CUT
    from gpevo.FunctionFolder.USPEX.src import gp_plot
    from gpevo.tools import ei_report
    from gpevo.uspexkit import core

    assert RESIDUAL_CUT == 10.0
    assert gp_ml.RESIDUAL_CUT is RESIDUAL_CUT
    assert gp_plot.RESIDUAL_CUT is RESIDUAL_CUT
    assert ei_report.RESIDUAL_CUT is RESIDUAL_CUT
    assert core.RESIDUAL_CUT is RESIDUAL_CUT


def test_ei_filter_uses_the_shared_cutoff():
    from gpevo.runtime import RESIDUAL_CUT

    params = gp_ml.parse_gp_params(_org(".", ".", u=0.001, k=1, den=1.0))
    nan = float("nan")

    def _table(residual):
        return {11: [nan, residual, nan, nan, 1.90, 0.05, nan, -10.0, 0.05]}

    # A residual above the shared cutoff must be rejected.
    _sel, _label, reason, _target = gp_ml._select_by_ei(
        [11], _table(RESIDUAL_CUT + 2), params)
    assert reason is not None and "no crystal with a usable" in reason

    _sel, _label, reason, _target = gp_ml._select_by_ei(
        [11], _table(RESIDUAL_CUT - 1), params)
    assert reason is None



