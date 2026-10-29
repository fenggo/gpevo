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
    """A structure ready for the learning loop; no GP prediction by default."""
    return struct(Number=number, Enthalpies=np.array([-100.0]),
                  numMols=[2], numIons=[2], distance=residual,
                  density_gp=density_gp, uq=sigma, energy_gp=energy_gp,
                  uq_eng=sigma_eng)


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
