"""pred() reads the training set once per call, not once per crystal.

GULP, ReaxFF and LAMMPS are never invoked: ``opt``/``get_feature``/
``get_hbond_feature`` are replaced by stubs, which leaves the database reads
and the model loading -- exactly what this test is about -- running for real.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from ase import Atoms
from ase.io import Trajectory

from gpevo.uspexkit import core

_HEADER = ", etot, ebond, eang, etor, evdw, ehb_cho,ehb_chn,ehb_chc, ecoul, density"


def _atoms(n):
    return Atoms("H2O", positions=[[0, 0, 0], [0.8, 0, 0], [0, 0.8, 0]],
                 cell=[6.0, 6.0, 6.0], pbc=True)


def _write_db(data_dir, n_rows):
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, values in (("feature_mlp.csv", rng.normal(size=(n_rows, 10))),
                         ("feature.csv", rng.normal(size=(n_rows, 10)))):
        with open(os.path.join(data_dir, name), "w") as fd:
            fd.write(_HEADER + "\n")
            for i, row in enumerate(values):
                fd.write("{},{}\n".format(i, ",".join("%.6f" % v for v in row)))
    with Trajectory(os.path.join(data_dir, "structures.traj"), "w") as traj:
        for _ in range(n_rows):
            traj.write(_atoms(0))


@pytest.fixture
def run(tmp_path):
    """results1/ (cwd of pred) plus results1-independent <data> database."""
    res_folder = tmp_path / "results1"
    res_folder.mkdir()
    data_dir = res_folder / "data"
    _write_db(str(data_dir), 6)
    with Trajectory(str(res_folder / "Individuals.traj"), "w") as traj:
        for _ in range(3):
            traj.write(_atoms(0))
    return str(res_folder), str(data_dir)


def _stub_features(monkeypatch):
    """Replace every calculator call; features come from a fixed vector."""
    monkeypatch.setattr(core, "opt", lambda atoms, **kw: atoms)
    monkeypatch.setattr(core, "get_feature",
                        lambda atoms, **kw: np.arange(13, dtype=float))
    monkeypatch.setattr(core, "get_hbond_feature",
                        lambda atoms, **kw: np.arange(13, dtype=float))


def test_pred_reuses_a_cached_feature_without_running_reaxff(run, monkeypatch):
    """A feature handed in for a crystal replaces opt() and the four GULP runs."""
    res_folder, data_dir = run
    calls = {"opt": 0, "feature": 0}

    def fake_opt(atoms, **kw):
        calls["opt"] += 1
        return atoms

    def fake_feature(atoms, **kw):
        calls["feature"] += 1
        return np.arange(13, dtype=float)

    monkeypatch.setattr(core, "opt", fake_opt)
    monkeypatch.setattr(core, "get_feature", fake_feature)
    monkeypatch.setattr(core, "get_hbond_feature", fake_feature)

    previous = os.getcwd()
    try:
        os.chdir(res_folder)
        # crystal 2 comes with its feature, 1 and 3 do not
        results = core.pred(t="Individuals.traj", ids="1 2 3", dat="data",
                            step=1, ncpu=1, features={2: np.arange(10, dtype=float)})
    finally:
        os.chdir(previous)

    assert sorted(results) == [1, 2, 3]
    assert calls["opt"] == 2                    # only the two uncached crystals
    assert calls["feature"] == 2 * 4            # 4 GULP runs each, none for #2


def test_pred_without_ids_keeps_the_low_density_crystals(run, monkeypatch):
    """pred() no longer drops crystals below the density threshold: only the
    fitness gates them, so a low-density crystal still gets its row in
    density_predict.log."""
    res_folder, _data_dir = run
    _stub_features(monkeypatch)
    with open(os.path.join(res_folder, "Individuals"), "w") as fd:
        fd.write("Gen   ID    Origin   Composition    Enthalpy   Volume  "
                 "Density   Fitness\n")
        for cid, density, fitness in ((1, 1.700, -1.700),   # below the default
                                      (2, 1.800, -1.800),   # density threshold
                                      (3, 1.950, 0.500)):   # discarded fitness
            fd.write("  5 {:5d}   Random    -1.630  1307.324   {:.3f}     "
                     "{:.3f} 1 0.315 3.465 1.525\n".format(cid, density, fitness))

    previous = os.getcwd()
    try:
        os.chdir(res_folder)
        results = core.pred(t="Individuals.traj", dat="data", step=1, ncpu=1)
    finally:
        os.chdir(previous)

    assert sorted(results) == [1, 2]


def test_pred_reads_the_database_once_per_call(run, monkeypatch):
    res_folder, data_dir = run
    _stub_features(monkeypatch)

    calls = {"loadtxt": 0, "trajectory": 0}
    real_loadtxt = np.loadtxt
    real_traj = core.Trajectory

    def counting_loadtxt(*args, **kwargs):
        calls["loadtxt"] += 1
        return real_loadtxt(*args, **kwargs)

    def counting_traj(*args, **kwargs):
        calls["trajectory"] += 1
        return real_traj(*args, **kwargs)

    monkeypatch.setattr(core.np, "loadtxt", counting_loadtxt)
    monkeypatch.setattr(core, "Trajectory", counting_traj)

    previous = os.getcwd()
    try:
        os.chdir(res_folder)
        results = core.pred(t="Individuals.traj", ids="1 2 3", dat="data",
                            step=1, ncpu=1)
    finally:
        os.chdir(previous)

    assert sorted(results) == [1, 2, 3]
    # 3 crystals: the two csv files are read once, the database trajectory
    # never (it was read per crystal before, and never used).
    assert calls["loadtxt"] == 2
    assert calls["trajectory"] == 1          # Individuals.traj, the query set

# ────────────────────────────────────────────────────────────────────
#  §2.3 standardised residual
# ────────────────────────────────────────────────────────────────────

def test_standardized_residual_is_invariant_to_column_units():
    """A raw distance is dominated by the largest-unit column (the §2.3 bug)."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 10))
    feature = X[0] + 0.1
    sc = StandardScaler().fit(X)
    res, imin, Xq = core.standardized_residual(feature, sc.transform(X), sc)
    assert Xq.shape == (1, 10)
    assert imin == 0

    # Scale one column by 1e6: a RAW squared distance would change by 1e12,
    # the standardised one must be unchanged.
    X2 = X.copy()
    X2[:, 0] *= 1e6
    f2 = feature.copy()
    f2[0] *= 1e6
    sc2 = StandardScaler().fit(X2)
    res2, imin2, _ = core.standardized_residual(f2, sc2.transform(X2), sc2)
    assert imin2 == 0
    assert np.allclose(res2, res)


def test_standardized_residual_is_a_squared_distance():
    from sklearn.preprocessing import StandardScaler

    X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    res, imin, _ = core.standardized_residual(np.array([0.0, 0.0]), Z, sc)
    assert imin == 0
    assert res[0] == pytest.approx(0.0)
    assert res[1] > 0 and res[2] > 0
    assert np.all(res >= 0)

