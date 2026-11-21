"""Hold out crystals from a GP training set and compare GP vs DFT values.

``gpevo gpt`` is a *GP test*: it reads the ``data`` option of INPUT.txt,
samples ``--n`` crystals from ``<data>/structures.traj``, refits the Gaussian
processes **without** those crystals, and prints for each held-out crystal:

  * DFT density / energy (read straight from ``feature.csv`` -- no new DFT or
    GULP calculation is run), and
  * GP predicted density / energy with the posterior standard deviation.

The feature vector used for a sampled crystal is row ``i`` of
``feature_mlp.csv`` (columns 1..10), i.e. the features computed on the
MLP/GULP-relaxed geometry stored in ``structures_mlp.traj``.  No calculator is
invoked here.

Row alignment is assumed: frame ``i`` of ``structures.traj`` and row ``i`` of
both CSV files describe the same crystal.  We check the counts and widths but
cannot verify the ordering without recomputing the features.

Usage::

    gpevo gpt --n 5
    gpevo gpt -p DIR --n 5
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
from ase.io.trajectory import Trajectory
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from gpevo.inputparser import InputParser
from gpevo.uspexkit.core import load_gaussian_process, _gp_predict


def _resolve_data_dir(path):
    """Locate the GP data directory from INPUT.txt's ``data`` option."""
    run = os.path.abspath(path)
    inp = os.path.join(run, "INPUT.txt")
    if not os.path.isfile(inp):
        raise RuntimeError(f"INPUT.txt not found in {run}")
    raw = InputParser(inp).option("data")
    name = str(raw) if raw else "data"
    data_dir = os.path.join(run, name)
    if not os.path.isdir(data_dir):
        # pred() uses the same fallback when called from a CalcFold<N> folder
        alt = os.path.join(os.path.dirname(run), name)
        if os.path.isdir(alt):
            return alt
        raise RuntimeError(f"data directory not found: {data_dir}")
    return data_dir


def format_report(records, n_total, n_sampled, data_dir):
    """Build the terminal table and the RMSE/MAE summary as a string."""
    lines = [
        f"GP holdout test: data={data_dir}  "
        f"sampled={n_sampled}  train={n_total - n_sampled}",
        "",
        (f"{'row':>4} {'dft_rho':>9} {'gp_rho':>9} {'sd_rho':>8}  "
         f"{'dft_E':>11} {'gp_E':>11} {'sd_E':>9}"),
    ]
    for rec in records:
        lines.append(
            f"{rec['row']:4d} {rec['dft_density']:9.4f} "
            f"{rec['gp_density']:9.4f} {rec['gp_density_sigma']:8.4f}  "
            f"{rec['dft_energy']:11.4f} {rec['gp_energy']:11.4f} "
            f"{rec['gp_energy_sigma']:9.4f}")

    d_err = np.array([r["dft_density"] - r["gp_density"] for r in records])
    e_err = np.array([r["dft_energy"] - r["gp_energy"] for r in records])
    lines.append("")
    lines.append(
        f"rho RMSE={np.sqrt(np.mean(d_err ** 2)):.4f} "
        f"MAE={np.mean(np.abs(d_err)):.4f} g/cm3 | "
        f"E RMSE={np.sqrt(np.mean(e_err ** 2)):.4f} "
        f"MAE={np.mean(np.abs(e_err)):.4f} eV")
    lines.append("(sd = GP posterior sigma; 95% half-width ~= 1.96*sd)")
    return "\n".join(lines)


def gpt(path=".", n=5, seed=0):
    """Refit the GP without ``n`` sampled crystals and report their predictions.

    Returns one record per sampled crystal (see ``format_report`` for the
    printed columns).  ``seed`` keeps sampling reproducible; it is not exposed
    as a CLI option.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise RuntimeError("--n must be a positive integer")

    data_dir = _resolve_data_dir(path)
    for name in ("structures.traj", "feature_mlp.csv", "feature.csv"):
        if not os.path.isfile(os.path.join(data_dir, name)):
            raise RuntimeError(f"missing {name} in {data_dir}")

    try:
        with Trajectory(os.path.join(data_dir, "structures.traj"), "r") as tr:
            n_rows = len(tr)
    except Exception as exc:
        raise RuntimeError(
            f"could not read {os.path.join(data_dir, 'structures.traj')}: {exc}"
        ) from exc

    if n >= n_rows:
        raise RuntimeError(
            f"--n (={n}) must be < number of crystals ({n_rows})")
    if n_rows - n < 2:
        raise RuntimeError(
            f"need at least 2 training rows after holding out {n} crystals "
            f"(have {n_rows})")

    try:
        feature_mlp = np.loadtxt(
            os.path.join(data_dir, "feature_mlp.csv"),
            delimiter=",", skiprows=1, ndmin=2)
        feature = np.loadtxt(
            os.path.join(data_dir, "feature.csv"),
            delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read feature CSVs in {data_dir}: {exc}") from exc

    if feature_mlp.shape != (n_rows, 11) or feature.shape != (n_rows, 11):
        raise RuntimeError(
            f"data mismatch: structures.traj has {n_rows} frames, "
            f"feature_mlp.csv {feature_mlp.shape}, feature.csv {feature.shape} "
            "(expected n_rows x 11 for both CSVs)")

    # Same representation as gp()'s residual check (core.py:817-825):
    # col 0 is the id, cols 1..10 the 10-D features, DFT density is the last
    # column of feature.csv and DFT energy is column 1.
    X_raw = feature_mlp[:, 1:]
    y = feature[:, -1]
    y_eng = feature[:, 1]

    idx = np.sort(
        np.random.default_rng(seed).choice(n_rows, size=n, replace=False))
    mask = np.ones(n_rows, dtype=bool)
    mask[idx] = False

    # Fit the scaler on the training rows only -- fitting it on all rows would
    # leak the held-out feature statistics into the evaluation.
    scaler = StandardScaler().fit(X_raw[mask])
    X_train = scaler.transform(X_raw[mask])
    X_query = scaler.transform(X_raw[idx])

    # load_gaussian_process caches to RELATIVE gpr_*.pkl and overwrites
    # gpcsp.log.  Fit in an empty temporary cwd so a stale full-data model
    # cannot be loaded and the run directory is left untouched.
    previous = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="gpevo_gpt_") as tmp:
        try:
            os.chdir(tmp)
            # Pin BLAS to one thread for the whole fit.  gpevo imports torch
            # (via irff), which raises the thread count to the core count;
            # multithreaded reductions make L-BFGS land in degenerate optima
            # (sd == 0, predictions tens of units off).  core.py applies the
            # same pin inside _load_or_fit_gpr; keep it here too so this
            # held-out evaluation stays reproducible even if that changes.
            with threadpool_limits(limits=1):
                gpr_energy, gpr_density = load_gaussian_process(
                    X_train, y[mask], y_eng[mask])
        finally:
            os.chdir(previous)

    mu_d, sd_d = _gp_predict(gpr_density, X_query)
    mu_e, sd_e = _gp_predict(gpr_energy, X_query)

    records = []
    for row, dft_d, dft_e, gp_d, s_d, gp_e, s_e in zip(
            idx, y[idx], y_eng[idx], mu_d, sd_d, mu_e, sd_e):
        records.append({
            "row": int(row),
            "dft_density": float(dft_d),
            "dft_energy": float(dft_e),
            "gp_density": float(gp_d),
            "gp_density_sigma": float(s_d),
            "gp_energy": float(gp_e),
            "gp_energy_sigma": float(s_e),
        })

    print(format_report(records, n_rows, n, data_dir))
    return records


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="gpevo gpt",
        description="Hold out crystals and compare GP vs DFT density/energy.")
    parser.add_argument("-p", "--path", default=".",
                        help="run directory containing INPUT.txt")
    parser.add_argument("--n", type=int, default=5,
                        help="number of crystals to sample from structures.traj")
    args = parser.parse_args(argv)
    gpt(path=args.path, n=args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
