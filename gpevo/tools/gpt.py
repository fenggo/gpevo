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

The refit honours INPUT.txt's ``gpbackend`` / ``gpiter`` / ``gplr`` -- the same
options ``gp_ml.parse_gp_params`` passes to ``gp()`` and ``pred()`` -- so the
holdout score measures the surrogate the production crystal-prediction loop
actually uses.  The backend is printed in the report header.  If ``gpytorch``
    is selected but not installed the command fails loudly rather than silently
    scoring sklearn.  GPyTorch is now the default when INPUT.txt has no
    ``gpbackend``; set ``gpbackend : sklearn`` for MATLAB-parity runs.

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

from gpevo.inputparser import InputParser
from gpevo.uspexkit.core import load_gaussian_process, _gp_predict


def _gpytorch_is_installed():
    import importlib.util
    return importlib.util.find_spec("gpytorch") is not None


def resolve_data_dir(path):
    """Locate the training database from INPUT.txt's ``data`` option.

    Shared by gpt() and rft() so both holdout tests read exactly the same
    rows.  pred() uses the same fallback when called from a CalcFold<N> folder.
    """
    run = os.path.abspath(path)
    inp = os.path.join(run, "INPUT.txt")
    if not os.path.isfile(inp):
        raise RuntimeError(f"INPUT.txt not found in {run}")
    raw = InputParser(inp).option("data")
    name = str(raw) if raw else "data"
    data_dir = os.path.join(run, name)
    if not os.path.isdir(data_dir):
        alt = os.path.join(os.path.dirname(run), name)
        if os.path.isdir(alt):
            return alt
        raise RuntimeError(f"data directory not found: {data_dir}")
    return data_dir


def prepare_holdout(path, n, seed):
    """Validate the database and build the standardised train/query split.

    Shared verbatim by gpt() and rft() so the two models are scored on the
    identical holdout sample, features and DFT targets.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise RuntimeError("--n must be a positive integer")

    data_dir = resolve_data_dir(path)
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
    y_density = feature[:, -1]
    y_energy = feature[:, 1]

    idx = np.sort(
        np.random.default_rng(seed).choice(n_rows, size=n, replace=False))
    mask = np.ones(n_rows, dtype=bool)
    mask[idx] = False

    # Fit the scaler on the training rows only -- fitting it on all rows would
    # leak the held-out feature statistics into the evaluation.
    scaler = StandardScaler().fit(X_raw[mask])
    return {
        "data_dir": data_dir,
        "n_rows": n_rows,
        "idx": idx,
        "mask": mask,
        "X_train": scaler.transform(X_raw[mask]),
        "X_query": scaler.transform(X_raw[idx]),
        "y_density": y_density,
        "y_energy": y_energy,
    }


def _read_input(path):
    """Return ``(data_dir, gp_options)`` from the run's INPUT.txt.

    ``gp_options`` mirrors the keys ``gp_ml.parse_gp_params`` hands to gp()
    and pred(), so the holdout test refits the same surrogate as production
    instead of silently defaulting to sklearn.
    """
    data_dir = resolve_data_dir(path)
    inp = os.path.join(os.path.abspath(path), "INPUT.txt")
    parser = InputParser(inp)

    raw_backend = parser.option("gpbackend")
    backend = str(raw_backend).lower() if raw_backend else "gpytorch"
    if backend not in ("sklearn", "gpytorch"):
        raise RuntimeError(
            f"unknown gpbackend '{raw_backend}' in {inp}; "
            "expected 'sklearn' or 'gpytorch'")
    try:
        training_iter = max(1, int(float(parser.option("gpiter") or 200)))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid gpiter in {inp}: {exc}") from exc
    try:
        learning_rate = float(parser.option("gplr") or 0.1)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid gplr in {inp}: {exc}") from exc

    options = {"backend": backend, "training_iter": training_iter,
               "learning_rate": learning_rate}
    if backend == "gpytorch" and not _gpytorch_is_installed():
        raise RuntimeError(
            "GP backend 'gpytorch' (explicit or default) selected but gpytorch "
            "is not installed; install it with `pip install gpytorch` (or "
            "`pip install gpevo[gpytorch]`), or set 'gpbackend : sklearn' in "
            "INPUT.txt. Refusing to fall back to sklearn silently: the holdout "
            "test would then score a model the production gp() never uses.")
    return data_dir, options


def format_report(records, n_total, n_sampled, data_dir, options=None,
                  model="GP", pred_key="gp", sd_note=None):
    """Build the terminal table and the RMSE/MAE summary as a string.

    ``model`` / ``pred_key`` let rft() reuse the same layout for RandomForest
    predictions (``rf_*`` record keys) without duplicating the table code.
    """
    options = options or {}
    backend = options.get("backend")
    extra = ""
    if backend:
        extra = f"  backend={backend}"
        if backend == "gpytorch":
            extra += (f"  iter={options.get('training_iter', 200)} "
                      f"lr={options.get('learning_rate', 0.1)}")
    header = (f"{model} holdout test: data={data_dir}{extra}  "
              f"sampled={n_sampled}  train={n_total - n_sampled}")
    lines = [
        header,
        "",
        (f"{'row':>4} {'dft_rho':>9} {pred_key + '_rho':>9} {'sd_rho':>8}  "
         f"{'dft_E':>11} {pred_key + '_E':>11} {'sd_E':>9}"),
    ]
    for rec in records:
        lines.append(
            f"{rec['row']:4d} {rec['dft_density']:9.4f} "
            f"{rec[pred_key + '_density']:9.4f} "
            f"{rec[pred_key + '_density_sigma']:8.4f}  "
            f"{rec['dft_energy']:11.4f} "
            f"{rec[pred_key + '_energy']:11.4f} "
            f"{rec[pred_key + '_energy_sigma']:9.4f}")

    d_err = np.array([r["dft_density"] - r[pred_key + "_density"]
                      for r in records])
    e_err = np.array([r["dft_energy"] - r[pred_key + "_energy"]
                      for r in records])
    lines.append("")
    lines.append(
        f"rho RMSE={np.sqrt(np.mean(d_err ** 2)):.4f} "
        f"MAE={np.mean(np.abs(d_err)):.4f} g/cm3 | "
        f"E RMSE={np.sqrt(np.mean(e_err ** 2)):.4f} "
        f"MAE={np.mean(np.abs(e_err)):.4f} eV")
    lines.append(sd_note or "(sd = GP posterior sigma; 95% half-width ~= 1.96*sd)")
    return "\n".join(lines)


def _parity_metrics(y_pred, y_true):
    """MAE, RMSE and R^2 of a parity plot."""
    resid = y_pred - y_true
    mae = np.mean(np.abs(resid))
    rmse = np.sqrt(np.mean(resid ** 2))
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def _linear_fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _save_parity_figure(x, y, sigma, xlabel, ylabel, unit, outpath,
                        pad_frac, decimals, scatter_color="#0072B2"):
    """One publication-quality parity scatter with model uncertainty bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.2, "lines.markersize": 5.5,
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.direction": "in", "ytick.direction": "in",
    }
    dark, orange = "#333333", "#D55E00"

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        ax.errorbar(x, y, yerr=1.96 * sigma, fmt="none", ecolor=scatter_color,
                    elinewidth=0.4, capsize=0, alpha=0.20, zorder=1)
        ax.scatter(x, y, c=scatter_color, s=30, edgecolors="white",
                   linewidths=0.3, alpha=0.85, zorder=3)

        all_xy = np.concatenate([x, y])
        lims = [all_xy.min(), all_xy.max()]
        pad = (lims[1] - lims[0]) * pad_frac
        lims = [lims[0] - pad, lims[1] + pad]
        ax.plot(lims, lims, color=dark, linestyle="--", linewidth=0.9,
                alpha=0.6, zorder=2, label="y = x")

        slope, intercept = _linear_fit(x, y)
        x_fit = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_fit, slope * x_fit + intercept, color=orange,
                linestyle="-", linewidth=1.0, alpha=0.8, zorder=2,
                label=f"Linear fit (slope = {slope:.3f})")

        mae, rmse, r2 = _parity_metrics(y, x)
        f = f".{decimals}f"
        textstr = (r"$\mathrm{MAE}$" + f" = {mae:{f}} {unit}\n"
                   r"$\mathrm{RMSE}$" + f" = {rmse:{f}} {unit}\n"
                   r"$R^2$" + f" = {r2:.4f}")
        props = dict(boxstyle="round,pad=0.4", facecolor="white",
                     edgecolor="#999999", alpha=0.92, linewidth=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", bbox=props)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", frameon=True, fancybox=False,
                  edgecolor="#333333", framealpha=0.9)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.4)

        fig.tight_layout()
        fig.savefig(outpath, transparent=True)
        plt.close(fig)
    return mae, rmse, r2, slope


def save_parity_plots(records, outdir=".", prefix="test_gp", model="GP",
                      pred_key="gp", scatter_color="#0072B2"):
    """Write ``<prefix>_energy.svg`` / ``<prefix>_density.svg`` parity plots.

    Shared by gpt() and rft(): identical styling, only the model label, the
    output prefix, the prediction record keys and the scatter colour differ.
    """
    e_dft = np.array([r["dft_energy"] for r in records])
    e_pred = np.array([r[pred_key + "_energy"] for r in records])
    u_e = np.array([r[pred_key + "_energy_sigma"] for r in records])
    e_dft_rel = e_dft - e_dft.min()
    e_pred_rel = e_pred - e_pred.min()

    d_dft = np.array([r["dft_density"] for r in records])
    d_pred = np.array([r[pred_key + "_density"] for r in records])
    u_d = np.array([r[pred_key + "_density_sigma"] for r in records])

    energy_path = os.path.join(outdir, f"{prefix}_energy.svg")
    density_path = os.path.join(outdir, f"{prefix}_density.svg")

    mae_e, rmse_e, r2_e, slope_e = _save_parity_figure(
        e_dft_rel, e_pred_rel, u_e,
        "DFT relative energy (eV)",
        f"{model}-predicted relative energy (eV)",
        "eV", energy_path, 0.08, 3, scatter_color)
    mae_d, rmse_d, r2_d, slope_d = _save_parity_figure(
        d_dft, d_pred, u_d,
        r"DFT density (g/cm$^3$)",
        rf"{model}-predicted density (g/cm$^3$)",
        "g/cm³", density_path, 0.15, 4, scatter_color)

    print("═" * 60)
    print("Energy parity — relative energies (E − E_min)")
    print(f"  MAE: {mae_e:.4f} eV,  RMSE: {rmse_e:.4f} eV, "
          f"R²: {r2_e:.4f},  slope: {slope_e:.4f}")
    print("─" * 60)
    print("Density parity")
    print(f"  MAE: {mae_d:.4f} g/cm³, RMSE: {rmse_d:.4f} g/cm³, "
          f"R²: {r2_d:.4f},  slope: {slope_d:.4f}")
    print("═" * 60)
    print(f"Saved: {energy_path}, {density_path}")
    return energy_path, density_path


# Backwards-compatible name used by earlier revisions of gpt().
_save_parity_plots = save_parity_plots


def gpt(path=".", n=5, seed=0, plot=True):
    """Refit the GP without ``n`` sampled crystals and report their predictions.

    Returns one record per sampled crystal (see ``format_report`` for the
    printed columns).  ``seed`` keeps sampling reproducible; it is not exposed
    as a CLI option.  The GP backend / training options come from INPUT.txt.
    With ``plot=True`` also writes ``test_gp_energy.svg`` and
    ``test_gp_density.svg`` (DFT vs GP parity scatters with GP uncertainty).
    """
    data_dir, options = _read_input(path)
    prep = prepare_holdout(path, n, seed)
    n_rows = prep["n_rows"]
    idx, mask = prep["idx"], prep["mask"]
    X_train, X_query = prep["X_train"], prep["X_query"]
    y, y_eng = prep["y_density"], prep["y_energy"]

    # load_gaussian_process caches to RELATIVE gpr_*.pkl and overwrites
    # gpcsp.log.  Fit in an empty temporary cwd so a stale full-data model
    # cannot be loaded and the run directory is left untouched.
    previous = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="gpevo_gpt_") as tmp:
        try:
            os.chdir(tmp)
            # Both sklearn and GPyTorch keep the ambient BLAS thread count in
            # core._load_or_fit_gpr so fits can use all cores.  Do NOT wrap
            # this call in threadpool_limits(1) -- that was the original
            # source of the slow GPyTorch refit here.
            gpr_energy, gpr_density = load_gaussian_process(
                X_train, y[mask], y_eng[mask], **options)
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

    print(format_report(records, n_rows, n, data_dir, options))
    if plot:
        save_parity_plots(records, outdir=os.getcwd())
    return records


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="gpevo gpt",
        description="Hold out crystals and compare GP vs DFT density/energy "
                    "(GP backend from INPUT.txt).")
    parser.add_argument("-p", "--path", default=".",
                        help="run directory containing INPUT.txt")
    parser.add_argument("--n", type=int, default=5,
                        help="number of crystals to sample from structures.traj")
    args = parser.parse_args(argv)
    gpt(path=args.path, n=args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
