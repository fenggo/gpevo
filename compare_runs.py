#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two USPEX runs (Python port `gpevo` vs the MATLAB/Octave reference).

Reads the `Individuals` file of every result folder and plots the density and
enthalpy distributions side by side:

  1/2  density / enthalpy histogram + KDE
  3/4  density / enthalpy box plot
  5    enthalpy vs density scatter
  6    best-so-far curves over the structure index

A folder that does not exist yet, or whose Individuals file has no usable row
(calculation still running), is skipped with a warning, so the script can be
re-run unchanged until every run has finished.

Usage:
    python compare_runs.py                       # tnt/results2 vs uspex-gp/results1
    python compare_runs.py --dirs A B C --labels gpevo octave ref
    python compare_runs.py --out compare.png
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

DEFAULT_DIRS = ["tnt/results3", "uspex-gp/results1"]
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def _tokens(line):
    """Split a line, keeping ``[ ... ]`` groups as one token."""
    out, buf, inside = [], [], False
    for tok in line.split():
        if inside:
            buf.append(tok)
            if tok.endswith("]"):
                inside = False
                out.append(" ".join(buf))
            continue
        if tok.startswith("["):
            inside = True
            buf = [tok]
            if tok.endswith("]") and len(tok) > 1:
                inside = False
                out.append(tok)
            continue
        out.append(tok)
    return out


def read_individuals(path):
    """Parse an USPEX Individuals file -> (gen, id, enthalpy, density)."""
    gen, ids, enth, dens = [], [], [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            if not line.strip():
                continue
            tok = _tokens(line)
            if len(tok) < 8 or not tok[0].isdigit():
                continue                      # header / blank / malformed
            comp = next((i for i, t in enumerate(tok) if t.startswith("[")), None)
            if comp is None or comp + 3 >= len(tok):
                continue
            try:
                g, n = int(tok[0]), int(tok[1])
                e = float(tok[comp + 1])
                d = float(tok[comp + 3])
            except (TypeError, ValueError):
                continue
            gen.append(g)
            ids.append(n)
            enth.append(e)
            dens.append(d)
    return (np.array(gen), np.array(ids),
            np.array(enth, dtype=float), np.array(dens, dtype=float))


def _robust_mask(values, mad_factor):
    """Keep the values within *mad_factor* robust sigmas of the median.

    ReaxFF relaxations occasionally collapse into a non-physical minimum tens of
    eV below the rest; those points would otherwise flatten every distribution.
    """
    if mad_factor <= 0:
        return np.ones(len(values), dtype=bool)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad < 1e-9:
        return np.ones(len(values), dtype=bool)
    return np.abs(values - med) <= mad_factor * 1.4826 * mad


def load_run(directory, mad_factor=6.0):
    """Valid structures of one result folder; None when unusable."""
    path = os.path.join(directory, "Individuals")
    if not os.path.isfile(path):
        print(f"  ! {directory}: no Individuals file, skipped")
        return None
    gen, ids, enth, dens = read_individuals(path)
    if len(enth) == 0:
        print(f"  ! {directory}: no parsed rows, skipped")
        return None
    # Drop failures: USPEX stores 100000/9999+ for structures that errored out.
    ok = np.isfinite(enth) & np.isfinite(dens) & (dens > 0) & (np.abs(enth) < 9999)
    dropped = int((~ok).sum())
    outliers = ok & ~_robust_mask(enth[ok], mad_factor)
    ok = ok & ~outliers
    label = os.path.basename(os.path.normpath(directory)) or directory
    print(f"  - {directory}: {len(enth)} rows, {int(ok.sum())} valid"
          f" ({dropped} failed, {int(outliers.sum())} enthalpy outliers"
          f" beyond {mad_factor:g} MAD)")
    if not ok.any():
        return None
    return {
        "dir": directory,
        "label": os.path.basename(os.path.dirname(os.path.normpath(directory))) + "/" + label,
        "gen": gen[ok], "id": ids[ok], "enth": enth[ok], "dens": dens[ok],
        "n_outliers": int(outliers.sum()),
    }


def _hist_kde(ax, series, color, label, bins, xlabel):
    lo = min(float(np.min(s)) for s in series)
    hi = max(float(np.max(s)) for s in series)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    for s, c, lb in zip(series, color, label):
        ax.hist(s, bins=edges, density=True, alpha=0.35, color=c,
                edgecolor=c, label=f"{lb} (n={len(s)})")
        if len(s) > 2 and np.std(s) > 1e-9:
            kde = gaussian_kde(s)
            x = np.linspace(lo, hi, 300)
            ax.plot(x, kde(x), color=c, lw=1.8)
        ax.axvline(float(np.mean(s)), color=c, ls="--", lw=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability density")
    ax.legend(fontsize=8)


def _cumulative_best(ax, ax2, runs, colors):
    for run, c in zip(runs, colors):
        order = np.argsort(run["id"])
        dens = run["dens"][order]
        enth = run["enth"][order]
        ax.plot(np.arange(1, len(dens) + 1), np.maximum.accumulate(dens),
                color=c, lw=1.6, label=f"{run['label']} max density")
        ax2.plot(np.arange(1, len(enth) + 1), np.minimum.accumulate(enth),
                 color=c, lw=1.6, ls="--", label=f"{run['label']} min enthalpy")
    ax.set_xlabel("structure index (ID order)")
    ax.set_ylabel("best density (g/cm$^3$)")
    ax2.set_ylabel("best enthalpy (eV)")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="lower right")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS,
                    help="result folders, each containing an Individuals file")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="legend labels (default: the folder path)")
    ap.add_argument("--out", default="compare_runs.png", help="output figure")
    ap.add_argument("--bins", type=int, default=30, help="histogram bins")
    ap.add_argument("--mad", type=float, default=6.0,
                    help="drop enthalpies further than this many robust sigmas "
                         "from the median (0 = keep everything)")
    args = ap.parse_args()

    print("Reading Individuals files:")
    runs = [r for r in (load_run(d, args.mad) for d in args.dirs) if r is not None]
    if not runs:
        sys.exit("No usable result folder — nothing to plot.")
    if args.labels:
        for run, lb in zip(runs, args.labels):
            run["label"] = lb
    colors = [COLORS[i % len(COLORS)] for i in range(len(runs))]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    _hist_kde(axes[0, 0], [r["dens"] for r in runs], colors,
              [r["label"] for r in runs], args.bins, "density (g/cm$^3$)")
    _hist_kde(axes[0, 1], [r["enth"] for r in runs], colors,
              [r["label"] for r in runs], args.bins, "enthalpy (eV)")

    labels = [r["label"] for r in runs]
    axes[0, 2].boxplot([r["dens"] for r in runs], tick_labels=labels)
    axes[0, 2].set_ylabel("density (g/cm$^3$)")
    axes[1, 0].boxplot([r["enth"] for r in runs], tick_labels=labels)
    axes[1, 0].set_ylabel("enthalpy (eV)")
    for ax in (axes[0, 2], axes[1, 0]):
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)

    for run, c in zip(runs, colors):
        axes[1, 1].scatter(run["dens"], run["enth"], s=14, alpha=0.6,
                           color=c, label=run["label"])
    axes[1, 1].set_xlabel("density (g/cm$^3$)")
    axes[1, 1].set_ylabel("enthalpy (eV)")
    axes[1, 1].legend(fontsize=8)

    _cumulative_best(axes[1, 2], axes[1, 2].twinx(), runs, colors)

    fig.suptitle("USPEX runs: density / enthalpy distributions")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")

    print("\n{:28s} {:>6s} {:>9s} {:>9s} {:>9s} {:>11s} {:>11s}".format(
        "run", "n", "rho_mean", "rho_std", "rho_max", "E_mean", "E_min"))
    for run in runs:
        print("{:28s} {:6d} {:9.4f} {:9.4f} {:9.4f} {:11.3f} {:11.3f}".format(
            run["label"], len(run["dens"]), float(np.mean(run["dens"])),
            float(np.std(run["dens"])), float(np.max(run["dens"])),
            float(np.mean(run["enth"])), float(np.min(run["enth"]))))
    print("\n(enthalpy outliers excluded from the statistics: "
          + ", ".join(f"{r['label']}={r['n_outliers']}" for r in runs) + ")")


if __name__ == "__main__":
    main()
