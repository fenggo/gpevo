#!/usr/bin/env python3
"""Does the structural fingerprint help the GP?  A controlled ablation.

Question
--------
gpevo's GP predicts DFT density / energy from a 10-dimensional feature vector
built by ``uspexkit`` (cheap-method energy decomposition + a density proxy).
Every relaxed individual already carries an RDF fingerprint
(``gpevo.FunctionFolder.sys.fingerprint_utils.compute_full_atom_fingerprint``),
so it is tempting to concatenate it onto those 10 dimensions.  This script
measures whether that actually helps.

Design
------
* Inputs: ``feature_mlp.csv`` (X, the cheap-method features) and
  ``feature.csv`` (y, the DFT truth), plus ``structures.traj`` for the
  fingerprints.  A delta-learning setup: cheap method in, DFT out.
* Model: **exactly** the production kernel of ``uspexkit.core``
  (``load_gaussian_process``, core.py:582-601) — ARD length scales,
  ``n_restarts_optimizer``.  Using anything else invalidates the comparison
  (see the note in feature_ablation.md).
* PCA is fitted **inside each CV fold** (train only) so nothing leaks.
* Control: the fingerprint rows are randomly permuted.  If the "improvement"
  survives shuffling it is an artifact of adding columns, not signal.

Usage
-----
    python docs/feature_ablation.py --data ~/tnt22/data22
    python docs/feature_ablation.py --data ~/tnt22/data22 --restarts 0   # fast
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import (  # noqa: E402
    DotProduct,
    Matern,
    RBF,
    WhiteKernel,
)
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def production_kernel(n_features: int, kind: str):
    """The kernel uspexkit actually fits (core.py:582-601).

    ``length_scale`` is a *list*, i.e. one length scale per feature (ARD).
    This matters: with a single isotropic length scale the optimiser collapses
    to the lower bound on 10-D data and the model predicts the mean.
    """
    length_scale = [0.1] * n_features
    core = (
        Matern(length_scale=length_scale, nu=2.5)
        if kind == "density"
        else RBF(length_scale=length_scale)
    )
    return (
        0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50))
        + 0.35**2 * core
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1e-1))
    )


def cross_val(X, y, kind, folds=5, restarts=10, seed=0):
    y = np.asarray(y, dtype=float)
    pred = np.zeros_like(y)
    for train, test in KFold(folds, shuffle=True, random_state=seed).split(y):
        scaler = StandardScaler().fit(X[train])
        model = GaussianProcessRegressor(
            kernel=production_kernel(X.shape[1], kind),
            n_restarts_optimizer=restarts,
            random_state=seed,
        )
        model.fit(scaler.transform(X[train]), y[train])
        pred[test] = model.predict(scaler.transform(X[test]))
    return mean_absolute_error(y, pred), r2_score(y, pred)


def cross_val_with_fingerprints(X, F, y, kind, k, folds=5, restarts=10, seed=0):
    y = np.asarray(y, dtype=float)
    pred = np.zeros_like(y)
    for train, test in KFold(folds, shuffle=True, random_state=seed).split(y):
        pca = PCA(n_components=k).fit(F[train])          # train fold only
        Xtr = np.hstack([X[train], pca.transform(F[train])])
        Xte = np.hstack([X[test], pca.transform(F[test])])
        scaler = StandardScaler().fit(Xtr)
        model = GaussianProcessRegressor(
            kernel=production_kernel(Xtr.shape[1], kind),
            n_restarts_optimizer=restarts,
            random_state=seed,
        )
        model.fit(scaler.transform(Xtr), y[train])
        pred[test] = model.predict(scaler.transform(Xte))
    return mean_absolute_error(y, pred), r2_score(y, pred)


def compute_fingerprints(traj_path, rmax, sigma, delta, cache=None):
    """RDF fingerprint of every structure, flattened into one row each."""
    if cache and os.path.isfile(cache):
        return np.load(cache)

    from ase.io import read
    from gpevo.FunctionFolder.sys.fingerprint_utils import (
        compute_full_atom_fingerprint,
    )

    images = read(traj_path, index=":")
    species = sorted({int(z) for img in images for z in img.get_atomic_numbers()})
    rows = []
    for img in images:
        z = [int(v) for v in img.get_atomic_numbers()]
        num_ions = [z.count(s) for s in species]
        # group atoms by species so the fingerprint is comparable across rows
        img = img[[i for s in species for i in range(len(z)) if z[i] == s]]
        (_, fing, _), _ = compute_full_atom_fingerprint(
            np.asarray(img.cell.array, dtype=float),
            np.asarray(img.get_positions(), dtype=float),
            num_ions,
            rmax=rmax,
            sigma=sigma,
            delta=delta,
            dimension=3,
        )
        rows.append(np.asarray(fing, dtype=float).ravel())
    out = np.vstack(rows)
    if cache:
        np.save(cache, out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="directory with feature*.csv + structures.traj")
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5, 8], help="PCA components")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--restarts", type=int, default=10,
                    help="n_restarts_optimizer; 10 == production, 0 == fast")
    ap.add_argument("--shuffles", type=int, default=3, help="permuted-fingerprint controls")
    ap.add_argument("--rmax", type=float, default=10.0)
    ap.add_argument("--delta", type=float, default=0.04)
    ap.add_argument("--sigma", type=float, default=0.015)
    ap.add_argument("--cache", default=None, help=".npy file to cache fingerprints in")
    args = ap.parse_args(argv)

    d = args.data
    mlp = np.loadtxt(os.path.join(d, "feature_mlp.csv"), delimiter=",", skiprows=1)
    dft = np.loadtxt(os.path.join(d, "feature.csv"), delimiter=",", skiprows=1)
    if mlp.shape[0] != dft.shape[0]:
        sys.exit("feature_mlp.csv and feature.csv disagree on the number of rows")

    X = mlp[:, 1:]                       # 10 cheap-method features
    targets = {"density": dft[:, -1], "energy": dft[:, 1]}

    print(f"{len(X)} samples, {X.shape[1]} base features")
    t0 = time.time()
    F = compute_fingerprints(os.path.join(d, "structures.traj"), args.rmax,
                             args.sigma, args.delta, args.cache)
    print(f"fingerprint matrix {F.shape}  ({time.time() - t0:.1f}s)\n")

    for kind, y in targets.items():
        print(f"=== {kind}  (std={y.std():.4f}) ===")
        mae, r2 = cross_val(X, y, kind, args.folds, args.restarts)
        print(f"  baseline 10-d          MAE={mae:8.4f}  R2={r2:7.4f}")
        for k in args.k:
            mae, r2 = cross_val_with_fingerprints(X, F, y, kind, k,
                                                  args.folds, args.restarts)
            print(f"  + fingerprint PCA-{k:<3}   MAE={mae:8.4f}  R2={r2:7.4f}")
        if args.shuffles:
            ctrl = [
                cross_val_with_fingerprints(
                    X, F[np.random.RandomState(s).permutation(len(F))],
                    y, kind, max(args.k), args.folds, args.restarts,
                )
                for s in range(args.shuffles)
            ]
            print(f"  + SHUFFLED PCA-{max(args.k):<3} (ctrl) "
                  f"MAE={np.mean([a for a, _ in ctrl]):8.4f}  "
                  f"R2={np.mean([b for _, b in ctrl]):7.4f}")
        print()
    print("If the shuffled control scores as well as the real fingerprint, "
          "the gain is an artifact, not signal.")


if __name__ == "__main__":
    main()
