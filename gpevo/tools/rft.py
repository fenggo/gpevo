"""Hold out crystals from a training set and compare RandomForest vs DFT.

``gpevo rft`` is the RandomForest counterpart of ``gpevo gpt``: it reads the
``data`` option of INPUT.txt, samples ``--n`` crystals from
``<data>/structures.traj``, fits two ``RandomForestRegressor`` models
**without** those crystals, and reports DFT vs predicted density/energy for
each held-out crystal, plus the same parity scatters as ``gpt``.

It reuses ``gpevo.tools.gpt``'s database preparation, report layout and
plotting helpers verbatim, so GP and RF scores are directly comparable: same
rows, same 10-D features, same train/query split and same standardisation.
No calculator is invoked.

Unlike a Gaussian process, a random forest has no posterior variance.  The
``sd`` columns are the **standard deviation of the individual tree
predictions** (ensemble disagreement), which is the usual RF uncertainty
proxy.  They are not calibrated posterior sigmas and should not be compared
numerically with GP sigmas.

Usage::

    gpevo rft --n 5
    gpevo rft -p DIR --n 5
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from gpevo.tools.gpt import (format_report, prepare_holdout,
                             save_parity_plots)

# Match the RandomForest settings historically used by uspexkit's load_rfr().
_N_ESTIMATORS = 300
_RANDOM_STATE = 37


def _rf_predict(rf, X):
    """Mean and across-tree standard deviation of an RF ensemble."""
    per_tree = np.stack([tree.predict(X) for tree in rf.estimators_], axis=0)
    return per_tree.mean(axis=0), per_tree.std(axis=0)


def _fit_rf(X_train, y, *, n_estimators, random_state, n_jobs):
    """Construct and fit one RandomForestRegressor (factored out for tests)."""
    rf = RandomForestRegressor(n_estimators=n_estimators,
                               random_state=random_state, n_jobs=n_jobs)
    rf.fit(X_train, y)
    return rf


def rft(path=".", n=5, seed=0, plot=True, n_estimators=_N_ESTIMATORS,
        n_jobs=-1, prefix="test_rf"):
    """Refit two random forests without ``n`` sampled crystals.

    Returns one record per sampled crystal with the DFT truth, the RF
    prediction and the ensemble standard deviation.  ``seed`` keeps the
    sampling reproducible; it is not exposed as a CLI option.  With
    ``plot=True`` writes ``<prefix>_energy.svg`` and ``<prefix>_density.svg``
    into the current directory (defaults: ``test_rf_*.svg``).
    """
    prep = prepare_holdout(path, n, seed)
    n_rows = prep["n_rows"]
    idx, mask = prep["idx"], prep["mask"]
    X_train, X_query = prep["X_train"], prep["X_query"]
    y_density, y_energy = prep["y_density"], prep["y_energy"]

    kw = dict(n_estimators=n_estimators, random_state=_RANDOM_STATE,
              n_jobs=n_jobs)
    rf_density = _fit_rf(X_train, y_density[mask], **kw)
    rf_energy = _fit_rf(X_train, y_energy[mask], **kw)

    mu_d, sd_d = _rf_predict(rf_density, X_query)
    mu_e, sd_e = _rf_predict(rf_energy, X_query)

    records = []
    for row, dft_d, dft_e, p_d, s_d, p_e, s_e in zip(
            idx, y_density[idx], y_energy[idx], mu_d, sd_d, mu_e, sd_e):
        records.append({
            "row": int(row),
            "dft_density": float(dft_d),
            "dft_energy": float(dft_e),
            "rf_density": float(p_d),
            "rf_density_sigma": float(s_d),
            "rf_energy": float(p_e),
            "rf_energy_sigma": float(s_e),
        })

    print(format_report(
        records, n_rows, n, prep["data_dir"],
        model="RF", pred_key="rf",
        sd_note="(sd = RF ensemble std across trees; "
                "95% half-width ~= 1.96*sd)"))
    if plot:
        save_parity_plots(records, outdir=os.getcwd(), prefix=prefix,
                          model="RF", pred_key="rf")
    return records


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="gpevo rft",
        description="Hold out crystals and compare RandomForest vs DFT "
                    "density/energy (RF test).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--path", default=".",
                        help="run directory containing INPUT.txt")
    parser.add_argument("--n", type=int, default=5,
                        help="number of crystals to sample from structures.traj")
    parser.add_argument("--n-estimators", type=int, default=_N_ESTIMATORS,
                        help="number of trees in each forest")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="parallel jobs for the forest fit (-1 = all cores)")
    args = parser.parse_args(argv)
    rft(path=args.path, n=args.n, n_estimators=args.n_estimators,
        n_jobs=args.n_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
