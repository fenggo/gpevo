"""Scheduler-side CLI for the vendored uspexkit GP evaluation step.

Only the ``gp`` subcommand is vendored here.  The in-process GP path already
calls ``gpevo.uspexkit.core`` directly (``gp_ml.run_gp``); this entry point
replaces the old external ``uspexkit gp ...`` command that scheduler scripts
used to emit, so gpevo no longer depends on an installed ``uspexkit`` package
on the compute node::

    python -m gpevo.uspexkit.cli gp --n=24 --data=data22 --den=1.8 \
        --u=0.03 --k=1 --mode=ei --optype=11 \
        --backend=sklearn --iter=200 --lr=0.1

The argument set mirrors the ``gp`` subcommand of the upstream uspexkit CLI,
with ``--backend/--iter/--lr`` added for the opt-in GPyTorch training backend.
"""
from __future__ import annotations

import argparse


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="gpevo-uspexkit",
        description="Vendored uspexkit GP evaluation step for scheduler jobs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gp = sub.add_parser(
        "gp", help="Gaussian process to predict the crystal density")
    p_gp.add_argument("--n", type=int, default=1,
                      help="number of CPUs to use")
    p_gp.add_argument("--t", type=float, default=0.005,
                      help="structure match tolerance")
    p_gp.add_argument("--step", type=int, default=1000,
                      help="MLP optimisation steps")
    p_gp.add_argument("--b", type=float, default=1.5,
                      help="energy deviation tolerance for a broken structure")
    p_gp.add_argument("--u", type=float, default=0.03,
                      help="GP uncertainty threshold")
    p_gp.add_argument("--f", type=int, default=1,
                      help="feature factor to use")
    p_gp.add_argument("--dft", type=int, default=0,
                      help="whether to run active learning / DFT")
    p_gp.add_argument("--den", type=float, default=1.82,
                      help="density criterion for active learning / DFT")
    p_gp.add_argument("--pop", type=int, default=100,
                      help="population size")
    p_gp.add_argument("--data", "--dat", dest="data", default="data",
                      help="training data directory (alias: --dat)")
    p_gp.add_argument("--ref", default="results1",
                      help="results file directory")
    p_gp.add_argument("--k", type=int, default=1,
                      help="top-K crystals for EI active learning")
    p_gp.add_argument("--mode", default="ei",
                      help="selection mode: ei | random | uncertainty")
    p_gp.add_argument("--optype", type=int, default=11,
                      help="optType 1: energy, 11: density")
    p_gp.add_argument("--id", type=int, default=None,
                      help="crystal global ID (bodyCount+1) from submitJob")
    p_gp.add_argument("--backend", default="sklearn",
                      choices=("sklearn", "gpytorch"),
                      help="GP training backend (default: sklearn)")
    p_gp.add_argument("--iter", dest="training_iter", type=int, default=200,
                      help="max L-BFGS iterations per restart (gpytorch)")
    p_gp.add_argument("--lr", dest="learning_rate", type=float, default=0.1,
                      help="L-BFGS learning rate (gpytorch)")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    from gpevo.uspexkit.core import gp

    gp(tolerance=args.t, step=args.step, n=args.n, b=args.b, u=args.u,
       f=args.f, dft=args.dft, den=args.den, pop=args.pop,
       dat=args.data, ref=args.ref, id_=args.id, k=args.k,
       mode=args.mode, optype=args.optype,
       backend=args.backend, training_iter=args.training_iter,
       learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
