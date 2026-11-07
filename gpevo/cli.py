"""Console entry point for the packaged ``gpevo`` application.

``gpevo`` with no arguments (or ``gpevo run``) starts a search, exactly as it
always did; the other subcommands are housekeeping around a run directory.

Adding a subcommand: write ``gpevo/tools/<name>.py`` with a ``<name>(...)``
function and add one ``add_parser`` block below.
"""
from __future__ import annotations

import argparse


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="gpevo",
        description="GPEvo: Python port of the USPEX-GP 310 molecular-crystal "
                    "evolutionary search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="subcommands", dest="command")

    subparsers.add_parser("run", help="run the evolutionary search (default)")

    parser_clean = subparsers.add_parser(
        "clean", help="clean a run directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_clean.add_argument("-p", "--path", default=".",
                              help="run directory to clean")
    parser_clean.add_argument("-a", "--all", action="store_true",
                              help="also delete results*/")
    parser_clean.add_argument("-f", "--force", action="store_true",
                              help="clean even if the directory has no INPUT.txt "
                                   "or CalcFold")
    parser_clean.add_argument("-n", "--dry-run", action="store_true",
                              help="list what would be removed instead of "
                                   "removing it")
    parser_ei = subparsers.add_parser(
        "ei-report",
        help="per-generation density evolution and EI convergence of a run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_ei.add_argument("-p", "--path", default=".", help="run directory")
    parser_ei.add_argument("--out", default=None,
                           help="output prefix (default <run>/results<N>/ei_report)")
    parser_ei.add_argument("--results", default=None,
                           help="results folder to read, e.g. 2 or results2 "
                                "(default: the newest results<N>)")
    parser_ei.add_argument("--trials", type=int, default=200,
                           help="random-selection draws of the convergence baseline")
    parser_ei.add_argument("-n", "--dry-run", action="store_true",
                           help="print the per-generation report without plotting")

    parser_zmat = subparsers.add_parser(
        "zmat",
        help="convert a structure to USPEX Z-matrix format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_zmat.add_argument("--geo", default="POSCAR",
                             help="input geometry file")
    parser_zmat.add_argument("--i", type=int, default=-1,
                             help="frame index")

    parser_pred = subparsers.add_parser(
        "pred",
        help="predict density/energy with GP+MLP+RandomForest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_pred.add_argument("--t", default="Individuals.traj",
                             help="trajectory file")
    parser_pred.add_argument("--g", type=str, default=None,
                             help="geometry structure file")
    parser_pred.add_argument("--gen", type=int, default=None,
                             help="generation number")
    parser_pred.add_argument("--f", type=int, default=1,
                             help="feature flag (1=8D)")
    parser_pred.add_argument("--x", type=int, default=-1, help="index")
    parser_pred.add_argument("--den", type=float, default=1.88,
                             help="density threshold")
    parser_pred.add_argument("--ids", default=None,
                             help="crystal indices (space-separated)")
    parser_pred.add_argument("--step", type=int, default=300,
                             help="optimization steps")
    parser_pred.add_argument("--ncpu", type=int, default=8,
                             help="number of CPUs")
    parser_pred.add_argument("--data", "--dat", dest="data",
                             default="data", help="data directory name")
    parser_pred.add_argument("--tolerance", type=float, default=0.001,
                             help="structure matching tolerance")
    parser_pred.add_argument("--c", type=str, default="nn",
                             help="calculator to use: nn, mtp")

    parser_calc = subparsers.add_parser(
        "calc",
        help="run reference calculations for selected crystals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_calc.add_argument("--t", default="Individuals.traj",
                             help="trajectory file")
    parser_calc.add_argument("--den", type=float, default=1.88,
                             help="density threshold")
    parser_calc.add_argument("--ids", default=None,
                             help="crystal indices (space-separated)")
    parser_calc.add_argument("--step", type=int, default=300,
                             help="MD steps")
    parser_calc.add_argument("--ncpu", type=int, default=8,
                             help="number of CPUs")
    parser_calc.add_argument("--data", "--dat", dest="data",
                             default="data", help="data directory name")
    parser_calc.add_argument("--tolerance", type=float, default=0.01,
                             help="structure matching tolerance")
    parser_calc.add_argument("--gen", type=int, default=None,
                             help="generation number")

    parser_cdata = subparsers.add_parser(
        "calcdata",
        help="generate training data with MLP relaxation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_cdata.add_argument("--t", default="structures.traj",
                              help="trajectory file")
    parser_cdata.add_argument("--n", type=int, default=1,
                              help="number of CPUs")
    parser_cdata.add_argument("--step", type=int, default=1000,
                              help="MLP optimization steps")
    parser_cdata.add_argument("--c", type=str, default="nn",
                              help="calculator: nn, mtp")
    parser_cdata.add_argument("--f", type=int, default=1,
                              help="feature mode (1=raw; 2=energy averaged per molecule)")

    parser_add = subparsers.add_parser(
        "add",
        help="add a structure to the training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_add.add_argument("--traj", default="structures.traj",
                            help="trajectory file to read the structure from")
    parser_add.add_argument("--i", type=int, default=-1,
                            help="frame index in the trajectory")
    parser_add.add_argument("--step", type=int, default=1000,
                            help="MLP optimization steps")
    parser_add.add_argument("--tolerance", type=float, default=0.005,
                            help="structure matching tolerance")
    parser_add.add_argument("--ncpu", type=int, default=1,
                            help="number of CPUs")
    parser_add.add_argument("--f", type=int, default=1,
                            help="feature mode (1=raw; 2=energy averaged per molecule)")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.command in (None, "run"):
        from gpevo.FunctionFolder.USPEX.Start import Start
        Start()
        return
    if args.command == "clean":
        from gpevo.tools.clean import clean
        clean(path=args.path, all_=args.all, force=args.force,
              dry_run=args.dry_run)
        return
    if args.command == "ei-report":
        from gpevo.tools.ei_report import ei_report
        if args.dry_run:
            print(ei_report(path=args.path, trials=args.trials, quiet=True,
                            results=args.results))
        else:
            ei_report(path=args.path, out=args.out, trials=args.trials,
                      results=args.results)
        return
    if args.command == "zmat":
        from gpevo.uspexkit.core import zmat
        zmat(geo=args.geo, i=args.i)
        return
    if args.command == "pred":
        from gpevo.uspexkit.core import pred
        pred(t=args.t, g=args.g, gen=args.gen, f=args.f, den=args.den,
             ids=args.ids, x=args.x, c=args.c, step=args.step,
             ncpu=args.ncpu, dat=args.data, tolerance=args.tolerance)
        return
    if args.command == "calc":
        from gpevo.uspexkit.core import calc
        calc(t=args.t, den=args.den, ids=args.ids, step=args.step,
             ncpu=args.ncpu, dat=args.data, tolerance=args.tolerance,
             gen=args.gen)
        return
    if args.command == "calcdata":
        from gpevo.uspexkit.core import calcdata
        calcdata(traj=args.t, step=args.step, n=args.n, c=args.c, f=args.f)
        return
    if args.command == "add":
        from gpevo.uspexkit.core import add
        add(traj=args.traj, i=args.i, step=args.step,
            tolerance=args.tolerance, ncpu=args.ncpu, f=args.f)
        return
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
