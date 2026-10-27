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
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
