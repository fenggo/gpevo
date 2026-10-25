"""Space-group coverage of a USPEX/GPEvo run.

Reads the ``Individuals`` file of one or more ``results*`` directories and
reports how thoroughly the search explored symmetry: how many distinct space
groups were reached, which ones dominate, and whether late generations are
still finding new ones (a sign that the exploration is not exhausted).

Usage
-----
    python sg_coverage.py results1
    python sg_coverage.py results1 results2 --top 15
    python sg_coverage.py --gen          # per-generation breakdown

Columns of ``Individuals`` (brackets removed):
    Gen ID Origin Enthalpy Volume Density Fitness KPOINTS SYMM Q_entr A_order S_order
"""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict

BRACKETS = re.compile(r"\[[^\]]*\]")


def read_individuals(path):
    """-> list of (gen, id, symm) ; invalid rows are skipped.

    ``Composition`` and ``KPOINTS`` are bracketed and hold a variable number
    of tokens, so the trailing columns are addressed from the end instead:
    the file ends with ``... SYMM Q_entr A_order S_order``.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 6:
                continue
            try:
                gen = int(float(fields[0]))
                cid = int(float(fields[1]))
                symm = int(float(fields[-4]))
            except (TypeError, ValueError):
                continue
            rows.append((gen, cid, symm))
    return rows


def report(path, rows, top, per_gen):
    label = os.path.basename(os.path.normpath(path)) or path
    print(f"\n=== {label}: {len(rows)} structures ===")
    if not rows:
        print("  (no parseable row)")
        return

    counts = Counter(symm for _, _, symm in rows)
    distinct = len(counts)
    common_sg, common_n = counts.most_common(1)[0]

    print(f"  distinct space groups : {distinct}")
    print(f"  most populated        : SG {common_sg} "
          f"({common_n} structures, {100.0 * common_n / len(rows):.0f}%)")
    if 100.0 * common_n / len(rows) >= 50:
        print("  !! one space group holds >= 50% of the population -- the "
              "sampling is heavily concentrated")
    singletons = sum(1 for v in counts.values() if v == 1)
    print(f"  seen only once        : {singletons} "
          f"({100.0 * singletons / distinct:.0f}% of the groups)")
    low = counts.get(1, 0) + counts.get(2, 0)
    if distinct <= 3 or 100.0 * low / len(rows) >= 80:
        print("  note: with `doSpaceGroup : 1` in INPUT.txt the SYMM column is "
              "the symmetry\n        detected *after* relaxation, and molecular "
              "crystals usually relax\n        to P1 -- check the generated "
              "groups before calling this under-exploration.")

    print(f"\n  top {top} space groups:")
    width = max(len(f"{sg}") for sg, _ in counts.most_common(top))
    for sg, n in counts.most_common(top):
        bar = "#" * max(1, round(30 * n / common_n))
        print(f"    {sg:>{width}d}  {n:4d}  {bar}")

    if per_gen:
        by_gen = defaultdict(list)
        for gen, _, symm in rows:
            by_gen[gen].append(symm)
        seen = set()
        print("\n  per generation (cumulative distinct):")
        print("    gen  structures  new SG  cumulative SG")
        for gen in sorted(by_gen):
            before = len(seen)
            seen.update(by_gen[gen])
            print(f"    {gen:3d}  {len(by_gen[gen]):10d}  {len(seen) - before:6d} "
                  f"  {len(seen):13d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*",
                    help="results* directories (or paths to Individuals files)")
    ap.add_argument("--top", type=int, default=10,
                    help="how many space groups to list (default 10)")
    ap.add_argument("--gen", action="store_true",
                    help="add the per-generation breakdown")
    ap.add_argument("--all", action="store_true",
                    help="also merge all directories into one report")
    args = ap.parse_args()

    if not args.dirs:
        ap.error("give at least one results* directory")

    all_rows = []
    for entry in args.dirs:
        path = entry if os.path.basename(entry) == "Individuals" else \
            os.path.join(entry, "Individuals")
        if not os.path.isfile(path):
            print(f"  ! {path}: not found, skipped")
            continue
        rows = read_individuals(path)
        all_rows.extend(rows)
        report(path, rows, args.top, args.gen)

    if args.all and len(args.dirs) > 1:
        print("\n=== merged ===")
        report("merged", all_rows, args.top, args.gen)


if __name__ == "__main__":
    main()
