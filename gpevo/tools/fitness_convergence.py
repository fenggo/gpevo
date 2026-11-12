"""Plot per-generation fitness convergence from a USPEX ``Individuals`` table.

``BESTIndividuals`` reports ``ranking[0]`` and can show the 100000 duplicate
penalty, so it is not a reliable convergence curve.  This tool reads the full
``Individuals`` table, discards failed / penalised rows, and plots min / p10 /
median / max per generation, plus the fraction of rows hit by the duplicate
penalty (``degradeSimilar``), which distorts selection near convergence.
"""
from __future__ import annotations

import re
import statistics

# Gen ID Origin [composition] Enthalpy Volume Density Fitness ...
_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\S+)\s+\[[^\]]*\]\s+"
    r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"
)


def parse_individuals(path):
    """Return one dict per data row of an ``Individuals`` table."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = _ROW.match(line)
            if not match:
                continue
            try:
                rows.append({
                    "gen": int(match.group(1)),
                    "id": int(match.group(2)),
                    "origin": match.group(3),
                    "enthalpy": float(match.group(4)),
                    "volume": float(match.group(5)),
                    "density": float(match.group(6)),
                    "fitness": float(match.group(7)),
                })
            except ValueError:
                continue
    return rows


def per_generation(rows, penalty=99999.0):
    """Per-generation stats over rows that are neither failed nor penalised."""
    out = []
    for gen in sorted({row["gen"] for row in rows}):
        gen_rows = [r for r in rows if r["gen"] == gen]
        good = sorted(r["fitness"] for r in gen_rows
                      if r["fitness"] < penalty and r["fitness"] == r["fitness"])
        if not good:
            continue
        out.append({
            "gen": gen,
            "n": len(gen_rows),
            "n_valid": len(good),
            "n_penalized": len(gen_rows) - len(good),
            "min": good[0],
            "p10": good[max(0, len(good) // 10)],
            "median": float(statistics.median(good)),
            "max": good[-1],
            "median_volume": float(statistics.median(r["volume"] for r in gen_rows)),
            "median_density": float(statistics.median(r["density"] for r in gen_rows)),
        })
    return out


def format_report(stats, tolerance=0.03):
    lines = [
        f"{'gen':>3} {'n':>4} {'min':>9} {'p10':>9} {'median':>9} {'max':>9} "
        f"{'penalised':>10} {'med V':>8} {'med rho':>8}"
    ]
    for s in stats:
        lines.append(
            f"{s['gen']:>3} {s['n']:>4} {s['min']:>9.4f} {s['p10']:>9.4f} "
            f"{s['median']:>9.4f} {s['max']:>9.4f} "
            f"{s['n_penalized']:>4}/{s['n']:<5} "
            f"{s['median_volume']:>8.0f} {s['median_density']:>8.3f}"
        )
    if stats:
        best = min(stats, key=lambda s: s["min"])
        lines.append(
            f"best: gen {best['gen']} min={best['min']:.4f}; "
            f"targetTolerance={tolerance:g}; "
            f"{'HIT' if best['min'] <= tolerance else 'not reached'}"
        )
    return "\n".join(lines)


def fitness_convergence(path="Individuals", out=None, tolerance=0.03,
                        show=False, csv=None):
    """Parse, report and plot.  Returns the per-generation stats."""
    import os

    rows = parse_individuals(path)
    if not rows:
        raise RuntimeError(f"no data rows parsed from {path}")
    stats = per_generation(rows)
    if not stats:
        raise RuntimeError(f"no valid (non-penalised) fitness values in {path}")

    print(format_report(stats, tolerance))

    if csv:
        import csv as _csv
        with open(csv, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=list(stats[0].keys()))
            writer.writeheader()
            writer.writerows(stats)
        print(f"wrote {csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens = [s["gen"] for s in stats]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for key, style, label in (
        ("min", "o-", "min"), ("p10", "s--", "p10"),
        ("median", "^-", "median"), ("max", "v:", "max"),
    ):
        ax1.plot(gens, [s[key] for s in stats], style, label=label, markersize=4)
    ax1.axhline(tolerance, color="red", linestyle="-.", linewidth=1.2,
                label=f"targetTolerance={tolerance:g}")
    ax1.set_ylabel("fitness (fingerprint cosine distance)")
    ax1.set_title(f"optType 14 convergence — {os.path.basename(path)}")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, ncol=3)

    frac = [s["n_penalized"] / s["n"] for s in stats]
    ax2.bar(gens, frac, color="tab:orange", alpha=0.7)
    ax2.set_ylabel("penalised\nfraction")
    ax2.set_xlabel("generation")
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_ylim(0, max(0.05, max(frac) * 1.2))

    fig.tight_layout()
    if out is None:
        out = os.path.join(os.path.dirname(os.path.abspath(path)),
                           "fitness_convergence.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    if show:
        plt.show()
    plt.close(fig)
    return stats


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot per-generation min/median fitness from an Individuals table.")
    parser.add_argument("-p", "--path", default="Individuals",
                        help="path to the Individuals table")
    parser.add_argument("-o", "--out", default=None,
                        help="output PNG (default: <Individuals dir>/fitness_convergence.png)")
    parser.add_argument("--tolerance", type=float, default=0.03,
                        help="targetTolerance hit threshold")
    parser.add_argument("--csv", default=None, help="also write per-generation stats as CSV")
    parser.add_argument("--show", action="store_true", help="show the figure interactively")
    args = parser.parse_args(argv)
    fitness_convergence(path=args.path, out=args.out, tolerance=args.tolerance,
                        show=args.show, csv=args.csv)


if __name__ == "__main__":
    main()
