"""``gpevo ei-report``: per-generation density evolution and EI convergence.

The run directory already holds everything this needs, and none of it is
recomputed here:

* ``results1/Individuals``      -- every structure: generation, id, ML density
* ``results1/density_predict.log`` -- GP predictions (μ, σ) per crystal
* ``results1/density.log``      -- the DFT reference of the labelled crystals
* ``ei.log``                    -- the ids EI sent to DFT, one line per generation
* ``log``                       -- the ``GP learning curve`` lines of each generation

Three panels:

1. **Evolution** -- every structure as a dot, the per-generation maximum and
   the best-so-far step curve, with the EI-selected crystals circled and the
   DFT-verified ones as diamonds.
2. **Convergence** -- the EI selection trend: every EI-labelled crystal as a
   dot at its generation (x) against its density (y), with the GP uncertainty
   (±1.96σ on the density) drawn as an error bar, the best density found so far
   as a step line and the DFT reference of each crystal as a red diamond.
3. **Learning** -- per generation the GP uncertainty (σ) of the population and
   the labels/LOO RMSE of the training set, when the run logged them.

Parsing follows ``ei_select.py``; the random baseline is the new part.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import re

UNCERT_SCALE = 1.96          # gp.csv / density_predict.log store 1.96 * std
RESIDUAL_CUT = 10.0          # CalcFitness_310 keeps residual <= 10

_LEARNING_CURVE = re.compile(
    r"GP learning curve \(gen\s+(\d+)\):\s*rows\s+(\d+)\s*->\s*(\d+)"
    # "(+N)" and the RMSE tail are both written after the row counts
    r"(?:.*?LOO RMSE density=([-\d.eE+]+)\s*g/cm\^3\s+energy=([-\d.eE+]+))?")


# ──────────────────────────────────────────────────────────────────────
#  parsing
# ──────────────────────────────────────────────────────────────────────

def _strip_brackets(line):
    return re.sub(r"\[[^\]]*\]", "", line)


def parse_individuals(path):
    """``(rows, gen_ids)``; rows are ``(gen, id, enthalpy, volume, density, fitness)``.

    Only structures with a negative fitness are kept, which is the same filter
    ``pred`` applies.
    """
    rows, gen_ids = [], {}
    if not path:
        return rows, gen_ids
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = _strip_brackets(line).split()
                if len(parts) < 7 or parts[0] in ("Gen", "ID"):
                    continue
                try:
                    gen, cid = int(parts[0]), int(parts[1])
                    enthalpy, volume = float(parts[3]), float(parts[4])
                    density, fitness = float(parts[5]), float(parts[6])
                except (ValueError, IndexError):
                    continue
                if fitness >= 0.0:
                    continue
                rows.append((gen, cid, enthalpy, volume, density, fitness))
                gen_ids.setdefault(gen, []).append(cid)
    except OSError:
        return [], {}
    return rows, gen_ids


def parse_density_predict(path):
    """``{id: {residual, density_gp, std, density_mlp, density_rf, energy}}``."""
    data = {}
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if not parts or parts[0].startswith("#") or len(parts) < 6:
                    continue
                try:
                    cid = int(float(parts[0]))
                    data[cid] = {
                        "residual": float(parts[1]),
                        "density_mlp": float(parts[2]),
                        "density_rf": float(parts[3]),
                        "density_gp": float(parts[4]),
                        "std": float(parts[5]),
                        "energy": float(parts[6]) if len(parts) >= 7 else float("nan"),
                    }
                except ValueError:
                    continue
    except OSError:
        return {}
    return data


def parse_density_log(path):
    """``{id: (density, energy)}`` of the DFT reference."""
    dft = {}
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3 and not parts[0].startswith("#"):
                    try:
                        dft[int(float(parts[0]))] = (float(parts[1]), float(parts[2]))
                    except ValueError:
                        continue
    except OSError:
        return {}
    return dft


def parse_ei_log(path):
    """``{generation: [ids]}``; a generation written twice keeps the last one.

    ``ei.log`` is appended to, so a restarted run repeats its early
    generations -- only the latest attempt describes what was labelled.
    """
    selected = {}
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = re.match(r"\s*gen\s+(\d+)\s*:\s*(.*)", line)
                if not match:
                    continue
                ids = []
                for token in match.group(2).split():
                    try:
                        ids.append(int(token))
                    except ValueError:
                        continue
                selected[int(match.group(1))] = ids
    except OSError:
        return {}
    return selected


def parse_learning_curve(path):
    """``{generation: {rows, added, rmse_density, rmse_energy}}`` from the log."""
    out = {}
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = _LEARNING_CURVE.search(line)
                if not match:
                    continue
                gen, before, after = int(match.group(1)), int(match.group(2)), int(match.group(3))
                out[gen] = {
                    "rows": after,
                    "added": after - before,
                    "rmse_density": float(match.group(4)) if match.group(4) else float("nan"),
                    "rmse_energy": float(match.group(5)) if match.group(5) else float("nan"),
                }
    except OSError:
        return {}
    return out


# ──────────────────────────────────────────────────────────────────────
#  per-generation summary
# ──────────────────────────────────────────────────────────────────────

def build_summary(rows, gen_ids, pred, dft, ei_log):
    """``{gen: {...}}`` with the densities, the GP spread and the EI picks."""
    by_id = {row[1]: row for row in rows}
    summary = {}
    for gen, ids in gen_ids.items():
        densities = [by_id[cid][4] for cid in ids if cid in by_id]
        gp = [pred[cid] for cid in ids
              if cid in pred and pred[cid]["residual"] <= RESIDUAL_CUT]
        sigmas = [UNCERT_SCALE * entry["std"] for entry in gp]
        picks = []
        for cid in ei_log.get(gen, []):
            entry = pred.get(cid, {})
            picks.append({
                "id": cid,
                "density": by_id[cid][4] if cid in by_id else float("nan"),
                "density_gp": entry.get("density_gp", float("nan")),
                "sigma": UNCERT_SCALE * entry.get("std", float("nan")),
                "dft": dft.get(cid, (float("nan"), float("nan")))[0],
            })
        summary[gen] = {
            "ids": ids,
            "n": len(densities),
            "max": max(densities) if densities else float("nan"),
            "mean": sum(densities) / len(densities) if densities else float("nan"),
            "gp_mean": sum(e["density_gp"] for e in gp) / len(gp) if gp else float("nan"),
            "sigma_mean": sum(sigmas) / len(sigmas) if sigmas else float("nan"),
            "n_gp": len(gp),
            "picks": picks,
        }
    return summary


def random_baseline(gen_ids, by_id, ei_log, trials=200, seed=0):
    """Best-labelled density of a random selection with the same budget.

    Returns ``(steps, mean, low, high)``: at every cumulative label count, the
    mean best density of *trials* random draws and its 10th/90th percentile.
    EI only pays for itself if its own curve beats this band.
    """
    rng = random.Random(seed)
    gens = sorted(gen_ids)
    runs = []
    for _ in range(max(1, trials)):
        best, curve = float("-inf"), []
        for gen in gens:
            budget = len(ei_log.get(gen, []))
            pool = [cid for cid in gen_ids[gen] if cid in by_id]
            if budget and pool:
                for cid in rng.sample(pool, min(budget, len(pool))):
                    best = max(best, by_id[cid][4])
            if budget:
                curve.append(best)
        runs.append(curve)
    steps = max((len(curve) for curve in runs), default=0)
    if not steps:
        return [], [], [], []
    mean, low, high = [], [], []
    for i in range(steps):
        values = sorted(curve[i] for curve in runs if i < len(curve)
                        and math.isfinite(curve[i]))
        if not values:
            mean.append(float("nan"))
            low.append(float("nan"))
            high.append(float("nan"))
            continue
        mean.append(sum(values) / len(values))
        low.append(values[max(0, int(0.1 * (len(values) - 1)))])
        high.append(values[min(len(values) - 1, int(0.9 * (len(values) - 1)))])
    return list(range(1, steps + 1)), mean, low, high


def ei_curve(summary, gen_ids, ei_log):
    """Best density among the crystals EI labelled, per cumulative label."""
    best, curve = float("-inf"), []
    for gen in sorted(gen_ids):
        for pick in summary.get(gen, {}).get("picks", []):
            if math.isfinite(pick["density"]):
                best = max(best, pick["density"])
                curve.append(best)
    return curve


# ──────────────────────────────────────────────────────────────────────
#  plots
# ──────────────────────────────────────────────────────────────────────

def _jitter(count, span=0.32):
    if count <= 1:
        return [0.0]
    return [span * (2.0 * i / (count - 1) - 1.0) for i in range(count)]


def _evolution_figure(summary, rows, dft):
    """The evolution figure; kept separate so tests can inspect its legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens = sorted(summary)
    fig, ax = plt.subplots(figsize=(11, 6))

    for gen in gens:
        ids = summary[gen]["ids"]
        offsets = _jitter(len(ids))
        for cid, offset in zip(ids, offsets):
            row = next((r for r in rows if r[1] == cid), None)
            if row is None:
                continue
            ax.scatter(gen + offset, row[4], s=9, color="#bdc3c7",
                       edgecolors="none", zorder=1)
    ax.scatter([], [], s=9, color="#bdc3c7", label="structures (ML density)")

    maxima = [summary[gen]["max"] for gen in gens]
    ax.plot(gens, maxima, "o-", color="#2980b9", lw=1.8, ms=5, zorder=3,
            label="best of the generation")

    best, running = float("-inf"), []
    for value in maxima:
        best = max(best, value)
        running.append(best)
    ax.step(gens, running, where="post", color="#1a5276", lw=2.2, zorder=4,
            label="best so far")

    # The label goes on the first marker actually drawn, not on the first
    # generation: EI starts labelling at generation 2, so keying it to gens[0]
    # left both of these out of the legend entirely.
    ei_label, dft_label = "EI-selected (sent to the reference calculation)", None
    dft_label = "DFT density of that crystal (connected)"
    for gen in gens:
        picks = summary[gen]["picks"]
        for pick, offset in zip(picks, _jitter(len(picks), span=0.16)):
            if not math.isfinite(pick["density"]):
                continue
            x = gen + offset
            ax.scatter(x, pick["density"], s=240, facecolors="none",
                       edgecolors="#27ae60", linewidths=1.8, zorder=5,
                       label=ei_label)
            ei_label = None
            if math.isfinite(pick["dft"]):
                # the same crystal: what the search had, what DFT measured
                ax.plot([x, x], [pick["density"], pick["dft"]], color="#7f8c8d",
                        lw=0.9, linestyle="--", alpha=0.75, zorder=4)
                ax.scatter(x, pick["dft"], s=90, marker="D", color="#c0392b",
                           edgecolors="#641e16", linewidths=0.8, zorder=6,
                           label=dft_label)
                dft_label = None

    ax.set_xlabel("Generation")
    ax.set_ylabel(r"Density (g/cm$^3$)")
    ax.set_title("Density evolution with the GP/EI selection", fontsize=12,
                 fontweight="bold")
    ax.set_xticks(gens)
    ax.set_xlim(min(gens) - 0.6, max(gens) + 0.6)
    ax.grid(True, alpha=0.25, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    ax.legend(seen.values(), seen.keys(), loc="lower right", fontsize=9,
              framealpha=0.9)
    return fig, ax


def plot_evolution(summary, rows, dft, out_png):
    try:
        fig, ax = _evolution_figure(summary, rows, dft)
    except Exception as exc:                      # pragma: no cover
        return f"  (evolution plot skipped: {exc})"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f"  Saved {out_png}"


def _ei_series(summary):
    """``[{gen, density, sigma, dft}, ...]`` of the EI-labelled crystals, in pick order."""
    series = []
    for gen in sorted(summary):
        for pick in summary[gen].get("picks", []):
            density = pick.get("density", float("nan"))
            if math.isfinite(density):
                series.append({
                    "gen": gen,
                    "density": density,
                    "sigma": pick.get("sigma", float("nan")),
                    "dft": pick.get("dft", float("nan")),
                })
    return series


def _convergence_figure(ei_series):
    """The convergence figure; split out so tests can inspect its axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    if not ei_series:
        ax.text(0.5, 0.5, "no crystal was labelled: ei.log is empty",
                transform=ax.transAxes, ha="center", va="center", fontsize=11,
                color="#c0392b")
        ax.set_xlabel("Generation")
        ax.set_ylabel(r"Density (g/cm$^3$)")
        ax.set_title("EI selection trend across generations", fontsize=12,
                     fontweight="bold")
        return fig, ax

    gens = [entry["gen"] for entry in ei_series]
    dens = [entry["density"] for entry in ei_series]
    sigmas = [entry["sigma"] for entry in ei_series]

    # best-so-far as generations advance
    running, best = float("-inf"), []
    for value in dens:
        running = max(running, value)
        best.append(running)
    ax.step(gens, best, where="post", color="#1a5276", lw=2.0, zorder=4,
            label="best so far")

    # the GP uncertainty (±1.96σ) on each EI-selected density, as error bars
    ax.errorbar(gens, dens, yerr=sigmas, fmt="o", color="#27ae60",
                ecolor="#1e8449", elinewidth=1.2, capsize=3, ms=7, zorder=5,
                label=r"EI-selected density $\pm$1.96$\sigma$")

    # the reference measurement of the same crystal, drawn next to the ML value
    dft_dens, dft_gens = [], []
    for entry in ei_series:
        if math.isfinite(entry["dft"]):
            dft_dens.append(entry["dft"])
            dft_gens.append(entry["gen"])
            ax.plot([entry["gen"], entry["gen"]], [entry["density"], entry["dft"]],
                    color="#7f8c8d", lw=0.8, ls="--", alpha=0.7, zorder=3)
    if dft_dens:
        ax.scatter(dft_gens, dft_dens, marker="D", s=50, color="#c0392b",
                   edgecolors="#641e16", linewidths=0.8, zorder=6,
                   label="DFT density of selected")

    ax.set_xlabel("Generation")
    ax.set_ylabel(r"Density (g/cm$^3$)")
    ax.set_title("EI selection trend across generations", fontsize=12,
                 fontweight="bold")
    ax.set_xticks(sorted(set(gens)))
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    return fig, ax


def plot_convergence(ei_series, out_png):
    """EI selection trend: x = generation, y = density, with GP uncertainty bars.

    Every EI-labelled crystal is a green dot at its generation, with the GP
    uncertainty (±1.96σ on the density) as an error bar; the blue step line is
    the best density found so far as generations advance, and a red diamond is
    the DFT reference of that same crystal.
    """
    try:
        fig, ax = _convergence_figure(ei_series)
    except Exception as exc:                      # pragma: no cover
        return f"  (convergence plot skipped: {exc})"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    message = f"  Saved {out_png}"
    with_sigma = sum(1 for e in ei_series if math.isfinite(e.get("sigma", float("nan"))))
    if ei_series and with_sigma < len(ei_series):
        # Runs made before density_predict.log started accumulating only have
        # the σ of the last generation, so only the newest pick gets a bar.
        message += (f"  (σ for {with_sigma}/{len(ei_series)} EI-selected "
                    f"crystals: this run's density_predict.log was rewritten "
                    f"every generation)")
    return message


def plot_learning(summary, learning, out_png):
    """Per-generation σ of the population, labels and LOO RMSE of the model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                      # pragma: no cover
        return f"  (learning plot skipped: matplotlib unavailable: {exc})"

    gens = sorted(summary)
    sigmas = [summary[gen]["sigma_mean"] for gen in gens]
    labelled = [learning.get(gen, {}).get("added", float("nan")) for gen in gens]
    rmses = [learning.get(gen, {}).get("rmse_density", float("nan")) for gen in gens]
    if not any(map(math.isfinite, sigmas)) and not any(map(math.isfinite, rmses)):
        return "  (learning plot skipped: the run logged neither σ nor a LOO RMSE)"

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    twin = ax.twinx()
    if any(map(math.isfinite, sigmas)):
        ax.plot(gens, sigmas, "o-", color="#8e44ad", lw=1.8, ms=5,
                label="mean GP ±1.96σ (± density)")
        ax.set_ylabel(r"mean GP uncertainty (g/cm$^3$)", color="#8e44ad")
    if any(map(math.isfinite, rmses)):
        twin.plot(gens, rmses, "s--", color="#d35400", lw=1.6, ms=5,
                  label="LOO RMSE (density)")
        twin.set_ylabel(r"LOO RMSE (g/cm$^3$)", color="#d35400")
    if any(map(math.isfinite, labelled)):
        twin.bar(gens, [0.0 if not math.isfinite(v) else v for v in labelled],
                 width=0.5, color="#27ae60", alpha=0.25,
                 label="labels added in this generation")
    ax.set_xlabel("Generation")
    ax.set_xticks(gens)
    ax.set_title("What the labels bought", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")
    main_handles, main_labels = ax.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    handles, labels = main_handles + twin_handles, main_labels + twin_labels
    seen = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=9,
              framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"  Saved {out_png}"


# ──────────────────────────────────────────────────────────────────────
#  report
# ──────────────────────────────────────────────────────────────────────

def report(summary, ei, steps, mean, low, high, learning):
    lines = ["", "======== per-generation report ========"]
    for gen in sorted(summary):
        entry = summary[gen]
        line = (f"gen {gen:3d}  n={entry['n']:4d}  max={entry['max']:.4f}  "
                f"mean={entry['mean']:.4f}")
        if math.isfinite(entry["sigma_mean"]):
            line += f"  σ={entry['sigma_mean']:.4f} ({entry['n_gp']} predicted)"
        lines.append(line)
        if entry["picks"]:
            lines.append("        EI: " + ", ".join(
                f"#{p['id']}(ρ={p['density']:.4f}"
                + (f", gp={p['density_gp']:.4f}" if math.isfinite(p["density_gp"]) else "")
                + (f", dft={p['dft']:.4f}" if math.isfinite(p["dft"]) else "")
                + ")" for p in entry["picks"]))
        if gen in learning:
            lines.append(f"        labelled: +{learning[gen]['added']} "
                         f"(rows {learning[gen]['rows']})")
    if ei and steps and mean:
        verdict = ("above" if ei[-1] > high[min(len(ei), len(high)) - 1]
                   else "inside" if ei[-1] >= low[min(len(ei), len(low)) - 1]
                   else "below")
        lines.append("")
        lines.append(f"EI best labelled = {ei[-1]:.4f} after {len(ei)} labels; "
                     f"random baseline {mean[-1]:.4f} "
                     f"[{low[-1]:.4f}, {high[-1]:.4f}] -> {verdict}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
#  entry point
# ──────────────────────────────────────────────────────────────────────

def _first(existing, *candidates):
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return existing


def _results_dirs(run_dir):
    """The existing ``results<N>`` folders of *run_dir*, ordered by N."""
    found = []
    try:
        names = os.listdir(run_dir)
    except OSError:
        return found
    for name in names:
        match = re.fullmatch(r"results(\d+)", name)
        if match and os.path.isdir(os.path.join(run_dir, name)):
            found.append((int(match.group(1)), name))
    return [os.path.join(run_dir, name) for _, name in sorted(found)]


def resolve_results_dir(run_dir, which=None):
    """The folder of *run_dir* the report reads: ``results<N>`` or *run_dir*.

    *which* selects one explicitly, either as ``results2`` or as ``2``.
    Without it the **newest** ``results<N>`` wins, since that is the run that
    most recently wrote an ``ei.log``; runs that wrote everything into the
    home directory (no ``results<N>`` at all) fall back to *run_dir*.
    """
    if which is not None:
        token = str(which).strip()
        candidate = os.path.join(
            run_dir, token if token.startswith("results") else f"results{token}")
        if not os.path.isdir(candidate):
            raise SystemExit(f"ei-report: {candidate} is not a directory")
        return candidate
    dirs = _results_dirs(run_dir)
    return dirs[-1] if dirs else run_dir


def ei_report(path=".", out=None, trials=200, quiet=False, results=None):
    """Write the three figures of *path* and print the per-generation report."""
    run_dir = os.path.abspath(path)
    res_dir = resolve_results_dir(run_dir, results)

    individuals = _first(None, os.path.join(res_dir, "Individuals"),
                         os.path.join(run_dir, "Individuals"))
    rows, gen_ids = parse_individuals(individuals)
    if not gen_ids:
        return f"No Individuals table under {run_dir}: nothing to plot."

    pred = parse_density_predict(_first(None,
                                        os.path.join(res_dir, "density_predict.log"),
                                        os.path.join(run_dir, "density_predict.log")))
    dft = parse_density_log(_first(None, os.path.join(res_dir, "density.log"),
                                   os.path.join(run_dir, "density.log")))
    # ei.log lives in the run's results<N> now; the home-directory copy is
    # only the fallback for runs finished before that move.
    ei_log = parse_ei_log(_first(None, os.path.join(res_dir, "ei.log"),
                                 os.path.join(run_dir, "ei.log")))
    learning = parse_learning_curve(_first(None, os.path.join(run_dir, "log"),
                                           os.path.join(run_dir, "OUTPUT.txt")))

    summary = build_summary(rows, gen_ids, pred, dft, ei_log)
    by_id = {row[1]: row for row in rows}
    ei = ei_curve(summary, gen_ids, ei_log)
    steps, mean, low, high = random_baseline(gen_ids, by_id, ei_log,
                                             trials=trials)
    ei_series = _ei_series(summary)

    prefix = out or os.path.join(res_dir, "ei_report")
    messages = [f"Generations {min(gen_ids)}–{max(gen_ids)}: "
                f"{len(rows)} structures, {len(pred)} GP predictions, "
                f"{len(dft)} DFT references, {sum(len(v) for v in ei_log.values())} "
                f"EI-selected crystals"]
    messages.append(plot_evolution(summary, rows, dft, prefix + "_evolution.png"))
    messages.append(plot_convergence(ei_series, prefix + "_convergence.png"))
    messages.append(plot_learning(summary, learning, prefix + "_learning.png"))
    text = "\n".join(messages)
    if not quiet:
        print(text)
        print(report(summary, ei, steps, mean, low, high, learning))
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="gpevo ei-report",
        description="Per-generation density evolution and EI convergence of a "
                    "GPEvo run directory.")
    parser.add_argument("-p", "--path", default=".", help="run directory")
    parser.add_argument("--out", default=None,
                        help="output prefix (default <run>/results<N>/ei_report)")
    parser.add_argument("--results", default=None,
                        help="results folder to read, e.g. 2 or results2 "
                             "(default: the newest results<N>)")
    parser.add_argument("--trials", type=int, default=200,
                        help="random-selection draws of the convergence baseline")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="print the per-generation report without plotting")
    args = parser.parse_args(argv)

    if args.dry_run:
        text = ei_report(path=args.path, trials=args.trials, quiet=True,
                         results=args.results)
        print(text)
        return 0
    ei_report(path=args.path, out=args.out, trials=args.trials,
              results=args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
