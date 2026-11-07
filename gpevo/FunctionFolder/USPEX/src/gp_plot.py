"""GP status plots for a USPEX generation — native port of ``uspexkit denevo``.

Unlike the CLI tool this module does **not** re-parse ``Individuals`` /
``gp.csv`` / ``density.log``: every value comes from memory, i.e. from the
dict ``uspexkit.gp()`` returned for each structure (stored by submitJob as
``s.gp``, see gp_ml.apply_gp_metrics).  No file round-trip, no id matching,
and the figure can show quantities the files never carried (per-structure
uncertainty, EI, residual, which density was actually used for the fitness).

Four panels, all driven by the same in-memory records:

  A  population distribution of the GP-predicted density, with the top-K,
     the EI-selected and the DFT-verified structures marked;
  B  exploration map (prediction vs uncertainty) with the ``u`` / ``den``
     INPUT thresholds drawn, i.e. a picture of the active-learning setup;
  C  GP accuracy: GP-predicted density vs the DFT-measured density from
     density.log (identity line, ±1.96σ error bars, RMSE + bias labels);
  D  extrapolation fraction across generations, read from the
     gp_status_history.csv the module itself writes (how many structures
     fall beyond the residual cutoff, per generation).
"""
from __future__ import annotations

import math
import os

import numpy as np

from gpevo.log_utils import console_print, write_log
from gpevo.runtime import RESIDUAL_CUT
from gpevo.FunctionFolder.USPEX.src.gp_ml import (
    _expected_improvement, observed_density_incumbent, parse_gp_params,
    UNCERT_SCALE,
)

# muted, print-friendly palette (close to the denevo one)
C_POP = "#4C72B0"          # population
C_POP_OUT = "#A8B4C8"      # extrapolated (residual > cut)
C_TOP = "#C56C39"          # top-K by predicted density
C_EI = "#2F6D5A"           # EI-selected
C_DFT = "#C84545"          # DFT-verified
C_LINE = "#7A7A7A"
C_GATE = "#B0B0B0"


# --------------------------------------------------------------------- records
def collect_records(POP):
    """Per-structure GP records for the current population.

    Returns a list of dicts (only structures that carry ``s.gp``), each with
    number / origin / predicted density / uncertainty / residual / whether
    the GP step ran a DFT / the density finally used for the fitness.
    """
    records = []
    for s in POP.POPULATION:
        gp = getattr(s, "gp", None)
        if not isinstance(gp, dict):
            continue
        try:
            dgp = float(gp.get("density_gp"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(dgp):
            continue

        def _f(key):
            try:
                value = float(gp.get(key))
            except (TypeError, ValueError):
                return float("nan")
            return value

        records.append({
            "number": int(getattr(s, "Number", 0)),
            "origin": str(getattr(s, "howCome", "") or ""),
            "dgp": dgp,
            # ``sigma`` stays the reported 95% half-width for the error bars
            # and the ``u`` gate; EI's closed form needs the posterior std.
            "sigma": _f("uncertainty"),
            "sd": _f("uncertainty") / UNCERT_SCALE,
            "residual": _f("residual"),
            "dft": bool(gp.get("dft", False)),
            "energy_pred": _f("energy_pred"),
            "uncert_eng": _f("uncertainty_eng"),
        })
    return records


def _ei_preview(records, params, incumbent=None):
    """What the active learning would select on this population.

    Same rules as gp_active_learning (gp_ml.py): EI per crystal (on the
    posterior std ``sd``), top-k, then keep only 1.96*sd > u and dgp >= den.
    ``u`` is a floor on the reported 95% half-width, matching
    gp_ml._select_by_ei.  ``incumbent`` is the observed DFT best density the
    real selection uses; without it the report falls back to max(dgp), which
    is a *different* f_best and made ct22/results2 show an EI top-1 the run
    never considered.
    """
    valid = [r for r in records if r["residual"] <= RESIDUAL_CUT]
    if not valid:
        return [], float("-inf"), []
    f_best = (incumbent if incumbent is not None and math.isfinite(incumbent)
              else max(r["dgp"] for r in valid))
    scored = []
    for r in valid:
        ei = _expected_improvement(r["dgp"], r["sd"], f_best)
        scored.append((ei, r))
    scored.sort(key=lambda t: -t[0])
    top = [(ei, r) for ei, r in scored[:max(1, int(params["k"]))]]
    selected = [(ei, r) for ei, r in top
                if r["sigma"] > params["u"] and r["dgp"] >= params["den"]]
    top_density = sorted(valid, key=lambda r: -r["dgp"])[:5]
    return selected, f_best, top_density


# ---------------------------------------------------------------------- figure
def _formula(numIons, atomType):
    """Compact formula like ``C96 H64 N48 O48`` for the title."""
    try:
        from gpevo.FunctionFolder.USPEX.src.Initialize import element_symbol
        parts = []
        for z, n in zip(atomType, numIons):
            n = int(n)
            if n:
                parts.append(f"{element_symbol(int(z))}{n}")
        return " ".join(parts) if parts else "?"
    except Exception:
        return "?"


def plot_gp_generation(ORG, POP, fitness=None, out_path=None):
    """Draw the GP status figure of the current generation.

    Returns the path of the written PNG, or None when there is nothing to
    draw (no GP metrics yet) or matplotlib is unavailable.  Never raises:
    a plotting problem must not kill a running USPEX job.
    """
    records = collect_records(POP)
    generation = int(getattr(POP, "generation", 0))
    out_dir = (getattr(ORG, "resFolder", None) or getattr(ORG, "homePath", "")
               or os.getcwd())
    dft_path = None
    for _cand in (os.path.join(out_dir, "density.log"),
                  os.path.join(getattr(ORG, "homePath", "") or ".", "density.log")):
        if _cand and os.path.isfile(_cand):
            dft_path = _cand
            break
    dft_map = _read_dft(dft_path)
    for _r in records:
        _d = dft_map.get(_r["number"])
        _r["dft_density"] = _d[0] if _d else float("nan")
    if not records:
        console_print("  GP plot: no structure carries GP metrics, skipped")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception as exc:
        console_print(f"  GP plot: matplotlib unavailable ({exc}), skipped")
        return None

    params = parse_gp_params(ORG)
    # The status report must rank EI against the same incumbent as the real
    # selection; only the density objective shares that physical quantity.
    incumbent = None
    if int(params.get("optType", 11) or 11) != 1:
        incumbent = observed_density_incumbent(ORG)
    selected, f_best, top_density = _ei_preview(records, params, incumbent)
    sel_numbers = {r["number"] for _, r in selected}
    top_numbers = {r["number"] for r in top_density}
    inside = [r for r in records if r["residual"] <= RESIDUAL_CUT]
    outside = [r for r in records if r["residual"] > RESIDUAL_CUT]

    # record this generation in the extrapolation-history file (panel D)
    hist_path = os.path.join(out_dir, _HISTORY_NAME)
    hist = _load_history(hist_path)
    hist = [h for h in hist if h["gen"] != generation]
    hist.append({"gen": generation, "n": len(records),
                 "extrapolated": len(outside)})
    _save_history(hist_path, hist)
    dft_records = [r for r in records if r["dft"]]

    # ---- title -------------------------------------------------------------
    s0 = POP.POPULATION[0]
    formula = _formula(getattr(s0, "numIons", []), getattr(ORG, "atomType", []))
    title = (f"{formula}   |   generation {generation}   |   "
             f"{len(records)} structures, {len(inside)} inside the training "
             f"manifold")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # ================= A: population distribution ==========================
    vals_in = [r["dgp"] for r in inside] or [0.0]
    vals_out = [r["dgp"] for r in outside]
    lo = min(min(vals_in), min(vals_out) if vals_out else min(vals_in))
    hi = max(max(vals_in), max(vals_out) if vals_out else max(vals_in))
    pad = max(0.02, 0.05 * (hi - lo))
    bins = np.linspace(lo - pad, hi + pad, 25)
    ax_a.hist(vals_in, bins=bins, color=C_POP, alpha=0.85,
              label=f"residual ≤ {RESIDUAL_CUT:g} ({len(inside)})")
    if vals_out:
        ax_a.hist(vals_out, bins=bins, color=C_POP_OUT, alpha=0.9,
                  label=f"extrapolated ({len(vals_out)})")
    # Anchor the histogram to the population's own best prediction; the
    # observed incumbent can lie well outside the plotted density range.
    if top_density:
        ax_a.axvline(top_density[0]["dgp"], color="black", lw=1.0, ls=":",
                     label=f"max GP density = {top_density[0]['dgp']:.4f}")
    # rug strip under the histogram: every structure as a tick, coloured by
    # the role it plays (clearer than stacking vertical lines on the bars)
    strip_y = -0.06 * ax_a.get_ylim()[1]
    groups = [(top_density, C_TOP, "top-5 density"),
              ([r for _, r in selected], C_EI, "EI-selected"),
              (dft_records, C_DFT, "DFT-verified"),
              (outside, C_POP_OUT, "extrapolated")]
    for members, color, label in groups:
        if members:
            ax_a.plot([r["dgp"] for r in members], [strip_y] * len(members),
                      "|", ms=9, mew=1.6, color=color, label=label)
    ax_a.set_ylim(bottom=2 * strip_y)
    ax_a.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_a.set_xlabel(r"GP-predicted density (g/cm$^3$)")
    ax_a.set_ylabel("structures")
    ax_a.set_title("A  population assessment", loc="left", fontsize=11)
    ax_a.legend(fontsize=7.5, frameon=False, loc="upper left")

    # ================= B: exploration map ==================================
    for r in records:
        inside_flag = r["residual"] <= RESIDUAL_CUT
        color = C_POP if inside_flag else C_POP_OUT
        ax_b.scatter(r["dgp"], r["sigma"], s=42, color=color,
                     edgecolors="white", linewidths=0.5, zorder=3,
                     alpha=0.9 if inside_flag else 0.6)
    if records:
        xs = [r["dgp"] for r in records]
        ys = [r["sigma"] for r in records]
        # the region the active learning sends to DFT: dgp >= den and sigma > u
        x0, x1 = max(min(xs), params["den"]), max(xs)
        y0, y1 = params["u"], max(ys)
        if x0 < x1 and y0 < y1:
            ax_b.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                         facecolor=C_EI, alpha=0.07, zorder=1))
        if params["den"] > 0:
            ax_b.axvline(params["den"], color=C_GATE, lw=1.0, ls="--")
            ax_b.text(params["den"], max(ys) * 0.97, f" den={params['den']:g}",
                      rotation=90, va="top", ha="left", fontsize=7.5,
                      color="#555555")
        ax_b.axhline(params["u"], color=C_GATE, lw=1.0, ls="--")
        ax_b.text(min(xs), params["u"], f"u={params['u']:g} ",
                  va="bottom", ha="left", fontsize=7.5, color="#555555")
    for _, r in selected:
        ax_b.scatter(r["dgp"], r["sigma"], s=200, facecolors="none",
                     edgecolors=C_EI, linewidths=1.4, zorder=5)
        ax_b.annotate(f"#{r['number']}", (r["dgp"], r["sigma"]),
                      textcoords="offset points", xytext=(7, 7), fontsize=7.5,
                      color=C_EI, fontweight="bold")
    for r in dft_records:
        ax_b.scatter(r["dgp"], r["sigma"], s=70, marker="D", color=C_DFT,
                     edgecolors="#7E2023", linewidths=0.6, zorder=4)
    ax_b.set_xlabel(r"GP-predicted density (g/cm$^3$)")
    ax_b.set_ylabel(r"uncertainty $\pm 1.96\sigma$")
    ax_b.set_title(f"B  exploration map — EI would pick "
                   f"{len(selected)}/{len(records)} for DFT", loc="left",
                   fontsize=11)

    # ================= C: GP accuracy vs DFT measurement =================
    # ``dft_density`` was populated from density.log above.  This is the
    # panel that actually validates the GP: prediction on the x-axis,
    # ground-truth DFT density on the y-axis, identity line through the
    # origin and the usual ±1.96σ horizontal error bars.
    dft_recs = [r for r in records
                if math.isfinite(r.get("dft_density", float("nan")))]
    if dft_recs:
        verified = [r for r in dft_recs if r["dft"]]
        reference = [r for r in dft_recs if not r["dft"]]
        for members, color, lab in ((verified, C_DFT, "DFT-verified"),
                                    (reference, C_POP, "DFT reference")):
            if not members:
                continue
            ax_c.errorbar([m["dgp"] for m in members],
                          [m["dft_density"] for m in members],
                          xerr=[max(m["sigma"], 0.0)
                                if math.isfinite(m["sigma"]) else 0.0
                                for m in members],
                          fmt="o", ms=5, lw=0.9, capsize=2, color=color,
                          markeredgecolor="white", markeredgewidth=0.4,
                          alpha=0.9, zorder=3, label=lab)
        xs = [r["dgp"] for r in dft_recs]
        ys = [r["dft_density"] for r in dft_recs]
        lim_lo = min(min(xs), min(ys)) - 0.05
        lim_hi = max(max(xs), max(ys)) + 0.05
        ax_c.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=C_LINE, lw=1.0,
                  ls="--", zorder=2, label="identity")
        ax_c.set_xlim(lim_lo, lim_hi)
        ax_c.set_ylim(lim_lo, lim_hi)
        err = np.array([used - pred for pred, used in zip(ys, xs)])
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        ax_c.text(0.03, 0.97,
                  f"RMSE = {rmse:.4f}  bias = {bias:+.4f}  (n={len(dft_recs)})",
                  transform=ax_c.transAxes, va="top", fontsize=8.5)
        ax_c.legend(fontsize=7.5, frameon=False, loc="lower right")
    else:
        ax_c.text(0.5, 0.5, "no DFT reference (density.log) yet",
                  transform=ax_c.transAxes, ha="center", va="center",
                  fontsize=9, color="#777777")
    ax_c.set_xlabel(r"GP-predicted density (g/cm$^3$)")
    ax_c.set_ylabel(r"DFT-measured density (g/cm$^3$)")
    ax_c.set_title("C  GP accuracy vs DFT (identity line, bars = ±1.96σ)",
                   loc="left", fontsize=11)

    # ================= D: extrapolation fraction across generations =======
    # Reads the gp_status_history.csv written (or updated) just above.  The
    # fraction of structures beyond the residual cutoff is the active-
    # learning health metric: a falling line means the GP's training
    # manifold is catching up with the search.
    hist_path = os.path.join(out_dir, _HISTORY_NAME)
    hist = _load_history(hist_path)
    if hist:
        gens = [h["gen"] for h in hist]
        frac = [100.0 * h["extrapolated"] / max(1, h["n"]) for h in hist]
        ax_d.plot(gens, frac, color=C_POP, lw=1.6, marker="o", ms=4,
                  zorder=3, label="extrapolation fraction")
        ax_d.fill_between(gens, frac, color=C_POP, alpha=0.10, zorder=1)
        ax_d.set_ylim(0, max(105.0, max(frac) + 5))
        cur = [h for h in hist if h["gen"] == generation]
        if cur:
            cx = cur[0]["gen"]
            cy = 100.0 * cur[0]["extrapolated"] / max(1, cur[0]["n"])
            ax_d.scatter([cx], [cy], color=C_TOP, s=55, zorder=5,
                         label=f"this gen = {cy:.0f}%")
        ax_d.set_xlabel("generation")
        ax_d.set_ylabel("structures beyond the residual cutoff (%)")
        ax_d.set_title("D  extrapolation fraction across generations",
                       loc="left", fontsize=11)
        ax_d.legend(fontsize=7.5, frameon=False, loc="upper right")
    else:
        ax_d.text(0.5, 0.5, "history not available",
                  transform=ax_d.transAxes, ha="center", va="center",
                  fontsize=9, color="#777777")
        ax_d.set_xlabel("generation")
        ax_d.set_ylabel("structures beyond the residual cutoff (%)")
        ax_d.set_title("D  extrapolation fraction across generations",
                       loc="left", fontsize=11)

    fig.suptitle(title, fontsize=12.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    if out_path is None:
        out_dir = getattr(ORG, "resFolder", None) or getattr(ORG, "homePath", "") \
            or os.getcwd()
        out_path = os.path.join(out_dir, f"gp_gen{generation}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    _report(records, selected, f_best, top_density, inside, outside, params,
            incumbent)
    console_print(f"  GP plot: {out_path}")
    try:
        write_log(f"GP status plot (generation {generation}): {out_path}\n")
    except Exception:
        pass
    return out_path


def _report(records, selected, f_best, top_density, inside, outside, params,
            incumbent=None):
    """Terminal summary — the numbers behind the figure."""
    console_print(f"  --- GP status: {len(records)} structures, "
                  f"{len(inside)} inside / {len(outside)} extrapolated ---")
    if math.isfinite(f_best):
        source = ("observed DFT" if incumbent is not None
                  and math.isfinite(incumbent) else "GP density fallback")
        console_print(f"  f_best (EI incumbent, {source}) = {f_best:.4f}")
    if top_density:
        best = top_density[0]
        console_print(f"  max GP density = {best['dgp']:.4f} "
                      f"[#{best['number']}]")
        console_print("  top-5 by GP density: " + ", ".join(
            f"#{r['number']}({r['dgp']:.4f})" for r in top_density))
    if selected:
        console_print(f"  EI top-{int(params['k'])} "
                      f"(u={params['u']:g}, den={params['den']:g}): " + ", ".join(
                          f"#{r['number']}(EI={ei:.5f}, σ={r['sd']:.4f}, "
                          f"95%={r['sigma']:.4f})"
                          for ei, r in selected))
    else:
        console_print(f"  EI: no crystal above u={params['u']:g} "
                      f"and den={params['den']:g}")
    dft_numbers = [r["number"] for r in records if r["dft"]]
    if dft_numbers:
        console_print(f"  DFT-verified this generation: "
                      f"{', '.join('#' + str(n) for n in dft_numbers)}")

# ──────────────────────────────────────────────────────────────────────
#  helpers: DFT reference + cross-generation extrapolation history
# ──────────────────────────────────────────────────────────────────────
_HISTORY_NAME = "gp_status_history.csv"


def _read_dft(path):
    """{id: (density, energy)} from a USPEX density.log (the DFT reference).

    Empty dict when *path* is missing -- the accuracy panel then shows a hint
    instead of crashing.
    """
    dft = {}
    if not path or not os.path.isfile(path):
        return dft
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3 and not parts[0].startswith("#"):
                    try:
                        dft[int(float(parts[0]))] = (float(parts[1]),
                                                      float(parts[2]))
                    except ValueError:
                        continue
    except OSError:
        return dft
    return dft


def _load_history(path):
    """List of per-generation rows from the history csv, or []."""
    if not os.path.isfile(path):
        return []
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                try:
                    rows.append({"gen": int(parts[0]),
                                "n": int(parts[1]),
                                "extrapolated": int(parts[2])})
                except (IndexError, ValueError):
                    continue
    except OSError:
        return []
    return rows


def _save_history(path, rows):
    """Persist the per-generation history (dedupe by generation)."""
    seen = {}
    for row in rows:
        seen[row["gen"]] = row
    ordered = sorted(seen.values(), key=lambda r: r["gen"])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# gen, n_structures, n_extrapolated\n")
        for row in ordered:
            fh.write(f"{row['gen']},{row['n']},{row['extrapolated']}\n")


