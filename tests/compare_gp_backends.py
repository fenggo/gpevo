#!/usr/bin/env python
"""Single-panel, publication-style figure comparing Gaussian-process density
predictions and their predictive uncertainties between the scikit-learn
(results1) and GPyTorch (results2) backends for one identical seed generation.

Run with: /home/feng/.local/anaconda/bin/python compare_gp_backends.py
"""

import csv
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parent
SK = BASE / "results1"          # scikit-learn backend (default)
GP = BASE / "results2"          # GPyTorch backend
OUT = BASE

# Serif face with STIX math gives a Times-like appearance suited to journals.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.unicode_minus": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.9,
    "figure.dpi": 130,
})

C_SK = "#1f77b4"   # scikit-learn
C_GP = "#d62728"   # GPyTorch

# ID Origin [composition] Enthalpy(eV) Volume [kpoints] SYMMETRY
# Enthalpy(GP) UC Density(GP) UC Residual
ROW_RE = re.compile(
    r"^\s*(?P<id>\d+)\s+(?P<origin>\S+)\s+"
    r"\[[^\]]*\]\s+"
    r"(?P<enthalpy>-?\d+\.\d+)\s+(?P<volume>\d+\.\d+)\s+"
    r"\[[^\]]*\]\s+(?P<sym>\d+)\s+"
    r"(?P<e_pred>-?\d+\.\d+)\s+(?P<uc_e>\d+\.\d+)\s+"
    r"(?P<d_pred>-?\d+\.\d+)\s+(?P<uc_d>\d+\.\d+)\s+"
    r"(?P<residual>\d+\.\d+)\s*$"
)

FIELDS = ("e_pred", "uc_e", "d_pred", "uc_d")


def parse(path):
    rows = []
    inside = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("ID  Origin"):
            inside = True
            continue
        if not inside:
            continue
        m = ROW_RE.match(line)
        if not m:
            if rows:
                break
            continue
        d = m.groupdict()
        rows.append(
            {
                "id": int(d["id"]),
                "enthalpy": float(d["enthalpy"]),
                "volume": float(d["volume"]),
                "residual": float(d["residual"]),
                **{k: float(d[k]) for k in FIELDS},
            }
        )
    return rows


def stats(a, b):
    diff = np.asarray(a) - np.asarray(b)
    r = np.corrcoef(a, b)[0, 1]
    return {
        "r2": r ** 2,
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "maxabs": float(np.max(np.abs(diff))),
        "ratio": float(np.mean(np.asarray(a) / np.asarray(b))),
    }


def main():
    sk, gp = parse(SK / "OUTPUT.txt"), parse(GP / "OUTPUT.txt")
    if not sk or not gp:
        raise SystemExit("Failed to parse the OUTPUT.txt table; check the file format.")
    assert len(sk) == len(gp), f"Row counts differ: {len(sk)} vs {len(gp)}"
    assert [r["id"] for r in sk] == [r["id"] for r in gp], "Structure IDs differ"

    for key in ("enthalpy", "volume", "residual"):
        delta = np.max(np.abs([r[key] for r in sk] - np.array([r[key] for r in gp])))
        print(f"[Consistency check] {key:9s} max |difference| = {delta:.6g}")

    ids = np.array([r["id"] for r in sk])
    A = {k: np.array([r[k] for r in sk]) for k in FIELDS}
    B = {k: np.array([r[k] for r in gp]) for k in FIELDS}

    # ---- Tabulated data ----
    csv_path = OUT / "gp_backend_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "density_pred_sklearn", "density_UC_sklearn",
                    "density_pred_gpytorch", "density_UC_gpytorch", "d_pred", "d_UC"])
        for i in range(len(ids)):
            w.writerow([ids[i], A["d_pred"][i], A["uc_d"][i],
                        B["d_pred"][i], B["uc_d"][i],
                        A["d_pred"][i] - B["d_pred"][i],
                        A["uc_d"][i] - B["uc_d"][i]])
    print(f"[Output] {csv_path}")

    labels = {"e_pred": "Enthalpy(GP)  [eV]",
              "uc_e": "UC(energy)",
              "d_pred": "Density(GP)  [g/cm3]",
              "uc_d": "UC(density)"}
    print("\n" + "=" * 92)
    print(f"{'Quantity':<38}{'scikit-learn mean +/- std':>28}"
          f"{'GPyTorch mean +/- std':>23}{'R^2':>8}")
    print("=" * 92)
    for k in FIELDS:
        s = stats(A[k], B[k])
        print(f"{labels[k]:<38}{np.mean(A[k]):>14.5f} +/-{np.std(A[k]):<9.5f}"
              f"{np.mean(B[k]):>12.5f} +/-{np.std(B[k]):<9.5f}{s['r2']:>8.4f}")
    print("=" * 92)

    # ---- Single panel: GP-predicted density with predictive uncertainty ----
    m1, u1 = A["d_pred"], A["uc_d"]     # scikit-learn
    m2, u2 = B["d_pred"], B["uc_d"]     # GPyTorch

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.fill_between(ids, m1 - u1, m1 + u1, color=C_SK, alpha=0.22,
                    edgecolor=C_SK, linewidth=0.7, linestyle=":")
    ax.fill_between(ids, m2 - u2, m2 + u2, color=C_GP, alpha=0.22,
                    edgecolor=C_GP, linewidth=0.7, linestyle=":")
    ax.plot(ids, m1, color=C_SK, lw=1.4)
    ax.plot(ids, m2, color=C_GP, lw=1.4, ls="--")

    handles = [
        Line2D([0], [0], color=C_SK, lw=1.6, label="Mean (scikit-learn)"),
        Line2D([0], [0], color=C_GP, lw=1.6, ls="--", label="Mean (GPyTorch)"),
        Patch(facecolor=C_SK, alpha=0.35, edgecolor="none",
              label=r"$\pm 1\sigma$ (scikit-learn)"),
        Patch(facecolor=C_GP, alpha=0.35, edgecolor="none",
              label=r"$\pm 1\sigma$ (GPyTorch)"),
    ]
    ax.legend(handles=handles, fontsize=10, ncol=2, loc="lower right",
              framealpha=0.95, edgecolor="0.8")

    ax.set_xlabel("Structure index", fontsize=11)
    ax.set_ylabel(r"GP-predicted density, $\rho$  (g cm$^{-3}$)", fontsize=11)
    # ax.set_xlim(left=1)
    ax.set_xlim(1,116)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.6)

    # fig.suptitle(
    #     "Gaussian Process Predictions of Density and Associated Uncertainty: "
    #     "scikit-learn versus GPyTorch",
    #     fontsize=13, fontweight="normal", y=0.975,
    # )
    # fig.text(0.5, 0.918,
    #          r"Generation 1, $N = 116$ identical seed structures. "
    #          r"Shaded regions denote $\mu \pm 1\sigma$.",
    #          ha="center", fontsize=10, color="0.3")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.11, top=0.875)

    out_png = OUT / "gp_prediction_with_uncertainty.png"
    out_pdf = OUT / "gp_prediction_with_uncertainty.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")   # vector version for submission
    plt.close(fig)
    print(f"[Output] {out_png}")
    print(f"[Output] {out_pdf}")


if __name__ == "__main__":
    main()
