#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel 2 of the GPEvo diagram -- Expected Improvement acquisition function
computed from the SAME Gaussian-process posterior as panel 1.

STANDALONE: no imports from build_wave.py, no third-party Python packages.
Copy this single file anywhere and run it. It writes:

    ei-acquisition.svg    (editable vector, live text)
    ei-acquisition.png    (2x raster, only if `rsvg-convert` is on PATH)

Usage
-----
    python3 plot_ei_acquisition.py
    python3 plot_ei_acquisition.py --no-guide       # drop the dashed query line
    python3 plot_ei_acquisition.py --scale 3 --outdir /tmp

Provenance
----------
Extracted verbatim (geometry, colours and GP maths) from
`outputs/gpevo-logo/build_wave.py`, section "4. annotated scientific
diagram", lines ~715-742. The EI panel box EX/EY/EW/EH and the
normalisation are intentionally kept identical to annotated-wave.svg;
a single <g transform> crops the panel into a compact canvas, so the EI
curve path `d` string is byte-identical to the full diagram.

Formula (matches the implementation exactly)
--------------------------------------------
    EI(x) = (mu(x) - f* - xi) * Phi(Z) + sigma(x) * phi(Z)
    Z     = (mu(x) - f* - xi) / sigma(x),   f* = max_i y_i
    Phi(Z)= 0.5*(1 + erf(Z/sqrt(2))),  phi(Z)= exp(-Z^2/2)/sqrt(2*pi)

LaTeX source for this formula is in wave/latex/ei-formula-inline.tex
(inline, sans-serif) and wave/latex/ei-formula.tex (full, 3-line).

Facts / caveats (carry these into any figure caption)
----------------------------------------------------
* EI is a real computation on the schematic GP posterior, not a hand-drawn
  curve; the posterior bands use a bounded visual compression. No
  experimental data are shown.
* The strict argmax of EI lies at the extrapolation boundary; the dashed
  line at x=0.90 marks a representative high-uncertainty candidate.
* Font is live text -- convert to outlines before print / submission.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess

# ---------------------------------------------------------------------------
# 1. Model parameters  (identical to build_wave.py, lines 55-64)
# ---------------------------------------------------------------------------
OBS_X = [0.14, 0.34, 0.56, 0.80]
OBS_Y = [0.70, 0.30, 0.76, 0.44]
M0 = 0.52
ELL, SF2, SN2 = 0.125, 1.0, 0.008
XI = 0.01
F_STAR = max(OBS_Y)
HW, K_SIG = 0.38, 0.80
N_GRID = 160
QUERY_X = 0.90

# ---------------------------------------------------------------------------
# 2. Real GP posterior + Expected Improvement (same maths as build_wave.py)
# ---------------------------------------------------------------------------


def kern(a, b):
    return SF2 * math.exp(-((a - b) ** 2) / (2.0 * ELL ** 2))


def mat_inv(A):
    """Gauss-Jordan inverse; avoids a numpy dependency."""
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        for j in range(2 * n):
            M[col][j] /= p
        for r in range(n):
            if r == col:
                continue
            fac = M[r][col]
            if fac:
                for j in range(2 * n):
                    M[r][j] -= fac * M[col][j]
    return [row[n:] for row in M]


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_pdf(z):
    return math.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi)


K = [[kern(a, b) + (SN2 if i == j else 0.0)
      for j, b in enumerate(OBS_X)] for i, a in enumerate(OBS_X)]
KINV = mat_inv(K)
RES = [y - M0 for y in OBS_Y]
ALPHA = [sum(KINV[i][j] * RES[j] for j in range(len(RES)))
         for i in range(len(RES))]

GRID, MU, SIG, EI = [], [], [], []
for i in range(N_GRID):
    x = i / (N_GRID - 1)
    ks = [kern(x, xo) for xo in OBS_X]
    mu = M0 + sum(ks[a] * ALPHA[a] for a in range(len(OBS_X)))
    var = SF2 - sum(ks[a] * KINV[a][b] * ks[b]
                    for a in range(len(OBS_X)) for b in range(len(OBS_X)))
    sigma = math.sqrt(max(var, 0.0))
    imp = mu - F_STAR - XI
    z = imp / sigma if sigma > 1e-9 else -10.0
    ei = imp * norm_cdf(z) + sigma * norm_pdf(z) if sigma > 1e-9 else 0.0
    GRID.append(x)
    MU.append(mu)
    SIG.append(sigma)
    EI.append(max(ei, 0.0))

EI_MAX = max(EI)
Q_IDX = min(range(N_GRID), key=lambda i: abs(GRID[i] - QUERY_X))

# ---------------------------------------------------------------------------
# 3. Palette, fonts and geometry
# ---------------------------------------------------------------------------
NAVY = "#0B2545"
VIOLET = "#A855F7"
L_MUTED = "#5B728C"
L_GRID = "#D8E6F5"
QUERY_TEXT = "#C2700D"
FONT = "'DejaVu Sans','Source Han Sans SC','Noto Sans CJK SC',sans-serif"

# EI panel box: EXACTLY the same numbers as annotated-wave.svg.
AX = 110                 # title / left axis reference x
EX, EY, EW, EH = 110, 670, 1030, 180
TX, TY = -86, -636       # crop translation
W, H = 1080, 260


def n(v):
    return ("%.2f" % v).rstrip("0").rstrip(".")


def segs(points):
    """Catmull-Rom -> cubic Bezier, for a smooth curve."""
    out = []
    L = len(points)
    for i in range(L - 1):
        p0 = points[i - 1] if i > 0 else points[0]
        p1, p2 = points[i], points[i + 1]
        p3 = points[i + 2] if i + 2 < L else points[-1]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        out.append("C %s,%s %s,%s %s,%s" %
                   (n(c1[0]), n(c1[1]), n(c2[0]), n(c2[1]),
                    n(p2[0]), n(p2[1])))
    return out


def path_d(points):
    return "M %s,%s " % (n(points[0][0]), n(points[0][1])) + " ".join(segs(points))


def build(show_guide=True):
    body = []

    body.append(f'<text x="{AX}" y="{EY-14}" font-family="{FONT}" '
                f'font-size="16" font-weight="700" fill="{NAVY}">'
                f'EI(x)  ·  Expected Improvement acquisition</text>')
    body.append(f'<rect x="{EX}" y="{EY}" width="{EW}" height="{EH}" '
                f'fill="#FBFDFF" stroke="{L_GRID}"/>')

    # filled area + EI curve
    ep = []
    for i, x in enumerate(GRID):
        px = EX + x * EW
        py = EY + EH - (EI[i] / EI_MAX) * (EH - 22) - 10
        ep.append((px, py))
    ed = path_d(ep) + f' L {n(EX+EW)},{n(EY+EH)} L {n(EX)},{n(EY+EH)} Z'
    body.append(f'<path d="{ed}" fill="{VIOLET}" opacity="0.13"/>')
    body.append(f'<path d="{path_d(ep)}" fill="none" stroke="{VIOLET}" '
                f'stroke-width="3" stroke-linejoin="round"/>')

    # reference gridlines
    for frac in (0.0, 0.5, 1.0):
        px = EX + frac * EW
        body.append(f'<line x1="{n(px)}" y1="{EY}" x2="{n(px)}" y2="{EY+EH}" '
                    f'stroke="{L_GRID}" stroke-width="1"/>')

    # query marker
    if show_guide:
        body.append(f'<line x1="{EX+QUERY_X*EW}" y1="{EY}" '
                    f'x2="{EX+QUERY_X*EW}" y2="{EY+EH}" '
                    f'stroke="{QUERY_TEXT}" stroke-width="1.6" '
                    f'stroke-dasharray="5 5"/>')
        body.append(f'<text x="{EX+QUERY_X*EW-8}" y="{EY+EH-8}" '
                    f'text-anchor="end" font-family="{FONT}" font-size="12" '
                    f'fill="{QUERY_TEXT}">query</text>')

    body.append(f'<text x="{EX+EW}" y="{EY+EH+28}" text-anchor="end" '
                f'font-family="{FONT}" font-size="13" fill="{L_MUTED}">'
                f'EI rises again where σ is large  →  exploration</text>')

    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}">\n'
            f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>\n'
            f'<g transform="translate({TX},{TY})">\n'
            + "\n".join(body) + '\n</g>\n</svg>\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-guide", action="store_true",
                    help="omit the dashed vertical query line and its label")
    ap.add_argument("--scale", type=int, default=2,
                    help="PNG supersampling factor (default: 2)")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="output directory (default: this script's folder)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sp = os.path.join(args.outdir, "ei-acquisition.svg")
    with open(sp, "w", encoding="utf-8") as fh:
        fh.write(build(show_guide=not args.no_guide))

    pp = os.path.join(args.outdir, "ei-acquisition.png")
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-w", str(W * args.scale),
                        "-h", str(H * args.scale), "-o", pp, sp], check=True)
    else:
        print("NOTE: rsvg-convert not found - SVG written, PNG skipped.")

    print("EI: max %.4f | left/right edge %.4f / %.4f | at query x=%.2f %.4f"
          % (EI_MAX, EI[0], EI[-1], QUERY_X, EI[Q_IDX]))
    print("wrote:", sp)
    if os.path.exists(pp):
        print("wrote:", pp)


if __name__ == "__main__":
    main()
