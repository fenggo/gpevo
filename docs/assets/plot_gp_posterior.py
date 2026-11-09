#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel 1 of the GPEvo diagram -- Gaussian-process posterior with layered
uncertainty bands and the active-learning query point.

STANDALONE: no imports from build_wave.py, no third-party Python packages.
Copy this single file anywhere and run it. It writes:

    gp-posterior.svg    (editable vector, live text)
    gp-posterior.png    (2x raster, only if `rsvg-convert` is on PATH)

Usage
-----
    python3 plot_gp_posterior.py
    python3 plot_gp_posterior.py --no-guide          # drop the dashed leader
    python3 plot_gp_posterior.py --scale 3 --outdir /tmp

Provenance
----------
Extracted verbatim (geometry, colours and GP maths) from
`outputs/gpevo-logo/build_wave.py`, section "4. annotated scientific
diagram", lines ~589-672. The main plot box AX/AY/AW/AH and the value
mapping are intentionally kept identical; a single <g transform> crops the
panel into a compact canvas. Therefore every path `d` string below is
byte-identical to the corresponding path in annotated-wave.svg.

Facts / caveats (carry these into any figure caption)
----------------------------------------------------
* The +/-1sigma and +/-1.96sigma bands use a bounded visual compression
  HW*(1-exp(-K_SIG*k*sigma)) for legibility. This is a schematic mark,
  NOT a plot of raw sigma values or experimental data.
* The ring marks a representative high-uncertainty candidate (x=0.90);
  the strict argmax of EI lies at the extrapolation boundary.
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
OBS_Y = [0.70, 0.30, 0.76, 0.44]        # up-down-up-down -> a wavy posterior
M0 = 0.52                               # constant prior mean
ELL, SF2, SN2 = 0.125, 1.0, 0.008       # RBF lengthscale, signal var, noise var
XI = 0.01                               # EI exploration parameter
F_STAR = max(OBS_Y)
HW, K_SIG = 0.38, 0.80                  # bounded visual band compression
N_GRID = 160
U_LO, U_HI = -0.05, 1.05               # value span mapped to the plot height
QUERY_X = 0.90                          # representative high-sigma query point

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


def half(k, sigma):
    """Bounded, monotone compression of k*sigma into the drawing's value space."""
    return HW * (1.0 - math.exp(-K_SIG * k * sigma))


HALF1 = [half(1.00, s) for s in SIG]        # +/-1 sigma  (inner band)
HALF2 = [half(1.96, s) for s in SIG]        # +/-1.96 sigma (95% band)
HALF3 = [half(2.45, s) for s in SIG]        # faint outer halo
EI_MAX = max(EI)
Q_IDX = min(range(N_GRID), key=lambda i: abs(GRID[i] - QUERY_X))
QUERY_Y = MU[Q_IDX]

# ---------------------------------------------------------------------------
# 3. Palette, fonts and geometry
# ---------------------------------------------------------------------------
NAVY = "#0B2545"
BLUE = "#2C7BE5"
VIOLET = "#A855F7"
L_MUTED = "#5B728C"
L_GRID = "#D8E6F5"
QUERY = "#E08A1E"
QUERY_TEXT = "#C2700D"
OBS_COLS = ["#0F7FC0", BLUE, "#7C5CD6", "#C056C8"]
FONT = "'DejaVu Sans','Source Han Sans SC','Noto Sans CJK SC',sans-serif"

# Main plot box: EXACTLY the same numbers as annotated-wave.svg.
AX, AY, AW, AH = 110, 150, 1030, 430
# Crop translation, applied once via <g transform>, so all path data is
# byte-identical to the full annotated diagram.
TX, TY = -56, -130
W, H = 1100, 500


def n(v):
    return ("%.2f" % v).rstrip("0").rstrip(".")


def xy(x, up):
    return (AX + x * AW, AY + (U_HI - up) / (U_HI - U_LO) * AH)


def segs(points):
    """Catmull-Rom -> cubic Bezier, for a smooth wave."""
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


def band(halflist):
    up = [xy(x, MU[i] + halflist[i]) for i, x in enumerate(GRID)]
    lo = [xy(x, MU[i] - halflist[i]) for i, x in enumerate(GRID)]
    d = (path_d(up) + " L %s,%s " % (n(lo[-1][0]), n(lo[-1][1]))
         + " ".join(segs(list(reversed(lo)))) + " Z")
    return d, up, lo


def defs():
    """Light-edition gradients, identical to defs('aw', 'light')."""
    return """<defs>
  <linearGradient id="curve-pp" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#0F7FC0"/>
    <stop offset="0.33" stop-color="#2C7BE5"/>
    <stop offset="0.68" stop-color="#7C5CD6"/>
    <stop offset="1" stop-color="#B049AE"/></linearGradient>
  <linearGradient id="band-pp" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#5FA8EA" stop-opacity="0"/>
    <stop offset="0.14" stop-color="#5FA8EA" stop-opacity="0.75"/>
    <stop offset="0.50" stop-color="#2C7BE5" stop-opacity="0.95"/>
    <stop offset="0.86" stop-color="#8E6BD8" stop-opacity="0.80"/>
    <stop offset="1" stop-color="#8E6BD8" stop-opacity="0"/></linearGradient>
  <linearGradient id="edge-pp" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#0F7FC0" stop-opacity="0"/>
    <stop offset="0.16" stop-color="#0F7FC0" stop-opacity="0.9"/>
    <stop offset="0.55" stop-color="#7C5CD6" stop-opacity="0.9"/>
    <stop offset="0.84" stop-color="#B049AE" stop-opacity="0.8"/>
    <stop offset="1" stop-color="#B049AE" stop-opacity="0"/></linearGradient>
</defs>"""


def build(show_guide=True):
    body = []

    # frame + grid
    body.append(f'<rect x="{AX}" y="{AY}" width="{AW}" height="{AH}" '
                f'fill="#FBFDFF" stroke="{L_GRID}"/>')
    for i in range(1, 8):
        xx = AX + i * AW / 8.0
        body.append(f'<line x1="{n(xx)}" y1="{AY}" x2="{n(xx)}" y2="{AY+AH}" '
                    f'stroke="{L_GRID}" stroke-width="1" opacity="0.65"/>')
    for j in range(1, 5):
        yy = AY + j * AH / 5.0
        body.append(f'<line x1="{AX}" y1="{n(yy)}" x2="{AX+AW}" y2="{n(yy)}" '
                    f'stroke="{L_GRID}" stroke-width="1" opacity="0.65"/>')

    # axis annotations
    body.append(f'<text x="{AX}" y="{AY+AH+30}" font-family="{FONT}" '
                f'font-size="14" fill="{L_MUTED}">candidate structure  x  →</text>')
    body.append(f'<text x="{AX+AW}" y="{AY+AH+30}" text-anchor="end" '
                f'font-family="{FONT}" font-size="14" fill="{L_MUTED}">'
                f'unexplored / high σ</text>')
    body.append(f'<text x="{AX-14}" y="{AY+16}" text-anchor="end" '
                f'font-family="{FONT}" font-size="14" fill="{L_MUTED}">μ, σ</text>')

    # three layered uncertainty bands
    halo, _, _ = band(HALF3)
    outer, up2, lo2 = band(HALF2)
    inner, up1, lo1 = band(HALF1)
    mean = path_d([xy(x, MU[i]) for i, x in enumerate(GRID)])
    body.append(f'<path d="{halo}" fill="url(#band-pp)" opacity="0.055"/>')
    body.append(f'<path d="{outer}" fill="url(#band-pp)" opacity="0.17"/>')
    body.append(f'<path d="{inner}" fill="url(#band-pp)" opacity="0.15"/>')
    for pts, dash, op in [(up2, 'stroke-dasharray="7 6"', 0.60),
                          (lo2, 'stroke-dasharray="7 6"', 0.60),
                          (up1, "", 0.38), (lo1, "", 0.38)]:
        body.append(f'<path d="{path_d(pts)}" fill="none" stroke="url(#edge-pp)" '
                    f'stroke-width="1.6" {dash} opacity="{n(op)}"/>')

    # posterior mean
    body.append(f'<path d="{mean}" fill="none" stroke="url(#curve-pp)" '
                f'stroke-width="9" stroke-linecap="round" opacity="0.15"/>')
    body.append(f'<path d="{mean}" fill="none" stroke="url(#curve-pp)" '
                f'stroke-width="4.6" stroke-linecap="round" stroke-linejoin="round"/>')

    # labelled observations
    for i, (x, y) in enumerate(zip(OBS_X, OBS_Y)):
        cx, cy = xy(x, y)
        col = OBS_COLS[i]
        body.append(f'<circle cx="{n(cx)}" cy="{n(cy)}" r="9" fill="#FFFFFF" '
                    f'stroke="{col}" stroke-width="3"/>')
        body.append(f'<text x="{n(cx)}" y="{n(cy-18)}" text-anchor="middle" '
                    f'font-family="{FONT}" font-size="12" fill="{col}">'
                    f'x{i+1}</text>')

    # next query point (ring + crosshair; optional dashed leader)
    qx, qy = xy(QUERY_X, QUERY_Y)
    if show_guide:
        body.append(f'<line x1="{n(qx)}" y1="{n(qy+18)}" x2="{n(qx)}" '
                    f'y2="{AY+AH}" stroke="{QUERY}" stroke-width="1.6" '
                    f'stroke-dasharray="5 6" opacity="0.7"/>')
    body.append(f'<circle cx="{n(qx)}" cy="{n(qy)}" r="19" fill="none" '
                f'stroke="{QUERY}" stroke-width="3"/>')
    body.append(f'<line x1="{n(qx-7)}" y1="{n(qy)}" x2="{n(qx+7)}" y2="{n(qy)}" '
                f'stroke="{QUERY}" stroke-width="2.4"/>')
    body.append(f'<line x1="{n(qx)}" y1="{n(qy-7)}" x2="{n(qx)}" y2="{n(qy+7)}" '
                f'stroke="{QUERY}" stroke-width="2.4"/>')
    body.append(f'<text x="{n(qx)}" y="{n(qy-30)}" text-anchor="middle" '
                f'font-family="{FONT}" font-size="14" font-weight="700" '
                f'fill="{QUERY_TEXT}">next query</text>')
    body.append(f'<text x="{n(qx)}" y="{n(qy-48)}" text-anchor="middle" '
                f'font-family="{FONT}" font-size="12" fill="{L_MUTED}">'
                f'high-σ region</text>')

    # vertical +/-1.96 sigma ruler in the left extrapolation arm
    rx = xy(0.035, 0)[0]
    yt = xy(0.035, MU[5] + HALF2[5])[1]
    yb = xy(0.035, MU[5] - HALF2[5])[1]
    body.append(f'<line x1="{n(rx)}" y1="{n(yt)}" x2="{n(rx)}" y2="{n(yb)}" '
                f'stroke="{VIOLET}" stroke-width="1.4"/>')
    for yy in (yt, yb):
        body.append(f'<line x1="{n(rx-7)}" y1="{n(yy)}" x2="{n(rx+7)}" '
                    f'y2="{n(yy)}" stroke="{VIOLET}" stroke-width="1.6"/>')
    body.append(f'<rect x="{n(rx+8)}" y="{n((yt+yb)/2-11)}" width="62" '
                f'height="19" rx="4" fill="#FFFFFF" opacity="0.90"/>')
    body.append(f'<text x="{n(rx+12)}" y="{n((yt+yb)/2+4)}" '
                f'font-family="{FONT}" font-size="13" fill="{VIOLET}">±1.96σ</text>')

    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}">\n'
            f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>\n'
            f'{defs()}\n'
            f'<g transform="translate({TX},{TY})">\n'
            + "\n".join(body) + '\n</g>\n</svg>\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-guide", action="store_true",
                    help="omit the dashed leader under the query ring")
    ap.add_argument("--scale", type=int, default=2,
                    help="PNG supersampling factor (default: 2)")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="output directory (default: this script's folder)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sp = os.path.join(args.outdir, "gp-posterior.svg")
    with open(sp, "w", encoding="utf-8") as fh:
        fh.write(build(show_guide=not args.no_guide))

    pp = os.path.join(args.outdir, "gp-posterior.png")
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-w", str(W * args.scale),
                        "-h", str(H * args.scale), "-o", pp, sp], check=True)
    else:
        print("NOTE: rsvg-convert not found - SVG written, PNG skipped.")

    extrema = [round(GRID[i], 3) for i in range(1, N_GRID - 1)
               if (MU[i] - MU[i - 1]) * (MU[i + 1] - MU[i]) < 0]
    print("GP posterior: sigma %.3f..%.3f | mu %.3f..%.3f | mu extrema %s"
          % (min(SIG), max(SIG), min(MU), max(MU), extrema))
    print("EI: max %.4f | left/right edge %.4f / %.4f | query x=%.2f"
          % (EI_MAX, EI[0], EI[-1], QUERY_X))
    print("wrote:", sp)
    if os.path.exists(pp):
        print("wrote:", pp)


if __name__ == "__main__":
    main()
