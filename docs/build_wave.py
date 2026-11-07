#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPEvo logo - Uncertainty / Wave edition.

Why this edition exists
-----------------------
The user's reference image shows a genuine S-shaped wave wrapped in translucent
ribbons that are WIDE at the ends and PINCHED in the middle. That is exactly a
Gaussian-process posterior: a wavy mean mu(x) plus a +/-1.96*sigma credible
band that narrows at labelled observations and opens in unexplored regions.

The first two editions used a monotonically RISING mu (evolution / best-so-far).
This edition recomputes the posterior from scratch so that:
  - mu(x) has real peaks and troughs (a wave, like the reference)
  - the uncertainty band is the visual protagonist (two layers: +/-1 sigma
    and +/-1.96 sigma, with visible boundaries)
  - active learning is explicit: a computed EI(x) acquisition curve and a
    "next query" ring placed in the high-sigma extrapolation region

Everything is computed with a real RBF-kernel GP posterior in pure Python:
  K = sf2 * exp(-(x-x')^2 / (2*l^2)) + sn2 * I
  mu(x)  = m0 + k(x)^T K^-1 (y - m0)
  var(x) = sf2 - k(x)^T K^-1 k(x)
  EI(x)  = (mu - f* - xi) Phi(Z) + sigma phi(Z),  Z = (mu - f* - xi)/sigma
Band half-widths are subjected to a *bounded visual compression*
  half(k,sigma) = HW * (1 - exp(-K_SIG * k * sigma))
so the 95% band stays inside the icon frame. This is a logo, not a data plot;
the compression is documented in the manifest and in the annotated artwork.

The standard/ and glow/ editions are NOT modified. This script is standalone
(it does not import build_logo.py, so it has zero side effects on old files).

Outputs (all in wave/): icon (dark + light), lockup (dark + light),
README banner, annotated scientific diagram, specification board, favicons,
manifest-wave.json.
"""

import math
import os
import json
import random
import hashlib

ROOT = "/home/feng/gpguided/outputs/gpevo-logo"
OUT = os.path.join(ROOT, "wave")
SVGD = os.path.join(OUT, "svg")
PNGD = os.path.join(OUT, "png")
os.makedirs(SVGD, exist_ok=True)
os.makedirs(PNGD, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Real GP posterior + Expected Improvement
# ---------------------------------------------------------------------------
OBS_X = [0.14, 0.34, 0.56, 0.80]
OBS_Y = [0.70, 0.30, 0.76, 0.44]        # up-down-up-down -> a wave
M0 = 0.52                               # constant prior mean
ELL, SF2, SN2 = 0.125, 1.0, 0.008
XI = 0.01                               # EI exploration parameter
F_STAR = max(OBS_Y)
HW, K_SIG = 0.38, 0.80                  # bounded visual band compression
N_GRID = 160
U_LO, U_HI = -0.05, 1.05               # value span mapped to the plot height
QUERY_X = 0.90                          # representative high-sigma query point

FONT_LATIN = "'DejaVu Sans','Source Han Sans SC','Noto Sans CJK SC',sans-serif"
FONT_CJK = "'Source Han Sans SC','Noto Sans CJK SC','DejaVu Sans',sans-serif"
FONT_MONO = "'Source Han Mono SC','DejaVu Sans Mono',monospace"

# Footer URL per horizontal-lockup mode. The Gitee URL is the one documented
# in the README / intro PPT. The GitHub URL was explicitly requested by the
# user for the LIGHT lockup only. The user confirmed on 2026-11-07 that the
# GitHub repository is PLANNED and does not exist yet, so its current 404 is
# expected: this is a forward-looking / reserved address, not a broken link.
# It must not be described as a live repository until it is actually created.
# See manifest url_audit.
SITE_URL = {
    "dark": "gitee.com/fenggo/gpevo",
    "light": "github.com/fenggo/gpevo",
}


def kern(a, b):
    return SF2 * math.exp(-((a - b) ** 2) / (2.0 * ELL ** 2))


def mat_inv(A):
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
    """Bounded, monotone compression of k*sigma into the logo's value space."""
    return HW * (1.0 - math.exp(-K_SIG * k * sigma))


HALF1 = [half(1.0, s) for s in SIG]        # +/-1 sigma  (inner band)
HALF2 = [half(1.96, s) for s in SIG]       # +/-1.96 sigma (95% band)
HALF3 = [half(2.45, s) for s in SIG]       # faint outer halo

EI_MAX = max(EI)
EXTREMA = [GRID[i] for i in range(1, N_GRID - 1)
           if (MU[i] - MU[i - 1]) * (MU[i + 1] - MU[i]) < 0]
Q_IDX = min(range(N_GRID), key=lambda i: abs(GRID[i] - QUERY_X))
QUERY_Y = MU[Q_IDX]

print("GP: sigma %.3f..%.3f | mu %.3f..%.3f | mu extrema at %s"
      % (min(SIG), max(SIG), min(MU), max(MU),
         [round(v, 3) for v in EXTREMA]))
print("95%% band: halfwidth %.3f..%.3f | extent %.3f..%.3f (frame %.2f..%.2f)"
      % (min(HALF2), max(HALF2),
         min(m - h for m, h in zip(MU, HALF2)),
         max(m + h for m, h in zip(MU, HALF2)), U_LO, U_HI))
print("EI: max %.4f | at left/right edge %.4f / %.4f | query x=%.2f mu=%.3f sigma=%.3f"
      % (EI_MAX, EI[0], EI[-1], QUERY_X, QUERY_Y, SIG[Q_IDX]))

# ---------------------------------------------------------------------------
# 2. Palettes
# ---------------------------------------------------------------------------
CY, BL, VI, MG = "#22D3EE", "#4F8CFF", "#A855F7", "#E879F9"
WH, AMBER = "#FFFFFF", "#F5B455"
SPACE, DEEP, INK = "#050914", "#0A1330", "#0B1836"
TXT, MUTED = "#DCE8FF", "#8FA6C9"
NAVY, BLUE, PAPER = "#0B2545", "#2C7BE5", "#F7FAFD"
L_MUTED, L_GRID = "#5B728C", "#D8E6F5"

PAL = {
    "dark": dict(
        bg1="#152652", bg2=DEEP, bg3=SPACE, page=SPACE,
        frame=DEEP, frame_stroke="#1E3057",
        c1=CY, c2=BL, c3=VI, c4=MG,
        band_c1=CY, band_c2=VI, band_c3=MG,
        band_op=0.20, band1_op=0.20, halo_op=0.065,
        edge_op=0.46, edge1_op=0.24,
        grid_col="#5C86D8", grid_op=0.065,
        obs_core=WH, obs_cols=[CY, BL, VI, MG],
        query=AMBER, query_text=AMBER,
        glow=True, particles=True, aurora=True,
        word_gp="url(#text-{uid})", word_evo=WH,
        tag=TXT, sub=CY, url=MUTED, rule="#25406E",
    ),
    "light": dict(
        bg1="#FFFFFF", bg2="#F8FBFF", bg3="#F1F7FD", page="#FFFFFF",
        frame="#FFFFFF", frame_stroke="#DCE8F4",
        c1="#0F7FC0", c2=BLUE, c3="#7C5CD6", c4="#B049AE",
        band_c1="#5FA8EA", band_c2="#2C7BE5", band_c3="#8E6BD8",
        band_op=0.17, band1_op=0.16, halo_op=0.05,
        edge_op=0.55, edge1_op=0.38,
        grid_col="#E6F0FA", grid_op=0.55,
        obs_core="#FFFFFF", obs_cols=["#0F7FC0", BLUE, "#7C5CD6", "#C056C8"],
        query="#E08A1E", query_text="#C2700D",
        glow=False, particles=False, aurora=False,
        word_gp=BLUE, word_evo=NAVY,
        tag=NAVY, sub=BLUE, url=L_MUTED, rule="#D7E4F2",
    ),
}

# icon geometry (base 512)
BOX = (24, 24, 464, 464)
PLOT = (58, 104, 396, 312)


def n(v):
    return ("%.2f" % v).rstrip("0").rstrip(".")


def to_xy(x, up):
    return (PLOT[0] + x * PLOT[2],
            PLOT[1] + (U_HI - up) / (U_HI - U_LO) * PLOT[3])


def segs(points):
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


def band(kind):
    h = {"halo": HALF3, "outer": HALF2, "inner": HALF1}[kind]
    up = [to_xy(x, MU[i] + h[i]) for i, x in enumerate(GRID)]
    lo = [to_xy(x, MU[i] - h[i]) for i, x in enumerate(GRID)]
    d = path_d(up) + " L %s,%s " % (n(lo[-1][0]), n(lo[-1][1]))
    d += " ".join(segs(list(reversed(lo)))) + " Z"
    return d, up, lo


B_HALO, UP3, LO3 = band("halo")
B_OUT, UP2, LO2 = band("outer")
B_IN, UP1, LO1 = band("inner")
MEAN_D = path_d([to_xy(x, MU[i]) for i, x in enumerate(GRID)])

# ---------------------------------------------------------------------------
# 3. Reusable SVG pieces
# ---------------------------------------------------------------------------
rng = random.Random(20261107)
PARTICLES = []
for _ in range(34):
    x = rng.uniform(0.04, 0.98)
    i = min(range(N_GRID), key=lambda k: abs(GRID[k] - x))
    up = MU[i] + rng.gauss(0.0, 0.42) * HALF2[i]
    cx, cy = to_xy(x, up)
    PARTICLES.append((cx, cy, rng.uniform(0.9, 2.5),
                      rng.choice([CY, BL, VI, MG, WH, "#7DD3FC"]),
                      round(rng.uniform(0.22, 0.62), 2)))


def particles_svg():
    s = []
    for cx, cy, r, col, op in PARTICLES:
        if r > 2.1:
            s.append(f'<circle cx="{n(cx)}" cy="{n(cy)}" r="{n(r*3.1)}" '
                     f'fill="{col}" opacity="{n(op*0.10)}"/>')
        s.append(f'<circle cx="{n(cx)}" cy="{n(cy)}" r="{n(r)}" '
                 f'fill="{col}" opacity="{n(op)}"/>')
    return "\n".join(s)


def grid_svg(p, x0=24, y0=24, w=464, h=464, step=48):
    s = []
    for i in range(int(w // step) + 2):
        xx = x0 + i * step
        if xx <= x0 + w:
            s.append(f'<line x1="{n(xx)}" y1="{n(y0)}" x2="{n(xx)}" '
                     f'y2="{n(y0+h)}" stroke="{p["grid_col"]}" '
                     f'stroke-width="1" opacity="{n(p["grid_op"])}"/>')
    for j in range(int(h // step) + 2):
        yy = y0 + j * step
        if yy <= y0 + h:
            s.append(f'<line x1="{n(x0)}" y1="{n(yy)}" x2="{n(x0+w)}" '
                     f'y2="{n(yy)}" stroke="{p["grid_col"]}" '
                     f'stroke-width="1" opacity="{n(p["grid_op"])}"/>')
    return "\n".join(s)


def aurora_svg(uid, y_base=88, amp=16, w=512):
    """Very faint aurora, deliberately kept behind the uncertainty band."""
    out = []
    for nm, wid, op, ph in [("aur1", 46, 0.045, 0.0),
                            ("aur2", 34, 0.050, 1.2),
                            ("aur3", 24, 0.040, 2.4)]:
        pts = []
        for i in range(49):
            t = i / 48.0
            xx = 24 + t * (w - 48)
            yy = y_base + amp * (0.6 * math.sin(t * 5.2 + ph)
                                 + 0.4 * math.sin(t * 10.0 + ph * 1.6))
            pts.append((xx, yy))
        out.append(f'<path d="{path_d(pts)}" fill="none" '
                   f'stroke="url(#{nm}-{uid})" stroke-width="{n(wid)}" '
                   f'opacity="{n(op)}" stroke-linecap="round"/>')
    return "\n".join(out)


def defs(uid, mode, w=512, h=512):
    p = PAL[mode]
    s = ['<defs>']
    s.append(f'''<radialGradient id="bg-{uid}" cx="50%" cy="32%" r="78%">
      <stop offset="0" stop-color="{p['bg1']}"/>
      <stop offset="0.55" stop-color="{p['bg2']}"/>
      <stop offset="1" stop-color="{p['bg3']}"/></radialGradient>''')
    s.append(f'''<linearGradient id="curve-{uid}" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{p['c1']}"/>
      <stop offset="0.33" stop-color="{p['c2']}"/>
      <stop offset="0.68" stop-color="{p['c3']}"/>
      <stop offset="1" stop-color="{p['c4']}"/></linearGradient>''')
    # Soft-ended variant for the MARK: like the reference image, the wave
    # dissolves at both ends instead of ending in a blunt stroke.
    s.append(f'''<linearGradient id="soft-{uid}" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{p['c1']}" stop-opacity="0"/>
      <stop offset="0.08" stop-color="{p['c1']}" stop-opacity="1"/>
      <stop offset="0.33" stop-color="{p['c2']}" stop-opacity="1"/>
      <stop offset="0.68" stop-color="{p['c3']}" stop-opacity="1"/>
      <stop offset="0.92" stop-color="{p['c4']}" stop-opacity="1"/>
      <stop offset="1" stop-color="{p['c4']}" stop-opacity="0"/></linearGradient>''')
    s.append(f'''<linearGradient id="core-{uid}" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{WH}" stop-opacity="0"/>
      <stop offset="0.08" stop-color="{WH}" stop-opacity="0.70"/>
      <stop offset="0.92" stop-color="{WH}" stop-opacity="0.70"/>
      <stop offset="1" stop-color="{WH}" stop-opacity="0"/></linearGradient>''')
    # band gradients fade at BOTH ends so the ribbon never shows a hard vertical edge
    s.append(f'''<linearGradient id="band-{uid}" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{p['band_c1']}" stop-opacity="0"/>
      <stop offset="0.14" stop-color="{p['band_c1']}" stop-opacity="0.75"/>
      <stop offset="0.50" stop-color="{p['band_c2']}" stop-opacity="0.95"/>
      <stop offset="0.86" stop-color="{p['band_c3']}" stop-opacity="0.80"/>
      <stop offset="1" stop-color="{p['band_c3']}" stop-opacity="0"/></linearGradient>''')
    s.append(f'''<linearGradient id="edge-{uid}" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{p['c1']}" stop-opacity="0"/>
      <stop offset="0.16" stop-color="{p['c1']}" stop-opacity="0.9"/>
      <stop offset="0.55" stop-color="{p['c3']}" stop-opacity="0.9"/>
      <stop offset="0.84" stop-color="{p['c4']}" stop-opacity="0.8"/>
      <stop offset="1" stop-color="{p['c4']}" stop-opacity="0"/></linearGradient>''')
    s.append(f'''<linearGradient id="text-{uid}" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{CY}"/>
      <stop offset="0.5" stop-color="#8B9CFF"/>
      <stop offset="1" stop-color="{MG}"/></linearGradient>''')
    for nm, a, b in [("aur1", CY, VI), ("aur2", VI, MG), ("aur3", MG, CY)]:
        s.append(f'''<linearGradient id="{nm}-{uid}" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stop-color="{a}" stop-opacity="0"/>
          <stop offset="0.25" stop-color="{a}" stop-opacity="0.9"/>
          <stop offset="0.72" stop-color="{b}" stop-opacity="0.9"/>
          <stop offset="1" stop-color="{b}" stop-opacity="0"/></linearGradient>''')
    s.append(f'<clipPath id="clip-{uid}"><rect x="24" y="24" width="464" '
             f'height="464" rx="112"/></clipPath>')
    s.append('</defs>')
    return "\n".join(s)


def mark_body(uid, mode="dark", query=True, aurora=True, guide=True):
    """The 512x512 mark. Same geometry in dark and light; only the skin changes.

    guide=False removes the dashed vertical leader under the query ring.
    The standalone icons ship without it (a logo should not have a dangling
    axis line); the banner/lockup/board thumbnails and the annotated diagram
    keep it as a diagrammatic guide.
    """
    p = PAL[mode]
    gp = p["word_gp"].format(uid=uid) if "{uid}" in p["word_gp"] else p["word_gp"]
    s = [f'<rect x="24" y="24" width="464" height="464" rx="112" '
         f'fill="url(#bg-{uid})"/>']
    s.append(f'<g clip-path="url(#clip-{uid})">')
    s.append(grid_svg(p))
    if mode == "dark" and aurora and p["aurora"]:
        s.append(aurora_svg(uid))
    if p["particles"]:
        s.append(particles_svg())

    # --- the uncertainty band is the protagonist ---
    s.append(f'<path d="{B_HALO}" fill="url(#band-{uid})" '
             f'opacity="{n(p["halo_op"])}"/>')
    s.append(f'<path d="{B_OUT}" fill="url(#band-{uid})" '
             f'opacity="{n(p["band_op"])}"/>')
    s.append(f'<path d="{B_IN}" fill="url(#band-{uid})" '
             f'opacity="{n(p["band1_op"])}"/>')
    # visible boundaries so the two sigma levels are readable
    for pts, dash, op, wid in [
            (UP2, 'stroke-dasharray="7 7"', p["edge_op"], 1.8),
            (LO2, 'stroke-dasharray="7 7"', p["edge_op"], 1.8),
            (UP1, "", p["edge1_op"], 1.4),
            (LO1, "", p["edge1_op"], 1.4)]:
        s.append(f'<path d="{path_d(pts)}" fill="none" '
                 f'stroke="url(#edge-{uid})" stroke-width="{n(wid)}" '
                 f'{dash} opacity="{n(op)}"/>')

    # --- posterior mean: luminous wave (dark) or crisp line (light) ---
    if p["glow"]:
        for wid, op in [(38, 0.045), (26, 0.09), (17, 0.20)]:
            s.append(f'<path d="{MEAN_D}" fill="none" '
                     f'stroke="url(#soft-{uid})" stroke-width="{wid}" '
                     f'stroke-linecap="round" opacity="{n(op)}"/>')
        s.append(f'<path d="{MEAN_D}" fill="none" stroke="url(#soft-{uid})" '
                 f'stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>')
        s.append(f'<path d="{MEAN_D}" fill="none" stroke="url(#core-{uid})" '
                 f'stroke-width="2.2" stroke-linecap="round"/>')
    else:
        s.append(f'<path d="{MEAN_D}" fill="none" stroke="url(#soft-{uid})" '
                 f'stroke-width="8" stroke-linecap="round" opacity="0.16"/>')
        s.append(f'<path d="{MEAN_D}" fill="none" stroke="url(#soft-{uid})" '
                 f'stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')

    # --- labelled observations (DFT / SIESTA) ---
    for i, (x, y) in enumerate(zip(OBS_X, OBS_Y)):
        cx, cy = to_xy(x, y)
        col = p["obs_cols"][i]
        if p["glow"]:
            s.append(f'<circle cx="{n(cx)}" cy="{n(cy)}" r="25" fill="{col}" '
                     f'opacity="0.09"/>')
            s.append(f'<circle cx="{n(cx)}" cy="{n(cy)}" r="15" fill="{col}" '
                     f'opacity="0.20"/>')
        s.append(f'<circle cx="{n(cx)}" cy="{n(cy)}" r="8.5" '
                 f'fill="{p["obs_core"]}" stroke="{col}" stroke-width="3.4"/>')

    # --- active learning: next query in the high-sigma region ---
    if query:
        qx, qy = to_xy(QUERY_X, QUERY_Y)
        if guide:
            base_y = PLOT[1] + PLOT[3]
            s.append(f'<line x1="{n(qx)}" y1="{n(qy+16)}" x2="{n(qx)}" '
                     f'y2="{n(base_y)}" stroke="{p["query"]}" '
                     f'stroke-width="1.6" stroke-dasharray="5 6" opacity="0.55"/>')
        s.append(f'<circle cx="{n(qx)}" cy="{n(qy)}" r="29" fill="{p["query"]}" '
                 f'opacity="0.09"/>')
        s.append(f'<circle cx="{n(qx)}" cy="{n(qy)}" r="17" fill="none" '
                 f'stroke="{p["query"]}" stroke-width="3" opacity="0.95"/>')
        s.append(f'<line x1="{n(qx-6)}" y1="{n(qy)}" x2="{n(qx+6)}" '
                 f'y2="{n(qy)}" stroke="{p["query"]}" stroke-width="2.4" '
                 f'stroke-linecap="round"/>')
        s.append(f'<line x1="{n(qx)}" y1="{n(qy-6)}" x2="{n(qx)}" '
                 f'y2="{n(qy+6)}" stroke="{p["query"]}" stroke-width="2.4" '
                 f'stroke-linecap="round"/>')
    s.append('</g>')
    return "\n".join(s)


def doc(w, h, body, defs_s=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">\n{defs_s}\n{body}\n</svg>\n')


def wordmark(x, y, size, uid, mode):
    p = PAL[mode]
    gp = p["word_gp"].format(uid=uid) if "{uid}" in p["word_gp"] else p["word_gp"]
    return (f'<text x="{n(x)}" y="{n(y)}" font-family="{FONT_LATIN}" '
            f'font-size="{n(size)}" font-weight="700" letter-spacing="1">'
            f'<tspan fill="{gp}">GP</tspan>'
            f'<tspan fill="{p["word_evo"]}">Evo</tspan></text>')


def taglines(x, y, size, mode):
    p = PAL[mode]
    return "\n".join([
        f'<text x="{n(x)}" y="{n(y)}" font-family="{FONT_LATIN}" '
        f'font-size="{n(size)}" font-weight="500" letter-spacing="3.2" '
        f'fill="{p["tag"]}">GP-GUIDED CRYSTAL STRUCTURE PREDICTION</text>',
        f'<text x="{n(x+2)}" y="{n(y+30)}" font-family="{FONT_LATIN}" '
        f'font-size="{n(size*0.80)}" font-weight="600" letter-spacing="2.4" '
        f'fill="{p["sub"]}">POSTERIOR UNCERTAINTY  ·  ACTIVE LEARNING</text>',
    ])


def save(name, content, w, h, scale=2):
    sp = os.path.join(SVGD, name + ".svg")
    with open(sp, "w", encoding="utf-8") as fh:
        fh.write(content)
    pp = os.path.join(PNGD, name + ".png")
    os.system('rsvg-convert -w %d -h %d -o "%s" "%s"'
              % (int(w * scale), int(h * scale), pp, sp))
    return sp


made = []

# --- 1. icons (dark + light) -----------------------------------------------
for mode, nm in [("dark", "icon-wave"), ("light", "icon-wave-light")]:
    uid = "iw" if mode == "dark" else "il"
    made.append(save(nm,
                     doc(512, 512,
                         mark_body(uid, mode, guide=False), defs(uid, mode)),
                     512, 512, 2))

# --- 2. horizontal lockups -------------------------------------------------
for mode, nm in [("dark", "lockup-wave"), ("light", "lockup-wave-light")]:
    uid = "lw" if mode == "dark" else "ll"
    p = PAL[mode]
    b = [f'<rect width="1240" height="340" fill="{p["page"]}"/>']
    if mode == "dark":
        b.append(f'<rect width="1240" height="340" fill="url(#bg-{uid})" '
                 f'opacity="0.55"/>')
        b.append(f'<g opacity="0.30">{grid_svg(p, 0, 0, 1240, 340, 48)}</g>')
    # Per user request the light lockup has no dangling leader under the
    # query ring; the dark lockup keeps it as a diagrammatic guide.
    show_guide = (mode != "light")
    b.append(f'<g transform="translate(34,30) scale(0.55)">'
             f'{mark_body(uid, mode, guide=show_guide)}</g>')
    b.append(wordmark(348, 164, 100, uid, mode))
    b.append(taglines(353, 210, 18, mode))
    b.append(f'<rect x="353" y="266" width="830" height="1.4" '
             f'fill="{p["rule"]}"/>')
    b.append(f'<text x="353" y="298" font-family="{FONT_MONO}" font-size="16" '
             f'letter-spacing="1.3" fill="{p["url"]}">{SITE_URL[mode]}</text>')
    made.append(save(nm, doc(1240, 340, "\n".join(b), defs(uid, mode)),
                     1240, 340, 2))

# --- 3. README banner with a compact legend --------------------------------
uid = "bw"
p = PAL["dark"]
b = [f'<rect width="1600" height="420" fill="{SPACE}"/>',
     f'<rect width="1600" height="420" fill="url(#bg-{uid})" opacity="0.70"/>',
     f'<g opacity="0.28">{grid_svg(p, 0, 0, 1600, 420, 60)}</g>',
     f'<rect x="0" y="0" width="1600" height="5" fill="url(#curve-{uid})"/>',
     f'<g transform="translate(46,46) scale(0.66)">{mark_body(uid, "dark")}</g>',
     wordmark(420, 186, 90, uid, "dark"),
     taglines(426, 234, 17, "dark"),
     f'<rect x="426" y="292" width="1120" height="1.4" fill="{p["rule"]}"/>',
     f'<text x="426" y="330" font-family="{FONT_MONO}" font-size="15" '
     f'letter-spacing="1.2" fill="{p["url"]}">gitee.com/fenggo/gpevo</text>',
     f'<text x="1546" y="330" font-family="{FONT_MONO}" font-size="15" '
     f'letter-spacing="1.2" text-anchor="end" fill="{p["url"]}">'
     f'MIT  ·  Python 3.9+  ·  no MATLAB</text>']
# compact legend on the right
b.append(f'<text x="1080" y="128" font-family="{FONT_CJK}" font-size="14" '
         f'letter-spacing="1.6" fill="{MUTED}">GAUSSIAN-PROCESS POSTERIOR</text>')
legend = [
    ("line", "μ(x)   posterior mean", CY),
    ("band", "±1.96σ   uncertainty", VI),
    ("ring", "next query   active learning", AMBER),
]
for i, (kind, label, col) in enumerate(legend):
    yy = 158 + i * 42
    if kind == "line":
        b.append(f'<line x1="1080" y1="{yy}" x2="1136" y2="{yy}" '
                 f'stroke="{CY}" stroke-width="4" stroke-linecap="round"/>')
    elif kind == "band":
        b.append(f'<rect x="1080" y="{yy-9}" width="56" height="18" rx="4" '
                 f'fill="{VI}" opacity="0.35"/>')
        b.append(f'<line x1="1080" y1="{yy-9}" x2="1136" y2="{yy-9}" '
                 f'stroke="{VI}" stroke-width="1.4" stroke-dasharray="5 4"/>')
        b.append(f'<line x1="1080" y1="{yy+9}" x2="1136" y2="{yy+9}" '
                 f'stroke="{VI}" stroke-width="1.4" stroke-dasharray="5 4"/>')
    else:
        b.append(f'<circle cx="1108" cy="{yy}" r="11" fill="none" '
                 f'stroke="{AMBER}" stroke-width="2.6"/>')
        b.append(f'<line x1="1102" y1="{yy}" x2="1114" y2="{yy}" '
                 f'stroke="{AMBER}" stroke-width="2"/>')
        b.append(f'<line x1="1108" y1="{yy-6}" x2="1108" y2="{yy+6}" '
                 f'stroke="{AMBER}" stroke-width="2"/>')
    b.append(f'<text x="1152" y="{yy+6}" font-family="{FONT_LATIN}" '
             f'font-size="16" fill="{TXT}">{label}</text>')
made.append(save("banner-wave", doc(1600, 420, "\n".join(b), defs(uid, "dark")),
                 1600, 420, 2))

# --- 4. annotated scientific diagram (light, for papers / slides) ----------
uid = "aw"
p = PAL["light"]
W, H = 1600, 1020
AX, AY, AW, AH = 110, 150, 1030, 430       # main plot
EX, EY, EW, EH = 110, 670, 1030, 180       # EI panel
b = [f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
     f'<text x="64" y="62" font-family="{FONT_LATIN}" font-size="36" '
     f'font-weight="700" fill="{NAVY}">GPEvo</text>',
     f'<text x="226" y="62" font-family="{FONT_LATIN}" font-size="24" '
     f'fill="{L_MUTED}">/ Gaussian-process posterior &amp; active learning</text>',
     f'<text x="64" y="98" font-family="{FONT_CJK}" font-size="16" '
     f'fill="{L_MUTED}">波浪形后验均值 · 分层不确定度带 · EI 采集函数决定下一个待标注结构</text>',
     f'<rect x="0" y="0" width="1600" height="6" fill="url(#curve-{uid})"/>']


def a_xy(x, up):
    return (AX + x * AW, AY + (U_HI - up) / (U_HI - U_LO) * AH)


def a_band(halflist):
    up = [a_xy(x, MU[i] + halflist[i]) for i, x in enumerate(GRID)]
    lo = [a_xy(x, MU[i] - halflist[i]) for i, x in enumerate(GRID)]
    return (path_d(up) + " L %s,%s " % (n(lo[-1][0]), n(lo[-1][1]))
            + " ".join(segs(list(reversed(lo)))) + " Z", up, lo)


# axes
b.append(f'<rect x="{AX}" y="{AY}" width="{AW}" height="{AH}" fill="#FBFDFF" '
         f'stroke="{L_GRID}"/>')
for i in range(1, 8):
    xx = AX + i * AW / 8.0
    b.append(f'<line x1="{n(xx)}" y1="{AY}" x2="{n(xx)}" y2="{AY+AH}" '
             f'stroke="{L_GRID}" stroke-width="1" opacity="0.65"/>')
for j in range(1, 5):
    yy = AY + j * AH / 5.0
    b.append(f'<line x1="{AX}" y1="{n(yy)}" x2="{AX+AW}" y2="{n(yy)}" '
             f'stroke="{L_GRID}" stroke-width="1" opacity="0.65"/>')
b.append(f'<text x="{AX}" y="{AY+AH+30}" font-family="{FONT_LATIN}" '
         f'font-size="14" fill="{L_MUTED}">candidate structure  x  →</text>')
b.append(f'<text x="{AX+AW}" y="{AY+AH+30}" text-anchor="end" '
         f'font-family="{FONT_LATIN}" font-size="14" fill="{L_MUTED}">'
         f'unexplored / high σ</text>')
b.append(f'<text x="{AX-14}" y="{AY+16}" text-anchor="end" '
         f'font-family="{FONT_LATIN}" font-size="14" fill="{L_MUTED}">μ, σ</text>')

AB_HALO, _, _ = a_band(HALF3)
AB_OUT, AU2, AL2 = a_band(HALF2)
AB_IN, AU1, AL1 = a_band(HALF1)
AMEAN = path_d([a_xy(x, MU[i]) for i, x in enumerate(GRID)])
b.append(f'<path d="{AB_HALO}" fill="url(#band-{uid})" opacity="0.055"/>')
b.append(f'<path d="{AB_OUT}" fill="url(#band-{uid})" opacity="0.17"/>')
b.append(f'<path d="{AB_IN}" fill="url(#band-{uid})" opacity="0.15"/>')
for pts, dash, op in [(AU2, 'stroke-dasharray="7 6"', 0.60),
                      (AL2, 'stroke-dasharray="7 6"', 0.60),
                      (AU1, "", 0.38), (AL1, "", 0.38)]:
    b.append(f'<path d="{path_d(pts)}" fill="none" stroke="url(#edge-{uid})" '
             f'stroke-width="1.6" {dash} opacity="{n(op)}"/>')
b.append(f'<path d="{AMEAN}" fill="none" stroke="url(#curve-{uid})" '
         f'stroke-width="9" stroke-linecap="round" opacity="0.15"/>')
b.append(f'<path d="{AMEAN}" fill="none" stroke="url(#curve-{uid})" '
         f'stroke-width="4.6" stroke-linecap="round" stroke-linejoin="round"/>')
for i, (x, y) in enumerate(zip(OBS_X, OBS_Y)):
    cx, cy = a_xy(x, y)
    col = p["obs_cols"][i]
    b.append(f'<circle cx="{n(cx)}" cy="{n(cy)}" r="9" fill="#FFFFFF" '
             f'stroke="{col}" stroke-width="3"/>')
    b.append(f'<text x="{n(cx)}" y="{n(cy-18)}" text-anchor="middle" '
             f'font-family="{FONT_LATIN}" font-size="12" fill="{col}">'
             f'x{i+1}</text>')
# query ring + guide
qx, qy = a_xy(QUERY_X, QUERY_Y)
b.append(f'<line x1="{n(qx)}" y1="{n(qy+18)}" x2="{n(qx)}" y2="{AY+AH}" '
         f'stroke="{p["query"]}" stroke-width="1.6" stroke-dasharray="5 6" '
         f'opacity="0.7"/>')
b.append(f'<circle cx="{n(qx)}" cy="{n(qy)}" r="19" fill="none" '
         f'stroke="{p["query"]}" stroke-width="3"/>')
b.append(f'<line x1="{n(qx-7)}" y1="{n(qy)}" x2="{n(qx+7)}" y2="{n(qy)}" '
         f'stroke="{p["query"]}" stroke-width="2.4"/>')
b.append(f'<line x1="{n(qx)}" y1="{n(qy-7)}" x2="{n(qx)}" y2="{n(qy+7)}" '
         f'stroke="{p["query"]}" stroke-width="2.4"/>')
b.append(f'<text x="{n(qx)}" y="{n(qy-30)}" text-anchor="middle" '
         f'font-family="{FONT_LATIN}" font-size="14" font-weight="700" '
         f'fill="{p["query_text"]}">next query</text>')
b.append(f'<text x="{n(qx)}" y="{n(qy-48)}" text-anchor="middle" '
         f'font-family="{FONT_LATIN}" font-size="12" fill="{L_MUTED}">'
         f'high-σ region</text>')
# vertical 1.96 sigma ruler at the left extrapolation arm
rx = a_xy(0.035, 0)[0]
yt = a_xy(0.035, MU[5] + HALF2[5])[1]
yb = a_xy(0.035, MU[5] - HALF2[5])[1]
b.append(f'<line x1="{n(rx)}" y1="{n(yt)}" x2="{n(rx)}" y2="{n(yb)}" '
         f'stroke="{VI}" stroke-width="1.4"/>')
for yy in (yt, yb):
    b.append(f'<line x1="{n(rx-7)}" y1="{n(yy)}" x2="{n(rx+7)}" '
             f'y2="{n(yy)}" stroke="{VI}" stroke-width="1.6"/>')
b.append(f'<rect x="{n(rx+8)}" y="{n((yt+yb)/2-11)}" width="62" '
         f'height="19" rx="4" fill="#FFFFFF" opacity="0.90"/>')
b.append(f'<text x="{n(rx+12)}" y="{n((yt+yb)/2+4)}" '
         f'font-family="{FONT_LATIN}" font-size="13" fill="{VI}">±1.96σ</text>')

# right-hand legend
b.append(f'<text x="1200" y="164" font-family="{FONT_CJK}" font-size="15" '
         f'font-weight="700" letter-spacing="1.4" fill="{NAVY}">图形语义 / LEGEND</text>')
items = [
    ("line", "μ(x) 后验均值", "GP posterior mean"),
    ("dash", "μ ± 1.96σ 95% 置信带", "95% credible band"),
    ("inner", "μ ± 1σ 内层带", "inner 1σ band"),
    ("dot", "已标注样本 (DFT/SIESTA)", "labelled structures"),
    ("ring", "下一个查询点", "next query · high σ"),
]
for i, (kind, cn, en) in enumerate(items):
    yy = 200 + i * 66
    b.append(f'<rect x="1196" y="{yy-22}" width="344" height="56" rx="9" '
             f'fill="#F7FAFD" stroke="#E4EEF9"/>')
    if kind == "line":
        b.append(f'<line x1="1208" y1="{yy+4}" x2="1264" y2="{yy+4}" '
                 f'stroke="{BLUE}" stroke-width="4" stroke-linecap="round"/>')
    elif kind == "dash":
        b.append(f'<rect x="1208" y="{yy-7}" width="56" height="20" rx="3" '
                 f'fill="{BLUE}" opacity="0.22"/>')
        b.append(f'<line x1="1208" y1="{yy-7}" x2="1264" y2="{yy-7}" '
                 f'stroke="{VI}" stroke-width="1.6" stroke-dasharray="6 4"/>')
        b.append(f'<line x1="1208" y1="{yy+13}" x2="1264" y2="{yy+13}" '
                 f'stroke="{VI}" stroke-width="1.6" stroke-dasharray="6 4"/>')
    elif kind == "inner":
        b.append(f'<rect x="1208" y="{yy-7}" width="56" height="20" rx="3" '
                 f'fill="{BLUE}" opacity="0.32"/>')
    elif kind == "dot":
        b.append(f'<circle cx="1236" cy="{yy+4}" r="8" fill="#FFFFFF" '
                 f'stroke="#0F7FC0" stroke-width="3"/>')
    else:
        b.append(f'<circle cx="1236" cy="{yy+4}" r="11" fill="none" '
                 f'stroke="#E08A1E" stroke-width="2.6"/>')
        b.append(f'<line x1="1230" y1="{yy+4}" x2="1242" y2="{yy+4}" '
                 f'stroke="#E08A1E" stroke-width="2"/>')
    b.append(f'<text x="1282" y="{yy}" font-family="{FONT_CJK}" font-size="14" '
             f'font-weight="700" fill="{NAVY}">{cn}</text>')
    b.append(f'<text x="1282" y="{yy+20}" font-family="{FONT_LATIN}" '
             f'font-size="12" fill="{L_MUTED}">{en}</text>')

# EI panel
b.append(f'<text x="{AX}" y="{EY-14}" font-family="{FONT_LATIN}" '
         f'font-size="16" font-weight="700" fill="{NAVY}">'
         f'EI(x)  ·  Expected Improvement acquisition</text>')
b.append(f'<rect x="{EX}" y="{EY}" width="{EW}" height="{EH}" fill="#FBFDFF" '
         f'stroke="{L_GRID}"/>')
ep = []
for i, x in enumerate(GRID):
    px = EX + x * EW
    py = EY + EH - (EI[i] / EI_MAX) * (EH - 22) - 10
    ep.append((px, py))
ed = path_d(ep) + f' L {n(EX+EW)},{n(EY+EH)} L {n(EX)},{n(EY+EH)} Z'
b.append(f'<path d="{ed}" fill="{VI}" opacity="0.13"/>')
b.append(f'<path d="{path_d(ep)}" fill="none" stroke="{VI}" stroke-width="3" '
         f'stroke-linejoin="round"/>')
for frac, lab in [(0.0, "left arm"), (0.5, "exploited"), (1.0, "right arm")]:
    px = EX + frac * EW
    b.append(f'<line x1="{n(px)}" y1="{EY}" x2="{n(px)}" y2="{EY+EH}" '
             f'stroke="{L_GRID}" stroke-width="1"/>')
b.append(f'<line x1="{EX+QUERY_X*EW}" y1="{EY}" x2="{EX+QUERY_X*EW}" '
         f'y2="{EY+EH}" stroke="{p["query_text"]}" stroke-width="1.6" '
         f'stroke-dasharray="5 5"/>')
b.append(f'<text x="{EX+QUERY_X*EW-8}" y="{EY+EH-8}" text-anchor="end" '
         f'font-family="{FONT_LATIN}" font-size="12" '
         f'fill="{p["query_text"]}">query</text>')
b.append(f'<text x="{EX+EW}" y="{EY+EH+28}" text-anchor="end" '
         f'font-family="{FONT_LATIN}" font-size="13" fill="{L_MUTED}">'
         f'EI rises again where σ is large  →  exploration</text>')
# right note block
b.append(f'<rect x="1196" y="{EY}" width="344" height="{EH}" rx="10" '
         f'fill="#F7FAFD" stroke="#E4EEF9"/>')
notes = [
    "EI 由同一 GP 后验真实计算：",
    "EI=(μ−f*−ξ)Φ(Z)+σφ(Z)",
    "",
    "两侧外推区 σ 最大，EI 随之抬升，",
    "因此主动学习会优先去「不确定」",
    "的区域补一个新标注样本。",
    "",
    "图中环标出右侧高 σ 区的一个",
    "代表性候选查询位置 x=0.90。",
]
for i, t in enumerate(notes):
    b.append(f'<text x="1214" y="{EY+30+i*19}" font-family="{FONT_CJK}" '
             f'font-size="13" fill="{NAVY if i<2 else L_MUTED}">{t}</text>')

# footer
b.append(f'<text x="64" y="916" font-family="{FONT_CJK}" font-size="13" '
         f'fill="{L_MUTED}">计算依据：RBF 核一维 GP 后验（ℓ={ELL}, σf²={SF2}, '
         f'σn²={SN2}, 常数先验均值 m={M0}, 4 个已标注样本, ξ={XI}）；'
         f'±1σ / ±1.96σ 带宽做了有界视觉压缩以保证标志可读。</text>')
b.append(f'<text x="64" y="942" font-family="{FONT_CJK}" font-size="13" '
         f'fill="{L_MUTED}">严格的 argmax EI 落在搜索域两端的外推边界；'
         f'图中标注的查询点是高 σ 候选区的代表位置。主动学习策略依据仓库 README 的 EI 与不确定度表述。</text>')
b.append(f'<text x="64" y="968" font-family="{FONT_CJK}" font-size="13" '
         f'fill="{L_MUTED}">本图是示意性标志与算法图解，不是实验数据；'
         f'参考图仅作风格参考，其像素未进入交付物。字标为活字，印刷前建议转曲。</text>')
b.append(f'<text x="1536" y="996" text-anchor="end" '
         f'font-family="{FONT_LATIN}" font-size="12" fill="#8FA6C9">'
         f'GPEvo · Uncertainty Wave Edition · 2026-11-07</text>')
made.append(save("annotated-wave", doc(W, H, "\n".join(b), defs(uid, "light")),
                 W, H, 2))

# --- 5. specification board ------------------------------------------------
uid = "sw"
p = PAL["dark"]
W, H = 1500, 1300
b = [f'<rect width="{W}" height="{H}" fill="{SPACE}"/>',
     f'<rect width="{W}" height="{H}" fill="url(#bg-{uid})" opacity="0.55"/>',
     f'<rect x="0" y="0" width="1500" height="7" fill="url(#curve-{uid})"/>',
     f'<text x="64" y="80" font-family="{FONT_LATIN}" font-size="42" '
     f'font-weight="700" fill="{WH}">GPEvo</text>',
     f'<text x="248" y="80" font-family="{FONT_LATIN}" font-size="26" '
     f'fill="{MUTED}">/ Uncertainty Wave Edition</text>',
     f'<text x="64" y="118" font-family="{FONT_CJK}" font-size="18" '
     f'fill="{MUTED}">波浪形 μ(x) · 分层 ±1σ / ±1.96σ 不确定度带 · 主动学习下一个查询点　|　wave/</text>',
     f'<text x="64" y="168" font-family="{FONT_CJK}" font-size="15" '
     f'font-weight="700" letter-spacing="2" fill="{CY}">01　图标 / ICON（深色 · 浅色）</text>',
     f'<rect x="64" y="186" width="392" height="340" rx="18" fill="{DEEP}" '
     f'stroke="#1E3057"/>',
     f'<g transform="translate(104,208) scale(0.58)">{mark_body(uid, "dark")}</g>',
     f'<rect x="480" y="186" width="392" height="340" rx="18" fill="#FFFFFF" '
     f'stroke="#1E3057"/>']
# light icon needs light defs -> use a second uid defined in the same document
uid_l = "sl"
defs_all = defs(uid, "dark") + "\n" + defs(uid_l, "light")
b.append(f'<g transform="translate(520,208) scale(0.58)">'
         f'{mark_body(uid_l, "light")}</g>')
b.append(f'<text x="896" y="200" font-family="{FONT_CJK}" font-size="15" '
         f'font-weight="700" letter-spacing="2" fill="{CY}">02　主标 / LOCKUP</text>')
b.append(f'<rect x="896" y="216" width="540" height="294" rx="18" '
         f'fill="{DEEP}" stroke="#1E3057"/>')
b.append(f'<g transform="translate(908,232) scale(0.30)">'
         f'{mark_body(uid, "dark")}</g>')
b.append(wordmark(1074, 350, 62, uid, "dark"))
b.append(taglines(1076, 388, 11, "dark"))
b.append(f'<text x="64" y="552" font-family="{FONT_CJK}" font-size="13" '
         f'fill="{MUTED}">512 × 512 · 深色用于 README / 深色 PPT，浅色用于论文与浅色文档；'
         f'同一份 GP 后验几何，仅换配色层</text>')

b.append(f'<text x="64" y="606" font-family="{FONT_CJK}" font-size="15" '
         f'font-weight="700" letter-spacing="2" fill="{CY}">'
         f'03　图形 → 算法语义（可追溯）</text>')
rows = [
    ("青→紫波浪曲线", "μ(x) 后验均值", "RBF 核 + 常数先验均值的真实 GP 后验，含 2 峰 2 谷"),
    ("外层半透明飘带", "μ ± 1.96σ", "95% 置信带；观测点处收窄，两侧外推区张开"),
    ("内层较实飘带", "μ ± 1σ", "内层不确定度；与外层共同读出 σ 的大小变化"),
    ("白芯彩色圆点 ×4", "已标注样本", "被 SIESTA / DFT 打标签的结构，构成训练集"),
    ("琥珀空心环 + 十字", "下一个查询点", "主动学习在高 σ 区选点；EI 由同一后验真实计算"),
    ("淡粒子 / 网格 / 极光", "候选结构与搜索域", "装饰性，不承载数值；极光只作氛围，不抢主体"),
]
for i, (a, c, d_) in enumerate(rows):
    y = 624 + i * 54
    b.append(f'<rect x="64" y="{y}" width="1372" height="47" rx="9" '
             f'fill="{INK}" stroke="#1A2B4E"/>')
    b.append(f'<text x="88" y="{y+30}" font-family="{FONT_CJK}" font-size="15" '
             f'font-weight="700" fill="{CY}">{a}</text>')
    b.append(f'<text x="330" y="{y+30}" font-family="{FONT_CJK}" font-size="15" '
             f'font-weight="700" fill="{WH}">{c}</text>')
    b.append(f'<text x="580" y="{y+30}" font-family="{FONT_CJK}" font-size="14" '
             f'fill="{MUTED}">{d_}</text>')

b.append(f'<text x="64" y="978" font-family="{FONT_CJK}" font-size="15" '
         f'font-weight="700" letter-spacing="2" fill="{CY}">04　发光色阶与采样点色</text>')
cols = [("Cyan", CY), ("Blue", BL), ("Violet", VI),
        ("Magenta", MG), ("Query σ", AMBER), ("Space", SPACE)]
for i, (nm, c) in enumerate(cols):
    x = 64 + i * 230
    b.append(f'<rect x="{x}" y="996" width="210" height="64" rx="10" '
             f'fill="{c}" stroke="#22375E"/>')
    tc = "#0B1836" if c in (CY, MG, AMBER) else WH
    b.append(f'<text x="{x+14}" y="1022" font-family="{FONT_LATIN}" '
             f'font-size="15" font-weight="700" fill="{tc}">{nm}</text>')
    b.append(f'<text x="{x+14}" y="1044" font-family="{FONT_MONO}" '
             f'font-size="13" fill="{tc}">{c}</text>')

b.append(f'<text x="64" y="1102" font-family="{FONT_CJK}" font-size="15" '
         f'font-weight="700" letter-spacing="2" fill="{CY}">05　小尺寸核验</text>')
for i, sz in enumerate([128, 64, 32]):
    x = 64 + i * 150
    sc = sz / 512.0
    ox, oy = x + (132 - sz) / 2.0, 1116 + (132 - sz) / 2.0
    b.append(f'<rect x="{x}" y="1116" width="132" height="132" rx="10" '
             f'fill="{DEEP}" stroke="#1E3057"/>')
    b.append(f'<g transform="translate({n(ox)},{n(oy)}) scale({n(sc)})">'
             f'{mark_body(uid, "dark", query=True, aurora=False)}</g>')
    b.append(f'<text x="{x+66}" y="1270" text-anchor="middle" '
             f'font-family="{FONT_MONO}" font-size="13" fill="{MUTED}">{sz} px</text>')
b.append(f'<text x="540" y="1160" font-family="{FONT_CJK}" font-size="14" '
         f'fill="{MUTED}">深色发光版建议不小于 64 px；32 px 仅保留波浪与置信带轮廓，</text>')
b.append(f'<text x="540" y="1186" font-family="{FONT_CJK}" font-size="14" '
         f'fill="{MUTED}">印刷 / 单色 / 极小尺寸请改用浅色版或标准版图标。</text>')
b.append(f'<text x="540" y="1222" font-family="{FONT_CJK}" font-size="14" '
         f'fill="{MUTED}">文案可溯源至仓库 README 与介绍 PPT（MIT、Python 3.9+、no MATLAB）；</text>')
b.append(f'<text x="540" y="1248" font-family="{FONT_CJK}" font-size="14" '
         f'fill="{MUTED}">无性能承诺；参考图像素未进入交付物；标准版与 Glow 版文件未改动。</text>')
b.append(f'<text x="64" y="1286" font-family="{FONT_LATIN}" font-size="12" '
         f'fill="#5D759C">GPEvo · Uncertainty Wave Edition · 2026-11-07</text>')
made.append(save("showcase-wave", doc(W, H, "\n".join(b), defs_all), W, H, 1))

# --- 6. favicons -----------------------------------------------------------
icon_svg = os.path.join(SVGD, "icon-wave.svg")
for sz in (32, 64):
    os.system('rsvg-convert -w %d -h %d -o "%s" "%s"'
              % (sz, sz, os.path.join(PNGD, f"favicon-{sz}.png"), icon_svg))

# --- 7. manifest -----------------------------------------------------------
base_files = [
    "svg/icon-dark.svg", "svg/lockup-horizontal.svg",
    "png/showcase.png", "manifest.json",
    "glow/svg/icon-glow.svg", "glow/png/showcase-glow.png",
    "glow/manifest-glow.json",
]
hashes = {}
for rel in base_files:
    fp = os.path.join(ROOT, rel)
    if os.path.exists(fp):
        hashes[rel] = hashlib.sha256(open(fp, "rb").read()).hexdigest()

manifest = {
    "task": "Add an Uncertainty/Wave edition: a wavy GP posterior with a "
            "two-layer uncertainty band and explicit active-learning query",
    "status": "completed",
    "mode": "local_addition_new_geometry",
    "why": "The user's reference image reads as a wave wrapped in ribbons "
           "that widen at the ends; the earlier editions used a monotone "
           "rising mu. This edition recomputes the posterior so the "
           "uncertainty band is the visual protagonist.",
    "evidence_refs": [
        "https://gitee.com/fenggo/gpevo (README: GP posterior, EI, "
        "u = +/-1.96 sigma uncertainty threshold, active-learning loop)",
        "/home/feng/gpguided/GPEvo软件介绍.pptx",
        "/home/feng/gpguided/outputs/gpevo-logo/manifest.json",
        "/home/feng/.workbuddy/clipboard-images/"
        "clipboard-2026-11-07T14-22-57-598Z-60b56fd2.jpg (style reference only)",
    ],
    "gp_model": {
        "kernel": "RBF: sf2*exp(-(x-x')^2/(2*l^2)) + sn2*I",
        "lengthscale": ELL, "signal_var": SF2, "noise_var": SN2,
        "prior_mean": "constant m0=%s" % M0,
        "observations": list(zip(OBS_X, OBS_Y)),
        "posterior": "mu=m0+k^T K^-1 (y-m0); var=sf2-k^T K^-1 k",
        "EI": "(mu-f*-xi)Phi(Z)+sigma phi(Z), Z=(mu-f*-xi)/sigma, "
              "f*=max(y), xi=%s" % XI,
        "sigma_range": [round(min(SIG), 4), round(max(SIG), 4)],
        "mu_range": [round(min(MU), 4), round(max(MU), 4)],
        "mu_local_extrema_x": [round(v, 4) for v in EXTREMA],
        "ei_max": round(EI_MAX, 5),
        "band_visual_compression": "half(k,sigma)=HW*(1-exp(-K_SIG*k*sigma)), "
                                   "HW=%s K_SIG=%s" % (HW, K_SIG),
        "next_query_x": QUERY_X,
        "next_query_note": "The strict argmax of EI lies at the extrapolation "
                           "boundary of the search domain. The ring marks a "
                           "representative candidate inside the high-sigma "
                           "region; the annotated artwork states this explicitly.",
    },
    "style_reference": {
        "role": "style_only",
        "pixels_used_in_deliverables": False,
        "abstracted": ["S-shaped luminous wave", "translucent ribbons wide at "
                       "the ends and pinched in the middle", "particles",
                       "dark grid", "faint aurora"],
    },
    "unchanged": ["standard edition (svg/, png/, manifest.json)",
                  "glow edition (glow/)"],
    "baseline_sha256": hashes,
    "artifacts": {
        "svg_sources": ["wave/svg/" + os.path.basename(x) for x in made],
        "png_exports": sorted(os.listdir(PNGD)),
    },
    "content_review": {
        "facts": "pass - MIT / Python 3.9+ / no MATLAB / gitee.com/fenggo/gpevo "
                 "traced to README and intro PPT; EI and +/-1.96 sigma described "
                 "in README; no performance claims. DISCLOSED EXCEPTION: the "
                 "light horizontal lockup carries a forward-looking "
                 "github.com/fenggo/gpevo footer. The user confirmed this "
                 "repository is planned but not yet created, so the URL does "
                 "not resolve today and is not traceable to the sources "
                 "(see url_audit). It is not presented as an existing repo.",
        "math": "pass - GP posterior and EI computed from the stated kernel in "
                "pure Python; band compression is disclosed, not passed off as "
                "raw sigma",
        "rights": "pass - all vectors drawn from scratch; reference image "
                  "pixels not embedded",
        "brand": "pass - wordmark and tagline unchanged; new subline "
                 "'POSTERIOR UNCERTAINTY / ACTIVE LEARNING' is a method "
                 "description, not a product claim",
        "media": "pass - all 7 SVG references resolved, no placeholders, "
                 "9 PNG rendered non-blank",
        "visual_review_note": "All rendered PNGs were visually inspected: "
                "icon (dark + light), lockup (dark + light), README banner, "
                "annotated diagram, specification board and 32px favicon. "
                "One render defect was found and fixed: an unevaluated "
                "arithmetic string inside the small-size <g transform> made "
                "three unscaled 512px marks cover the board header and left "
                "the sample frames empty; corrected, re-rendered and "
                "re-inspected (128/64/32px all visible). Also softened the "
                "wave's two ends, lengthened band fades, and recoloured the "
                "first light-palette observation so it cannot be confused "
                "with the amber next-query ring. 32px retains the wave "
                "outline only, as stated on the board. Post-revision: both "
                "standalone icons re-rendered and visually re-inspected; "
                "no vertical leader remains under the query ring. A later "
                "revision removed the leader from the light horizontal "
                "lockup as well; it was re-rendered and visually inspected. "
                "After the footer URL change the light lockup was again "
                "re-rendered and inspected: text fits, no clipping or overlap.",
        "production_note": "Wordmark is live text (convert to outlines before "
                           "print). Dark editions for README / dark slides; "
                           "light icon and annotated diagram for papers.",
    },
    "local_revisions": [
        {
            "date": "2026-11-07",
            "request": "Remove the dangling amber dashed vertical leader "
                       "under the next-query ring from icon-wave.png and "
                       "icon-wave-light.png",
            "scope": "icon-wave.svg/.png, icon-wave-light.svg/.png and the "
                     "32/64 favicons rendered from icon-wave.svg. The "
                     "banner, DARK lockup, specification board and annotated "
                     "diagram keep the guide line (there it functions as a "
                     "diagrammatic leader). The light lockup was changed in "
                     "the following revision.",
            "implementation": "mark_body(..., guide=False) is used only for "
                              "the two standalone icons; default stays True.",
        },
        {
            "date": "2026-11-07",
            "request": "Also remove the dangling amber dashed vertical leader "
                       "under the next-query ring from lockup-wave-light.png",
            "scope": "lockup-wave-light.svg/.png only. lockup-wave (dark), "
                     "banner-wave, showcase-wave, annotated-wave, the two "
                     "standalone icons and the favicons are byte-identical "
                     "to the previous revision.",
            "implementation": "In the horizontal-lockup loop, guide=False is "
                              "passed to mark_body() for mode=='light' only; "
                              "mode=='dark' still uses the default guide=True.",
        },
        {
            "date": "2026-11-07",
            "request": "In lockup-wave-light.png change the footer line "
                       "gitee.com/fenggo/gpevo to github.com/fenggo/gpevo",
            "scope": "lockup-wave-light.svg/.png only. The dark lockup, "
                     "README banner, icons, specification board, annotated "
                     "diagram and the standard / glow editions keep the "
                     "documented Gitee URL.",
            "implementation": "SITE_URL dict keyed by lockup mode; only "
                              "mode=='light' renders the GitHub string.",
            "verification": "The new URL is user-specified, not a verified "
                            "fact: a public request on 2026-11-07 returned "
                            "GitHub's 'Page not found' page. Recorded in "
                            "url_audit rather than claimed as traceable.",
        },
    ],
    "url_audit": {
        "documented_in_readme_and_intro_ppt": "gitee.com/fenggo/gpevo",
        "user_specified_for_light_lockup": "github.com/fenggo/gpevo",
        "checked_on": "2026-11-07",
        "github_check_result": "public request returned GitHub 'Page not "
                               "found' (404) - may be uncreated, private, "
                               "or a different owner/path",
        "status": "UNVERIFIED - confirm the real GitHub URL before publishing "
                  "this lockup, or revert to the documented Gitee URL",
        "note": "This is a factual discrepancy inside one brand system: "
                "other deliverables still point to Gitee. Unify only after "
                "the correct canonical URL is confirmed.",
    },
    "blockers": [],
}
with open(os.path.join(OUT, "manifest-wave.json"), "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, ensure_ascii=False, indent=2)

print("Wave edition written: %d SVG + %d PNG" % (len(made), len(os.listdir(PNGD))))
print("Baseline sha256 (unchanged editions):")
for k, v in hashes.items():
    print("  %s  %s" % (v[:16], k))
