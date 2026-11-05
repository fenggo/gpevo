import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(22, 16))
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis('off')

# --- Color palette ---
C_INIT   = '#4A90D9'
C_OPT    = '#E74C3C'
C_FIT    = '#F39C12'
C_GPML   = '#9B59B6'   # purple for GPML
C_SEL    = '#8E44AD'
C_VAR    = '#3498DB'
C_ELITE  = '#1ABC9C'
C_END    = '#27AE60'
C_JUDGE  = '#F1C40F'
C_ZONE   = '#F5F6FA'
C_DARK   = '#2C3E50'
C_WHITE  = '#FFFFFF'
C_STATS  = '#95A5A6'
C_GPZONE = '#F3E5F5'   # light purple zone for GPML

def draw_box(ax, x, y, w, h, color, text, text_color=C_DARK, fontsize=10, bold=False):
    shadow = FancyBboxPatch((x - w/2 + 0.06, y - h/2 - 0.06), w, h,
                             boxstyle="round,pad=0.10",
                             facecolor='#d5d8dc', edgecolor='none', alpha=0.4)
    ax.add_patch(shadow)
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.10",
                          facecolor=color, edgecolor=C_DARK, linewidth=1.0)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=weight)

def draw_diamond(ax, x, y, w, h, color, text, text_color=C_DARK, fontsize=10):
    shadow = Polygon([
        (x + 0.05, y + h/2), (x + w/2 + 0.05, y), (x + 0.05, y - h/2), (x - w/2 + 0.05, y)
    ], facecolor='#d5d8dc', edgecolor='none', alpha=0.4)
    ax.add_patch(shadow)
    diamond = Polygon([
        (x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)
    ], facecolor=color, edgecolor=C_DARK, linewidth=1.0)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold')

def arrow(ax, x1, y1, x2, y2, color=C_DARK, lw=1.5, style='solid'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad=0', linestyle=style))

def arrow_label(ax, x1, y1, x2, y2, label, color=C_DARK):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                               connectionstyle='arc3,rad=0'))
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mx, my + 0.22, label, ha='center', va='center', fontsize=8,
            color=color, fontweight='bold', fontstyle='italic')

# ============================================
# TITLE
# ============================================
ax.text(11, 15.5, 'USPEX Evolutionary Algorithm with Gaussian Process Active Learning',
        ha='center', va='center', fontsize=18, fontweight='bold', color=C_DARK)
ax.text(11, 14.9, 'optType = 310  |  TNT₄·CL20₄  |  228 atoms  |  UCB/EI Acquisition',
        ha='center', va='center', fontsize=9, color='#7f8c8d', fontstyle='italic')

# ============================================
# LEFT COLUMN: Main EA pipeline
# ============================================
LX = 3.5

# 1. Init
draw_box(ax, LX, 13.5, 3.6, 0.85, C_INIT, 'Initialization', C_WHITE, 10, True)
ax.text(LX, 13.1, 'Random structures (Gen 1)', ha='center', fontsize=7, color='#7f8c8d')
arrow(ax, LX, 13.05, LX, 12.2)

# 2. Local Opt
draw_box(ax, LX, 11.7, 3.6, 0.85, C_OPT, 'Local Optimization', C_WHITE, 10, True)
ax.text(LX, 11.3, 'VASP / GULP relaxation', ha='center', fontsize=7, color='#ecf0f1')
arrow(ax, LX, 11.25, LX, 10.5)

# 3. Fitness
draw_box(ax, LX, 10.0, 3.6, 0.85, C_FIT, 'Compute Fitness', C_WHITE, 10, True)
ax.text(LX, 9.6, 'fitness = -density', ha='center', fontsize=7, color='#fdebd0')

# Stats side box
draw_box(ax, LX + 2.5, 10.0, 1.8, 0.55, C_STATS, 'Statistics', C_WHITE, 7)
ax.text(LX + 2.5, 9.75, 'AntiSeeds / Ranking', ha='center', fontsize=6, color='#ecf0f1')
arrow(ax, LX + 1.8, 10.0, LX + 1.6, 10.0)

arrow(ax, LX, 9.55, LX, 9.0)

# ============================================
# CENTER-LEFT: GPML Active Learning (big block)
# ============================================
# GPML zone background
gp_zone = FancyBboxPatch((2.0, 4.8), 8.5, 4.2,
                          boxstyle="round,pad=0.3",
                          facecolor=C_GPZONE, edgecolor=C_GPML,
                          linewidth=1.5, linestyle='--')
ax.add_patch(gp_zone)
ax.text(2.5, 8.8, 'GPML Active Learning (UQ)', fontsize=10, color=C_GPML, fontweight='bold')

# GPML internal steps
GX = 4.3

# G1: Read gp.csv
draw_box(ax, GX, 8.0, 3.0, 0.65, C_GPML, 'Read gp.csv', C_WHITE, 8, True)
ax.text(GX, 7.7, 'GP model predictions', ha='center', fontsize=6.5, color='#d7bde2')

arrow(ax, GX, 7.65, GX, 7.2)

# G2: Scan & Filter
draw_box(ax, GX, 6.85, 3.0, 0.65, C_GPML, 'Scan & Filter', C_WHITE, 8, True)
ax.text(GX, 6.55, 'residual > 10 → skip (damaged)', ha='center', fontsize=6.5, color='#d7bde2')

arrow(ax, GX, 6.5, GX, 6.05)

# G3: EI/UCB selection
draw_box(ax, GX, 5.7, 3.0, 0.65, C_GPML, 'EI / UCB Selection', C_WHITE, 8, True)
ax.text(GX, 5.4, 'EI = (μ−f_best)·Φ(Z) + σ·φ(Z)', ha='center', fontsize=6.5, color='#d7bde2')

# G3 decision diamond
draw_diamond(ax, GX, 4.85, 2.2, 0.7, C_JUDGE, 'σ > thresh?', C_DARK, 7)

# YES path: DFT trigger
arrow_label(ax, GX + 1.1, 4.85, GX + 2.8, 4.85, 'YES', C_OPT)
draw_box(ax, GX + 4.0, 4.85, 2.2, 0.55, C_OPT, 'DFT Calc', C_WHITE, 7, True)
ax.text(GX + 4.0, 4.6, 'traj → calc → pred', ha='center', fontsize=6, color='#ecf0f1')

# NO path: skip
arrow_label(ax, GX, 4.5, GX, 4.05, 'NO', C_STATS)

# G4: Update fitness
draw_box(ax, GX, 3.75, 3.2, 0.55, C_GPML, 'Update Fitness', C_WHITE, 8, True)
ax.text(GX, 3.5, 'fitness = -density_gp', ha='center', fontsize=6.5, color='#d7bde2')

# Arrow from DFT to update
arrow(ax, GX + 2.9, 4.85, GX + 1.6, 3.75, C_OPT, 1.2, 'dashed')

# Arrow from GP input to GP block
arrow(ax, LX, 9.0, GX, 8.35, C_GPML, 1.5)

# Arrow out of GPML to decision
arrow(ax, GX, 3.45, GX, 3.0)

# ============================================
# Decision: Converged?
# ============================================
draw_diamond(ax, GX, 2.55, 2.4, 0.85, C_JUDGE, 'Converged?', C_DARK, 10)

# YES → End
arrow_label(ax, GX + 1.2, 2.55, 13.0, 2.55, 'YES', C_END)
draw_box(ax, 14.5, 2.55, 2.6, 0.75, C_END, 'Best Structure', C_WHITE, 10, True)

# NO → Selection
arrow_label(ax, GX, 2.1, GX, 1.55, 'NO', C_OPT)

# Selection
draw_box(ax, GX, 1.1, 3.6, 0.75, C_SEL, 'Selection', C_WHITE, 10, True)
ax.text(GX, 0.8, 'Tournament / Roulette Wheel', ha='center', fontsize=7, color='#d5c6e0')

# Arrow from selection to variation
arrow(ax, GX + 1.8, 1.1, 9.0, 1.1)

# ============================================
# RIGHT: Variation Zone
# ============================================
zone = FancyBboxPatch((8.5, 0.2), 9.0, 5.8,
                       boxstyle="round,pad=0.3",
                       facecolor=C_ZONE, edgecolor='#bdc3c7',
                       linewidth=1.2, linestyle='--')
ax.add_patch(zone)
ax.text(9.0, 5.8, 'Variation Operators', fontsize=10, color='#7f8c8d', fontweight='bold')

var_names = [
    'Heredity\nCrossover',
    'Random\nGeneration',
    'Permutation\nMol. Swap',
    'Rotation\nMol. Rotate',
    'LatMutation\nLattice',
    'SoftMode\nAtomic',
]
var_params = [
    'howManyOffsprings', 'howManyRand', 'howManyPermutations',
    'howManyRotations', 'howManyMutations', 'howManyAtomMutations',
]

VX0 = 9.5
VY0 = 5.0
VDX = 2.7
VDY = 1.6
VW = 2.4
VH = 0.85

for i in range(6):
    row = i // 3
    col = i % 3
    vx = VX0 + col * VDX
    vy = VY0 - row * VDY
    draw_box(ax, vx, vy, VW, VH, C_VAR, var_names[i], C_WHITE, 7.5, True)
    ax.text(vx, vy - 0.55, var_params[i], ha='center', fontsize=6, color='#95a5a6')

# Arrow from variation to elitism
arrow(ax, 13.0, 1.1, 13.0, 0.8)

# ============================================
# Elitism
# ============================================
draw_box(ax, 13.0, 0.4, 3.6, 0.7, C_ELITE, 'Elitism', C_WHITE, 10, True)
ax.text(13.0, 0.1, 'Best individuals survive', ha='center', fontsize=7, color='#d1f2eb')

# ============================================
# LOOP BACK: Elitism → Local Opt
# ============================================
loop = FancyArrowPatch((14.8, 0.4), (14.8, 11.7),
                        connectionstyle='arc3,rad=-0.55',
                        arrowstyle='->', color=C_OPT, lw=2.2,
                        linestyle='dashed')
ax.add_patch(loop)
ax.text(17.5, 6.0, 'Next\nGeneration', ha='center', va='center',
        fontsize=10, color=C_OPT, fontweight='bold', rotation=90)

# ============================================
# KEY FORMULAS CALL-OUT (right side)
# ============================================
ax.text(17.0, 14.5, 'Key Formulas', fontsize=10, fontweight='bold', color=C_DARK)
ax.text(17.0, 13.8, 'UCB: μ + κ·σ', fontsize=8, color='#555', fontfamily='monospace')
ax.text(17.0, 13.3, 'EI: (μ−f*)·Φ(Z) + σ·φ(Z)', fontsize=8, color='#555', fontfamily='monospace')
ax.text(17.0, 12.8, 'Z = (μ−f*) / σ', fontsize=8, color='#555', fontfamily='monospace')
ax.text(17.0, 12.2, 'fitness = -density_gp', fontsize=8, color='#555', fontfamily='monospace')

# ============================================
# LEGEND
# ============================================
ly = 15.2
leg = [
    (C_INIT,  'Init', C_WHITE),
    (C_OPT,   'Local Opt.', C_WHITE),
    (C_FIT,   'Fitness', C_WHITE),
    (C_GPML,  'GPML / UQ', C_WHITE),
    (C_JUDGE, 'Decision', C_DARK),
    (C_SEL,   'Selection', C_WHITE),
    (C_VAR,   'Variation', C_WHITE),
    (C_ELITE, 'Elitism', C_WHITE),
    (C_END,   'Output', C_WHITE),
]

lx = 1.0
for color, label, tc in leg:
    if label == 'Decision':
        ax.fill([lx, lx+0.1, lx+0.2, lx+0.1, lx],
                [ly+0.05, ly+0.1, ly+0.05, ly, ly+0.05],
                color=color, edgecolor=C_DARK, linewidth=0.4)
        lx += 0.22
    else:
        box = FancyBboxPatch((lx, ly), 0.2, 0.1,
                              boxstyle="round,pad=0.01",
                              facecolor=color, edgecolor=C_DARK, linewidth=0.5)
        ax.add_patch(box)
        lx += 0.22
    ax.text(lx, ly + 0.05, label, ha='left', va='center', fontsize=7, color='#555')
    lx += 1.3

plt.savefig('/home/feng/uspex_tnt/uspex_gp/docs/ea_mechanism.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('PNG saved successfully')