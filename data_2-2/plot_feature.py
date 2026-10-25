#!/usr/bin/env python
"""Academic-style scatter plots of DFT feature vs density.

Reads `feature.csv` and produces two publication-ready figures:
    * Etotal.pdf  -- total energy vs density with a fitted trend line
    * Hbond.pdf   -- H-bond energy (ehb) vs density with a fitted trend line

Each panel carries an inset histogram of the target property together with
its fitted normal distribution, using a consistent, journal-friendly style.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---------------------------------------------------------------------------
# Academic journal style: STIX fonts, minimal grid, constrained layout.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    # --- Fonts ---
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,

    # --- Figure ---
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.transparent': False,

    # --- Axes ---
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.axisbelow': True,

    # --- Lines & markers ---
    'lines.linewidth': 1.2,
    'lines.markersize': 6,
    'lines.markeredgewidth': 0.8,

    # --- Ticks ---
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,

    # --- Legend ---
    'legend.frameon': True,
    'legend.framealpha': 0.85,
    'legend.edgecolor': '0.5',
    'legend.fancybox': False,
})

# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Wong 2011, Nature Methods).
# ---------------------------------------------------------------------------
C_GREEN = '#009E73'          # Etotal scatter
C_RED = '#D55E00'            # Highlight / annotation
C_FIT = '#404040'            # Trend line (dark grey)
C_BLUE = '#0072B2'           # Hbond scatter
C_HIST = '#CCCCCC'           # Inset histogram fill

FEATURES = ['none', 'E$_{\\mathrm{total}}$', 'E$_{\\mathrm{angle}}$',
            'E$_{\\mathrm{torsion}}$', 'E$_{\\mathrm{vdw}}$',
            'E$_{\\mathrm{hbond}}$', 'E$_{\\mathrm{coul}}$']

# Annotations are placed for the outliers near the upper density tail or for
# unusually low H-bond energies (the hand-tuned value removes a stray point).
D_ANNOT_THRESH = 1.864
H_ANNOT_THRESH = -0.340


def _read_data():
    """Load feature.csv and return the filtered sample, original indexing."""
    data = np.loadtxt('feature.csv', delimiter=',', skiprows=1)
    etot = data[:, 1]
    ehb  = data[:, 6] + data[:, 7] + data[:, 8]
    density = data[:, -1]

    idx = np.where((density > 1.75) &
                   (np.abs(etot - etot[0]) < 1.75))[0]
    return (density[idx], etot[idx], ehb[idx], idx)


def _fit_trend(x, y):
    """Ordinary least-squares fit; return fit line and fit summary text."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = slope * x_fit + intercept
    return x_fit, y_fit, r_value, slope, intercept, std_err


def _add_inset(ax, values, color, xlabel, ylabel='Density'):
    """Add a histogram + normal-fit inset in the top-right corner."""
    axins = inset_axes(ax, width='32%', height='28%', loc='upper right',
                       borderpad=1.5)
    mu, std = np.mean(values), np.std(values)
    x_norm = np.linspace(values.min(), values.max(), 100)
    y_norm = stats.norm.pdf(x_norm, mu, std)

    axins.hist(values, bins=15, density=True, alpha=0.5,
               color=C_HIST, edgecolor='0.4', linewidth=0.5)
    axins.plot(x_norm, y_norm, '-', linewidth=1.2, color=C_RED,
               label=r'$\mathcal{N}$(%s, %s)' % (f'{mu:.2f}', f'{std:.2f}'))

    axins.tick_params(axis='both', which='major', labelsize=7,
                      direction='in', length=2, width=0.5)
    axins.tick_params(axis='both', which='minor', labelsize=7,
                      direction='in', length=1.5, width=0.4)
    axins.spines['top'].set_visible(False)
    axins.spines['right'].set_visible(False)
    axins.set_xlabel(xlabel, fontsize=8, labelpad=2)
    axins.set_ylabel(ylabel, fontsize=8, labelpad=2)
    axins.legend(fontsize=6.5, frameon=False, loc='upper right',
                 handlelength=1.0, handletextpad=0.4)
    return axins


def _annotate_outliers(ax, x, y, idx, color):
    """Label the structural outliers with their row ID."""
    for xi, yi, ii in zip(x, y, idx):
        if xi > D_ANNOT_THRESH or (yi < H_ANNOT_THRESH and ii != 495):
            ax.annotate(f'ID:{ii}', (xi, yi), xytext=(0, 6),
                        textcoords='offset points', ha='center',
                        fontsize=6.5, color=color)


def _plot_scatter(density, y, idx, color, marker, ylabel, outfile,
                  panel_label=''):
    """Render one density-vs-feature panel and save the figure."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)

    # --- Panel label (e.g., "(a)") ---
    if panel_label:
        ax.text(0.02, 0.96, panel_label, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top', ha='left')

    ax.set_ylabel(ylabel)
    ax.set_xlabel(r'Density (g/cm$^{-3}$)')

    # Scatter: hollow markers for over-plotting clarity.
    ax.scatter(density, y, s=28, marker=marker, alpha=0.80,
               facecolors='none', edgecolors=color, label=ylabel,
               linewidths=0.8, zorder=3)

    _annotate_outliers(ax, density, y, idx, color=C_RED)

    # Trend line: true OLS fit, no offset.
    x_fit, y_fit, r_value, slope, intercept, std_err = _fit_trend(density, y)
    ax.plot(x_fit, y_fit, '--', color=C_FIT, linewidth=1.0,
            label=r'Fit ($r = %s$)' % f'{r_value:.3f}', zorder=2)

    _add_inset(ax, y, color, ylabel)

    # Legend.
    # ax.legend(loc='best', ncol=1, handlelength=1.5, handletextpad=0.5,
    #           borderpad=0.4, labelspacing=0.3)

    fig.savefig(outfile)
    plt.close(fig)


def main():
    density, etot, ehb, idx = _read_data()

    # Panel 1: total energy vs density.
    _plot_scatter(density, etot, idx, color=C_GREEN, marker='s',
                  ylabel=FEATURES[1], outfile='Etotal.pdf',
                  panel_label='(a)')

    # Panel 2: H-bond energy vs density.
    _plot_scatter(density, ehb, idx, color=C_BLUE, marker='o',
                  ylabel=r'$E_{\mathrm{H-bond}}$ (eV)', outfile='Hbond.pdf',
                  panel_label='(b)')


if __name__ == '__main__':
    main()
    