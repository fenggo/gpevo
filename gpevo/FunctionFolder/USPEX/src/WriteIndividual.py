"""Python port of WriteIndividual.m.

Rewrites ``resFolder/Individuals`` from the cumulative
``USPEX_STRUC.POPULATION`` record, one row per structure ever relaxed.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G

HEADER = "Gen   ID    Origin   Composition    Enthalpy   Volume  Density   Fitness   KPOINTS  SYMM  Q_entr A_order S_order"
UNITS = "                                      (eV)     (A^3)   (g/cm^3)"


def _composition(num_ions):
    return "[" + "".join(f"{int(n):3d}" for n in num_ions) + "]"


def _row(index, record):
    num_ions = list(getattr(record, "numIons", []) or [])
    total = int(sum(int(v) for v in num_ions))
    enth = float(np.asarray(record.Enthalpies, dtype=float).ravel()[-1])
    volume = getattr(record, "Vol", None)
    density = getattr(record, "density", None)
    symg = getattr(record, "symg", None)
    fitness = getattr(record, "Fitness", None)
    entropy = getattr(record, "struc_entr", None)
    s_order = getattr(record, "S_order", None)
    order = np.asarray(getattr(record, "order", []), dtype=float).reshape(-1)
    kpoints = np.asarray(getattr(record, "K_POINTS", []), dtype=float)
    if kpoints.ndim == 2 and kpoints.shape[0] > 0:
        kpoints = kpoints[-1]
    else:
        kpoints = np.ones(3)

    def num(value, width, precision, fallback="N/A"):
        if value is None:
            return f"{fallback:>{width}s}"
        try:
            return f"{float(value):{width}.{precision}f}"
        except (TypeError, ValueError):
            return f"{fallback:>{width}s}"

    a_o = float(np.sum(order) / total) if (order.size and total > 0) else 0.0
    kp = "[" + " ".join(f"{int(round(float(v))):2d}" for v in np.asarray(kpoints).reshape(-1)[:3]) + "]"
    if symg is None or str(symg).strip().upper() == "N/A":
        sym_s = f"{'N/A':>5s}"
    else:
        sym_s = f"{int(float(symg)):3d}  "

    # Origin names come with MATLAB's own padding ('  Random  ') and with
    # different lengths ('Random' vs 'softmutate'); strip them so that the
    # following columns stay aligned.
    origin = str(getattr(record, "howCome", "") or "").strip()

    return (
        f"{int(getattr(record, 'gen', 0) or 0):3d} {index:4d} "
        f"{origin:<11s} "
        f"{_composition(num_ions):<13s} "
        f"{enth:9.3f} "
        f"{num(volume, 9, 3)} "
        f"{num(density, 7, 3)} "
        f"{num(fitness, 10, 3)} "
        f"{kp} "
        f"{sym_s} "
        f"{num(entropy, 6, 3)} "
        f"{a_o:6.3f} "
        f"{num(s_order, 6, 3)}"
    )


def WriteIndividual(resFolder, energyUnits=None):
    USPEX = G.get("USPEX_STRUC")
    if USPEX is None:
        return
    population = list(getattr(USPEX, "POPULATION", []) or [])
    path = f"{resFolder}/Individuals"
    try:
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(HEADER + "\n")
            if energyUnits == "kcal/mol":
                fp.write("                                   (kcal/mol)  (A^3)   (g/cm^3)\n")
            else:
                fp.write(UNITS + "\n")
            for index, record in enumerate(population, start=1):
                if record is None:
                    continue
                fp.write(_row(index, record) + "\n")
    except OSError:
        return
