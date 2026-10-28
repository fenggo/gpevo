"""Python equivalent of WriteGenerationOutput_310.m (imperfect, best-effort 310 subset)."""
from __future__ import annotations

import os
import shutil
import numpy as np

from gpevo.runtime import G
from gpevo.FunctionFolder.sys.Write_POSCAR import Write_POSCAR, Write_POSCAR_order
from gpevo.FunctionFolder.sys.elementMass import calc_density

_HEADER = "Gen   ID    Origin   Composition    Enthalpy   Volume  Density   Fitness   KPOINTS  SYMM  Q_entr A_order S_order"
_UNIT_LINE = "                                   (eV)  (A^3)  (g/cm^3)"


def _composition(s):
    if hasattr(s, "numIons") and s.numIons is not None:
        return " [" + "".join(f"{int(n):3d}" for n in s.numIons) + "]"
    return "N/A"


def _volume(s):
    try:
        lattice = np.asarray(s.LATTICE, dtype=float).reshape(3, 3)
        return abs(np.linalg.det(lattice))
    except Exception:
        return None


def _kpoints(s):
    try:
        vals = [int(v) for v in np.asarray(s.K_POINTS)[-1]]
    except Exception:
        return "N/A"
    while len(vals) < 3:
        vals.append(1)
    return "[" + " ".join(str(v) for v in vals[:3]) + "]"


def _fixed(value, width, precision, default="N/A"):
    if value is None:
        return f"{default:>{width}s}"
    try:
        return f"{float(value):{width}.{precision}f}"
    except Exception:
        return f"{default:>{width}s}"


def _line(gen, ind_id, s, fitness):
    enth = float(np.asarray(s.Enthalpies)[-1])
    volume = _volume(s)
    symg = getattr(s, "symg", None)
    if symg is None or str(symg).strip().upper() == "N/A":
        symg = 1
    entropy = float(getattr(s, "struc_entr", 0.0) or 0.0)
    s_order = float(getattr(s, "S_order", 0.0) or 0.0)
    origin = str(getattr(s, "howCome", "N/A"))
    composition = _composition(s)

    if volume is None:
        volume_s = f"{'N/A':>8s}"
    else:
        volume_s = f"{volume:8.3f}"

    density = getattr(s, "density", None)
    if density is None or str(density).strip().upper() == "N/A":
        density_s = f"{'N/A':>9s} "
    else:
        density_s = f"{float(density):7.3f}  "

    if symg == "N/A":
        sym_s = f"{'N/A':>5s} "
    else:
        sym_s = f"{int(float(symg)):3d}   "

    order = getattr(s, "order", None)
    if order is None or str(order).strip().upper() == "N/A":
        ao_s = f"{0.0:7.3f} "
    else:
        order = np.asarray(order, dtype=float)
        total_ions = max(1, sum(map(int, s.numIons)))
        ao_s = f"{float(np.sum(order)) / total_ions:7.3f} "

    return (
        f"{gen:3d}" + " " * 3
        + f"{ind_id:4d}" + " " * 2
        + f"{origin:<9s}"
        + f"{composition:<15s}"
        + f"{enth:10.3f} "
        + volume_s
        + density_s
        + f"{float(fitness):9.3f} "
        + f"{_kpoints(s):<9s}"
        + sym_s
        + f"{entropy:6.3f} "
        + ao_s
        + f"{s_order:6.3f} "
        + "\n"
    )


def _append_with_header(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    add_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a") as fp:
        if add_header:
            fp.write(_HEADER + "\n")
            fp.write(_UNIT_LINE + "\n")
        fp.write(text)


def _append_binary(src, dst):
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "ab") as out:
        with open(src, "rb") as inp:
            shutil.copyfileobj(inp, out)


def _individual_id(s, fallback):
    """Return the global cumulative structure ID, matching bodyCount."""
    value = getattr(s, "Number", 0)
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 0
    return value if value > 0 else fallback


def _gp_columns(s):
    """``Enthalpy(GP) UC Density(GP) UC Residual`` block for one structure (54 chars).

    ``s.gp`` is the dict ``uspexkit.gp()`` returned, stored by submitJob when
    the GP step ran in-process.  The individual fields are the fallback for
    structures evaluated by the CLI, where ReadJobs read them back from
    CalcFold1/gp.csv.  A blank block is returned when nothing is known, so
    the table stays rectangular.
    """
    e = ue = d = ud = None
    gp = getattr(s, "gp", None)
    if isinstance(gp, dict):
        e, ue = gp.get("energy_pred"), gp.get("uncertainty_eng")
        d, ud = gp.get("density_gp"), gp.get("uncertainty")
    values = []
    for v in (e, ue, d, ud):
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            values.append(float("nan"))
    if not all(np.isfinite(v) for v in values):
        values = [float("nan")] * 4
        for k, name in enumerate(("energy_gp", "uq_eng", "density_gp", "uq")):
            try:
                values[k] = float(getattr(s, name, None))
            except (TypeError, ValueError):
                pass
    if not all(np.isfinite(v) for v in values):
        return " " * 54
    # Residual = Hilbert-space distance to the training set (optType 13's
    # objective).  Left blank when the GP step did not report one.
    residual = float("nan")
    for name in ("residual", "distance"):
        try:
            candidate = float(getattr(s, name, None))
        except (TypeError, ValueError):
            continue
        if np.isfinite(candidate):
            residual = candidate
            break
    res_s = f"{residual:10.4f}" if np.isfinite(residual) else " " * 10
    return (f"{values[0]:12.3f}{values[1]:10.4f}{values[2]:12.3f}"
            f"{values[3]:10.4f}{res_s}")


def output_table_header(gp=0):
    """Title line of the OUTPUT.txt table, aligned with ``output_row``.

    The label texts are the ones MATLAB writes; only the spacing follows the
    column boundaries of the data rows, so every label sits over its column
    (the raw MATLAB header is 6 characters longer than its own rows).  With
    ``gp`` on, the GP prediction columns are appended after SYMMETRY, exactly
    over the values ``_gp_columns`` writes.
    """
    h = ("ID"               # 0-1    id, right-aligned like the data
         "  "               # 2-3
         "Origin"           # 4-9    data column 7-17
         "     "            # 10-14
         "Composition"      # 15-25  data column 18-32
         " "                # 26
         "Enthalpy(eV)"     # 27-38  data values end at 42
         " "                # 39
         "Volume(A^3)"      # 40-50  data values end at 51
         " "                # 51
         "KPOINTS"          # 52-58  data column 54-62
         " "                # 59
         "SYMMETRY"         # 60-67  data column 63-65
         " ")               # 68     gap before the GP columns
    if gp:
        from gpevo.FunctionFolder.USPEX.src.gp_ml import GP_OUTPUT_HEADER
        h += GP_OUTPUT_HEADER       # 69-122, same 12/10/12/10/10 split as the data
    return h


def output_row(s, fallback=0, gp=0):
    """Row appended to OUTPUT.txt when one structure finishes.

    Mirrors ``WriteOUTPUT.m`` for the columns of the "ID Origin Composition
    Enthalpy Volume KPOINTS SYMMETRY" table; unlike Individuals there is no
    Gen / Density / Fitness column here.  With ``gp`` on, the GP prediction
    and its uncertainty are appended after SYMMETRY.
    """
    enth = float(np.asarray(s.Enthalpies)[-1])
    volume = _volume(s)
    if volume is None:
        volume_s = f"{'N/A':>8s}"
    else:
        volume_s = f"{volume:8.3f}"

    symg = getattr(s, "symg", None)
    if symg is None or str(symg).strip().upper() == "N/A":
        symg = 1
    sym_s = f"{int(float(symg)):3d}   "

    origin = str(getattr(s, "howCome", "N/A") or "N/A").strip()

    return (
        f"{_individual_id(s, fallback):4d}" + " " * 3
        + f"{origin:<11s}"
        + f"{_composition(s):<15s}"
        + f"{enth:10.3f} "
        + volume_s + "  "
        + f"{_kpoints(s):<9s}"
        + sym_s
        + (_gp_columns(s) if gp else "")
    )


def WriteGenerationOutput_310(fitness):
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    fitness = np.asarray(fitness, dtype=float)

    for _s in POP.POPULATION:
        _s.density = calc_density(_s.numIons, ORG.atomType, _volume(_s))

    # MATLAB takes the best from the ranking, not from argmin(fitness)
    # (WriteGenerationOutput_310.m:8) - with AntiSeeds corrections applied to
    # fitness, the two are not necessarily the same structure.
    from gpevo.FunctionFolder.USPEX.src.variation_common import valid_ranking

    ranking = valid_ranking(POP)
    best = ranking[0] if ranking else int(np.argmin(fitness))
    best_s = POP.POPULATION[best]

    # Per-generation summary of USPEX_STRUC (WriteGenerationOutput_310.m:19).
    # IND is the 1-based index of the best structure.
    from gpevo.FunctionFolder.USPEX.src.update_USPEX_GENERATION import update_USPEX_GENERATION

    update_USPEX_GENERATION([best + 1], fitness, 1)
    best_line = _line(POP.generation, _individual_id(best_s, best + 1), best_s, fitness[best])
    _append_with_header(os.path.join(ORG.resFolder, "BESTIndividuals"), best_line)

    individuals = "".join(
        _line(POP.generation, _individual_id(s, i + 1), s, fitness[i])
        for i, s in enumerate(POP.POPULATION)
    )
    _append_with_header(os.path.join(ORG.resFolder, "Individuals"), individuals)

    num_ions = list(getattr(best_s, "numIons", []))
    lattice = np.asarray(best_s.LATTICE, dtype=float)
    coor = np.asarray(best_s.COORDINATES, dtype=float)
    symg = getattr(best_s, "symg", 0)
    count = _individual_id(best_s, best + 1)
    order = getattr(best_s, "order", None)
    Write_POSCAR(list(ORG.atomType), count, symg, num_ions, lattice, coor)
    _append_binary("POSCAR", os.path.join(ORG.resFolder, "BESTgatheredPOSCARS"))
    _append_binary("POSCAR", os.path.join(ORG.resFolder, "goodPOSCARS"))
    Write_POSCAR_order(list(ORG.atomType), count, symg, num_ions, lattice, coor, order)
    _append_binary("POSCAR_order", os.path.join(ORG.resFolder, "BESTgatheredPOSCARS_order"))

    # Mean cell volume of everything seen so far, logged like MATLAB does
    # (WriteGenerationOutput_310.m:26-31).
    USPEX = G.get("USPEX_STRUC")
    volumes = []
    for record in (getattr(USPEX, "POPULATION", None) or []) if USPEX else []:
        if record is None:
            continue
        volume = _volume(record)
        if volume:
            volumes.append(float(volume))
    if volumes:
        ORG.latVolume = float(np.mean(volumes))
        from gpevo.log_utils import write_log

        # MATLAB does ``echo "Approximate volume(s): " num2str(mean(V))`` and
        # num2str prints four decimals.
        write_log(f'Approximate volume(s):  {ORG.latVolume:.4f}\n')

    # Full tables rebuilt from the cumulative record (WriteIndividual/Best/
    # Generation in WriteGenerationOutput_310.m:31,54,55).
    from gpevo.FunctionFolder.USPEX.src.WriteIndividual import WriteIndividual
    from gpevo.FunctionFolder.USPEX.src.WriteBest import WriteBest
    from gpevo.FunctionFolder.USPEX.src.WriteGeneration import WriteGeneration

    WriteIndividual(ORG.resFolder)
    WriteBest(ORG.resFolder)
    WriteGeneration(ORG.resFolder)
    G["ORG_STRUC"] = ORG
