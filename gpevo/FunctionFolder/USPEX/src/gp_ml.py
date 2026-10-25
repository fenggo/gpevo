"""Bridge from gpevo to the Gaussian-process (GP) routines of ``uspexkit``.

Every uspexkit call goes through the Python API (``uspexkit.gp(...)``) instead
of a shell command.  Two call sites are covered:

* ``run_gp``            — the GP evaluation step that ``1 : gp`` appends after
  the relaxations (used to be a ``uspexkit gp`` line in commandExecutable).
* ``gp_active_learning``— port of the "GPML active learning with EI top-K"
  block of CalcFitness_310.m:114-422, which used to shell out to
  ``uspexkit traj`` / ``uspexkit calc`` / ``uspexkit pred``.

Both are configured from INPUT.txt: ``1 : gp`` (switch), ``data : <dir>`` and
``den : <density>``.

gp() reads and writes relative paths (``gulp.cif``, ``output``, ``geometry``,
``gp.csv``, ``../<dat>/feature_mlp.csv``), so it must run with the CalcFold
directory as the working directory.  It also returns the very numbers it
appends to gp.csv, which is what gpevo now consumes instead of the file.
"""
from __future__ import annotations

import csv
import glob
import math
import os
import random
import shutil

import numpy as np

from gpevo.log_utils import console_only, console_print

# Active-learning defaults of CalcFitness_310.m:124-129, used when the ORG has
# no INPUT.txt value (older pickles).
_EI_DEFAULTS = {"u": 0.04, "k": 1, "mode": "ei"}

_PKL_NAMES = ("gpr_density.pkl", "gpr_energy.pkl", "rfr.pkl")


def _uspexkit_func(name):
    """Return the ``uspexkit.<name>`` callable, or None when unavailable."""
    try:
        import uspexkit
    except Exception as exc:
        console_print(f"  Warning: uspexkit is not importable ({exc})")
        return None
    func = getattr(uspexkit, name, None)
    if func is None:                       # install without the top-level alias
        try:
            import uspexkit.core as _core
            func = getattr(_core, name, None)
        except Exception:
            func = None
    if func is None:
        console_print(f"  Warning: uspexkit.{name} is not available")
    return func


def _finite(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _call(name, cwd=None, **kwargs):
    """Call ``uspexkit.<name>(**kwargs)``, optionally from *cwd*.

    Returns ``(ok, result)``: *result* is whatever the uspexkit function
    returned (gp() returns its metrics dict).  uspexkit prints its own
    diagnostics with ``print``; ``console_only`` keeps them on the terminal,
    exactly as they were when it ran as a subprocess.
    """
    func = _uspexkit_func(name)
    if func is None:
        return False, None
    previous = os.getcwd()
    try:
        if cwd:
            os.chdir(cwd)
        with console_only():
            result = func(**kwargs)
    except Exception as exc:
        console_print(f"  Warning: uspexkit.{name} failed: {exc}")
        return False, None
    finally:
        try:
            os.chdir(previous)
        except OSError:
            pass
    return True, result


def _opt(ORG, name, default):
    """INPUT option value, or *default* when it was never set."""
    value = getattr(ORG, name, None)
    return default if value is None else value


def _ncpu(ORG):
    """Core count for the GP step: ``24 : ncpu`` else ``% numProcessors``."""
    value = getattr(ORG, "gpNcpu", 0)
    if value:
        return int(value)
    procs = getattr(ORG, "numProcessors", None)
    if isinstance(procs, (list, tuple)) and procs:
        try:
            return int(procs[-1])
        except (TypeError, ValueError):
            pass
    return 8


def run_gp(ORG, cwd=None, id_=None):
    """Run ``uspexkit.gp(...)``; returns its metrics dict (None on failure)."""
    kwargs = {
        "n": _ncpu(ORG),
        "dat": str(getattr(ORG, "gpData", "data") or "data"),
        "den": float(getattr(ORG, "gpDen", 0.0) or 0.0),
        "optype": int(getattr(ORG, "optType", 11) or 11),
    }
    if id_ is not None:
        kwargs["id_"] = int(id_)
    ok, result = _call("gp", cwd=cwd, **kwargs)
    return result if ok and isinstance(result, dict) else None


def apply_gp_metrics(s, result):
    """Store what ``uspexkit.gp()`` returned on one structure.

    This replaces the old round-trip through CalcFold1/gp.csv: the same three
    quantities that were read back from columns 3/6/7 (residual, density_gp,
    uncertainty) are now taken straight from the return value.
    """
    if not isinstance(result, dict):
        return False
    s.distance = _finite(result.get("residual"))
    s.density_gp = _finite(result.get("density_gp"))
    s.uq = _finite(result.get("uncertainty"))
    # Energy prediction and its uncertainty, for the OUTPUT.txt table.
    s.energy_gp = _finite(result.get("energy_pred"))
    s.uq_eng = _finite(result.get("uncertainty_eng"))
    s.gp = dict(result)
    return True


# Columns appended after SYMMETRY in the OUTPUT.txt table when `1 : gp` is
# set: the GP-predicted enthalpy with its uncertainty (UC), then the
# GP-predicted density with its uncertainty.  The widths (12/10/12/10) match
# the value columns of WriteGenerationOutput_310._gp_columns.
GP_OUTPUT_HEADER = (f"{'Enthalpy(GP)':<12s}{'UC':>10s}"
                    f"{'Density(GP)':>12s}{'UC':>10s}")


def apply_pred_metrics(s, entry):
    """Store what ``uspexkit.pred()`` returned for one structure.

    ``pred()`` returns its predictions, so the re-prediction after a DFT
    update goes straight onto the individual -- no round-trip through
    density_predict.log.
    """
    if not isinstance(entry, dict):
        return False
    s.density_gp = _finite(entry.get("density_gp"))
    s.uq = _finite(entry.get("uncertainty"))
    s.energy_gp = _finite(entry.get("energy_pred"))
    s.uq_eng = _finite(entry.get("uncertainty_eng"))
    s.gp_pred = dict(entry)
    return True


def _metrics_row(s):
    """Row layout of ``_read_gp_csv``, built from stored per-structure values."""
    return [float("nan"), _finite(getattr(s, "distance", None)), float("nan"),
            float("nan"), _finite(getattr(s, "density_gp", None)),
            _finite(getattr(s, "uq", None))]


def metrics_table(POP, valid, numbers, gp_file):
    """``{Number: row}`` from the values gp() returned; gp.csv as fallback.

    The file is only consulted when no structure carries the returned values
    (e.g. a run resumed after the switch, or carried out by the CLI).
    """
    table = {}
    for index, number in zip(valid, numbers):
        row = _metrics_row(POP.POPULATION[index])
        if math.isfinite(row[1]) or math.isfinite(row[4]):
            table[number] = row
    if not table:
        table = _read_gp_csv(gp_file)
    return table


def parse_gp_params(ORG):
    """Active-learning parameters from the GP options of INPUT.txt.

    Port of CalcFitness_310.m:123-177.  Options: ``1 : gp``, ``24 : ncpu``,
    ``data : <dir>``, ``den : <density>``, ``u``, ``k``, ``mode``.
    """
    return {
        "ncpu": _ncpu(ORG),
        "dat": str(_opt(ORG, "gpData", "data")),
        "den": float(_opt(ORG, "gpDen", 0.0)),
        "u": float(_opt(ORG, "gpU", _EI_DEFAULTS["u"])),
        "k": int(_opt(ORG, "gpK", _EI_DEFAULTS["k"])),
        "mode": str(_opt(ORG, "gpMode", _EI_DEFAULTS["mode"])),
    }


def gp_enabled(ORG):
    """True when INPUT.txt enabled the GP with ``1 : gp``."""
    return int(getattr(ORG, "gp", 0) or 0) == 1


def is_gp_step(ORG, step):
    """True for the extra GP step appended after the relaxation steps."""
    return gp_enabled(ORG) and int(step) == len(getattr(ORG, "abinitioCode", []) or [])


def _read_gp_csv(path):
    """Read CalcFold1/gp.csv into ``{crystal_id: row}`` (first row wins).

    Columns (CalcFitness_310.m:183-189):
    0 id, 1 nearest index, 2 residual, 3 density_min, 4 density_rf,
    5 density_gp, 6 uncertainty, 7 energy_min, 8 energy_pred, 9 uncertainty_eng
    """
    table = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fp:
            reader = csv.reader(fp)
            next(reader, None)
            for row in reader:
                if len(row) < 7:
                    continue
                try:
                    key = int(float(row[0]))
                    values = [float(value) for value in row[1:7]]
                except (TypeError, ValueError):
                    continue
                table.setdefault(key, values)
    except OSError:
        return {}
    return table


def _write_ei_log(ORG, generation, crystals):
    """Append the EI-selected crystal ids to ``<homePath>/ei.log``."""
    path = os.path.join(getattr(ORG, "homePath", os.getcwd()), "ei.log")
    ids = " ".join(str(int(c)) for c in crystals)
    try:
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(f"gen {int(generation)}:{ids}\n")
    except OSError:
        pass


def _mol_factor(ORG, s):
    """The ``numMols`` normalisation CalcFitness_310 applies for optType 1."""
    try:
        factor = sum(map(int, s.numMols)) / max(1, sum(map(int, ORG.numMols)))
    except Exception:
        return 1.0
    return factor if factor > 0 else 1.0


def _expected_improvement(mu, sigma, f_best):
    """EI for maximisation: (mu-f_best)*Phi(Z) + sigma*phi(Z)."""
    diff = mu - f_best
    if sigma < 1e-10:
        return max(0.0, diff)
    z = diff / sigma
    phi_z = 0.5 * math.erfc(-z / math.sqrt(2.0))
    pdf_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return diff * phi_z + sigma * pdf_z


def gp_active_learning(ORG, POP, fitness):
    """EI top-K active learning: DFT the uncertain candidates, then re-predict.

    Port of CalcFitness_310.m:114-422.  ``fitness`` is returned unchanged when
    the generation is the first one, when nothing is valid, or when neither the
    structures nor ``CalcFold1/gp.csv`` carry any GP metrics.
    """
    fitness = np.asarray(fitness, dtype=float)
    if not gp_enabled(ORG) or int(getattr(POP, "generation", 0)) <= 1:
        return fitness
    valid = [i for i in range(len(POP.POPULATION)) if fitness[i] < 100000]
    if not valid:
        return fitness

    params = parse_gp_params(ORG)
    home_path = getattr(ORG, "homePath", os.getcwd())
    gp_file = os.path.join(home_path, "CalcFold1", "gp.csv")
    numbers = [int(getattr(POP.POPULATION[i], "Number", 0)) for i in valid]
    table = metrics_table(POP, valid, numbers, gp_file)
    if not table:
        return fitness

    # Step 0: f_best = max density_gp over the valid crystals (residual <= 10)
    f_best = -float("inf")
    filtered = 0
    for number in numbers:
        row = table.get(number)
        if row is None:
            continue
        if row[1] > 10:                     # residual
            filtered += 1
            continue
        f_best = max(f_best, row[4])        # density_gp
    if filtered > 0:
        console_print(f"  Filtered out {filtered} crystals with residual > 10")

    if f_best < params["den"]:
        console_print(f"  density threshold gate: f_best={f_best:.4f} < "
                      f"den={params['den']:.4f}, skipping GP/DFT/pred")
        return fitness

    # Step 1: EI of every valid crystal
    ei_list, crystal_list, uncert_list, den_gp_list = [], [], [], []
    for number in numbers:
        row = table.get(number)
        if row is None or row[1] > 10:
            continue
        mu, sigma = row[4], row[5]
        ei_list.append(_expected_improvement(mu, sigma, f_best))
        crystal_list.append(number)
        uncert_list.append(sigma)
        den_gp_list.append(mu)
    if not ei_list:
        return fitness

    # Step 2: select top-K
    n_sel = min(int(params["k"]), len(ei_list))
    mode = str(params["mode"])
    if mode == "uncertainty":
        order = sorted(range(len(ei_list)), key=lambda i: -uncert_list[i])
        mode_label = "uncertainty"
    elif mode == "random":
        order = list(range(len(ei_list)))
        random.shuffle(order)
        mode_label = "random"
    elif mode == "ei":
        order = sorted(range(len(ei_list)), key=lambda i: -ei_list[i])
        mode_label = "EI"
    else:
        console_print(f"  Unknown select_mode \"{mode}\", falling back to EI")
        order = sorted(range(len(ei_list)), key=lambda i: -ei_list[i])
        mode_label = "EI"

    selected = [(crystal_list[i], uncert_list[i], den_gp_list[i], ei_list[i])
                for i in order[:n_sel]]
    selected = [s for s in selected if s[1] > params["u"]]
    selected = [s for s in selected if s[2] >= params["den"]]

    if not selected:
        console_print(f"  -> No crystals above uncertainty threshold "
                      f"({params['u']:.4f}), skipping DFT and pred")
        return fitness

    console_print(f"{mode_label} top-{int(params['k'])} selection: "
                  f"{len(selected)}/{n_sel} crystals above threshold "
                  f"({params['u']:.4f})")
    for rank, (number, sigma, mu, ei) in enumerate(selected, start=1):
        console_print(f"  {rank}: crystal {number}  EI={ei:.6f}  den_gp={mu:.4f}"
                      f"  sigma={sigma:.4f}  f_best={f_best:.4f}")
    _write_ei_log(ORG, POP.generation, [s[0] for s in selected])

    res_folder = getattr(ORG, "resFolder", None)
    if not res_folder or not os.path.isdir(res_folder):
        console_print("  Warning: resFolder is missing, skipping traj/calc/pred")
        return fitness
    crystal_ids = " ".join(str(int(s[0])) for s in selected)

    # Step 3: trajectory for the whole generation (once)
    console_print(f"  Running: uspexkit.traj() in {res_folder}")
    if not _call("traj", cwd=res_folder, fposcar="gatheredPOSCARS")[0]:
        return fitness

    # Step 4: DFT on the selected crystals
    console_print(f"  Running: uspexkit.calc(ids='{crystal_ids}')")
    if not _call("calc", cwd=res_folder, t="Individuals.traj", den=1.88,
                 ids=crystal_ids, step=500, ncpu=int(params["ncpu"]),
                 dat=params["dat"], tolerance=0.01)[0]:
        return fitness

    # Step 5: retrain the GP on the enlarged database and re-predict
    pred_ids = [number for number in numbers
                if number in table and table[number][1] <= 10]
    if not pred_ids:
        console_print("  No valid crystals with residual <= 10, skipping pred")
        return fitness

    pred_log = os.path.join(res_folder, "density_predict.log")
    if os.path.isfile(pred_log):
        os.remove(pred_log)
    # Force a retrain: the pickled models move to CalcFold1 and the data
    # directory loses them (CalcFitness_310.m:361).
    data_dir = os.path.join(home_path, str(params["dat"]))
    calc_fold = os.path.join(home_path, "CalcFold1")
    for src in glob.glob(os.path.join(data_dir, "*.pkl")):
        try:
            shutil.copy2(src, calc_fold)
        except OSError:
            pass
    for name in _PKL_NAMES:
        try:
            os.remove(os.path.join(data_dir, name))
        except OSError:
            pass

    console_print(f"  Running: uspexkit.pred(ids='{' '.join(str(i) for i in pred_ids)}')")
    ok, results = _call("pred", cwd=res_folder, t="Individuals.traj", step=100,
                        ncpu=int(params["ncpu"]), den=0.0,
                        ids=" ".join(str(i) for i in pred_ids), dat=params["dat"])
    if not ok or not isinstance(results, dict) or not results:
        console_print("  Warning: uspexkit.pred returned no predictions, "
                      "fitness left unchanged")
        return fitness

    # Store the re-predictions on the structures; the fitness below then reads
    # them back from memory.
    for index, number in zip(valid, numbers):
        entry = results.get(number)
        if entry is not None:
            apply_pred_metrics(POP.POPULATION[index], entry)

    opt_type = int(getattr(ORG, "optType", 1) or 1)
    # CalcFitness has already applied opt_sign; the replacement below has to
    # carry it too (it is 1 for a minimisation run, -1 for `-N : optType`).
    opt_sign = float(getattr(ORG, "opt_sign", 1) or 1)

    updated = 0
    skipped = 0
    for index, number in zip(valid, numbers):
        row = table.get(number)
        if row is not None and row[1] > 10:
            skipped += 1
            continue
        if opt_type == 1:
            # Enthalpy target: use the GP-predicted energy, normalised the
            # same way CalcFitness_310 does (enthalpy / factor).  Previously
            # this wrote -density for every optType, which silently turned a
            # minimisation of the enthalpy into a maximisation of the density.
            value = getattr(POP.POPULATION[index], "energy_gp", None)
            if value is None or not math.isfinite(float(value)):
                continue
            factor = _mol_factor(ORG, POP.POPULATION[index])
            fitness[index] = opt_sign * (float(value) / max(1e-12, factor))
        elif opt_type == 11:
            value = getattr(POP.POPULATION[index], "density_gp", None)
            if value is None or not math.isfinite(float(value)):
                continue
            fitness[index] = opt_sign * (-1.0 * float(value))
        else:
            # optType 12/13 mix density with other quantities (entropy,
            # distance): no single ML value can replace the fitness, so keep
            # what CalcFitness computed.
            continue
        updated += 1
    console_print(f"  Updated fitness for {updated}/{len(valid)} valid crystals "
                  f"(skipped {skipped} with residual > 10)")
    return fitness
