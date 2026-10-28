"""Bridge from gpevo to the Gaussian-process (GP) routines of ``uspexkit``.

Every uspexkit call goes through the Python API (``uspexkit.gp(...)``) instead
of a shell command.  Two call sites are covered:

* ``run_gp``            — the GP evaluation step that ``1 : gp`` appends after
  the relaxations (used to be a ``uspexkit gp`` line in commandExecutable).
* ``gp_active_learning``— port of the "GPML active learning with EI top-K"
  block of CalcFitness_310.m:114-422, which used to shell out to
  ``uspexkit traj`` / ``uspexkit calc`` / ``uspexkit pred``.  The traj
  conversion is gone: ``WriteIndividualOutput_310`` appends a frame to
  ``Individuals.traj`` for every structure it writes to gatheredPOSCARS.

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

# Files uspexkit.gp() reads from ../<data> (see core.py:196-198).
_DB_FILES = ("feature_mlp.csv", "feature.csv", "structures.traj")

# Minimum number of labelled rows before uspexkit can train: both gp() and
# pred() index the training array as data[:, 1:], which raises on a single row.
_MIN_DB_ROWS = 2


def gp_data_paths(ORG):
    """``(home, resFolder, data_dir)`` used by the GP training database.

    ``uspexkit.calc`` resolves the database as ``<parent of cwd>/<dat>`` and is
    called with ``cwd=resFolder``, while ``gp()`` uses ``../<dat>`` relative to
    a CalcFold -- both land on ``<home>/<dat>``.
    """
    home = getattr(ORG, "homePath", None) or os.getcwd()
    res_folder = getattr(ORG, "resFolder", None) or ""
    data_dir = os.path.join(home, str(_opt(ORG, "gpData", "data") or "data"))
    return home, res_folder, data_dir


def csv_rows(path):
    """Number of data rows of a uspexkit csv (first line is the header)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fp:
            lines = [line for line in fp if line.strip()]
    except OSError:
        return 0
    return max(0, len(lines) - 1)


def _db_rows(data_dir):
    """Labelled rows of the training database, 0 when it is unusable."""
    if not data_dir or not os.path.isdir(data_dir):
        return 0
    for name in _DB_FILES:
        if not os.path.isfile(os.path.join(data_dir, name)):
            return 0
    rows = min(csv_rows(os.path.join(data_dir, "feature_mlp.csv")),
               csv_rows(os.path.join(data_dir, "feature.csv")))
    return max(0, rows)


def gp_db_ready(ORG):
    """True when uspexkit.gp() has something to train on."""
    _, _, data_dir = gp_data_paths(ORG)
    return _db_rows(data_dir) >= _MIN_DB_ROWS


# (generation, data_dir) already reported to the user.  Module level, so a
# resumed run starts clean and nothing lands in the ORG pickle.
_WARNED_NO_DB = set()


def gp_db_unavailable(ORG, generation=None):
    """Explain once per generation why the GP step produces nothing.

    Before the first generation has been labelled there is nothing for
    ``gp()`` to fit, and without this guard every structure would print its own
    failure line.
    """
    _, _, data_dir = gp_data_paths(ORG)
    key = (int(generation or 0), data_dir)
    if key in _WARNED_NO_DB:
        return
    _WARNED_NO_DB.add(key)
    console_print(f"  GP step skipped: {data_dir} holds fewer than "
                  f"{_MIN_DB_ROWS} labelled rows.")
    console_print("                   the training set is seeded at the end of "
                  "the generation.")


def _invalidate_pkl(home, res_folder, data_dir):
    """Drop cached models so the next prediction retrains on the grown set.

    ``gp()`` fits from the CalcFold it runs in, ``pred()`` from the database
    directory, so the pickles end up in several places; leaving any of them
    behind means the next call silently reuses the old model.
    """
    directories = [data_dir, res_folder]
    directories += sorted(glob.glob(os.path.join(str(home), "CalcFold*")))
    removed = []
    for directory in directories:
        if not directory or not os.path.isdir(directory):
            continue
        for name in _PKL_NAMES:
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    removed.append(path)
                except OSError:
                    pass
    return removed


def _prune_stale_id_dirs(res_folder, ids):
    """Delete leftover ``<resFolder>/<id>`` directories with no result.

    ``uspexkit.calc`` skips every id whose directory already exists, so a
    failed reference calculation would blacklist that structure forever.
    """
    pruned = []
    for number in ids:
        directory = os.path.join(res_folder, str(int(number)))
        if not os.path.isdir(directory):
            continue
        if os.path.isfile(os.path.join(directory, "POSCAR.{}".format(int(number)))):
            continue                        # a completed run: keep its data
        try:
            shutil.rmtree(directory)
            pruned.append(directory)
        except OSError:
            pass
    return pruned


def _pick_by_fitness(entries, fitness, nboot):
    """The best *nboot* ``(index, number)`` entries, fitness ascending."""
    order = sorted(entries, key=lambda entry: fitness[entry[0]])
    return order[:max(0, int(nboot))]

def _uspexkit_func(name):
    """Return the ``uspexkit.<name>`` callable from the vendored ``gpevo.uspexkit``.

    Returns None when the package or the function is unavailable.  uspexkit is
    bundled with GPEvo (``gpevo/uspexkit``), so there is no standalone fallback.
    """
    try:
        from gpevo import uspexkit as _pkg
    except Exception as exc:
        console_print(f"  Warning: gpevo.uspexkit is not importable ({exc})")
        return None
    func = getattr(_pkg, name, None)
    if func is None:
        console_print(f"  Warning: gpevo.uspexkit.{name} is not available")
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
    # ``residual`` is the Hilbert-space distance to the training set; it is
    # stored under its own name as well as under ``distance``, which is the
    # field CalcFitness_310.m:47 looks up for optType 13.
    s.residual = _finite(result.get("residual"))
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
                    f"{'Density(GP)':>12s}{'UC':>10s}"
                    f"{'Residual':>10s}")


def apply_pred_metrics(s, entry):
    """Store what ``uspexkit.pred()`` returned for one structure.

    ``pred()`` returns its predictions, so the re-prediction after a DFT
    update goes straight onto the individual -- no round-trip through
    density_predict.log.
    """
    if not isinstance(entry, dict):
        return False
    # pred() returns 'residual' just like gp() does; store it so optType 13
    # stays usable after the training set has been enlarged.
    residual = _finite(entry.get("residual"))
    if not math.isnan(residual):
        s.residual = residual
        s.distance = residual
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
        # Structures labelled to seed an empty database; two is the minimum:
        # uspexkit indexes the training array as data[:, 1:], which raises on
        # a single row.
        "nboot": int(_opt(ORG, "gpNboot", 2)),
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


def _select_by_ei(numbers, table, params):
    """EI / uncertainty / random top-K selection of one generation.

    Returns ``(selected, mode_label, reason)``: *selected* holds
    ``(number, sigma, density_gp, ei)`` for every crystal that cleared all
    gates, and *reason* is a printable sentence naming the gate that stopped
    the selection (``None`` on success).
    """
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
        return [], "", (f"density threshold gate: f_best={f_best:.4f} < "
                        f"den={params['den']:.4f}")

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
        return [], "", "no crystal left after the residual filter"

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

    candidates = [(crystal_list[i], uncert_list[i], den_gp_list[i], ei_list[i])
                  for i in order[:n_sel]]
    above_u = [s for s in candidates if s[1] > params["u"]]
    if not above_u:
        return [], mode_label, (f"no crystal above the uncertainty threshold "
                                f"({params['u']:.4f}) among the top-{n_sel}")
    selected = [s for s in above_u if s[2] >= params["den"]]
    if not selected:
        return [], mode_label, (f"every top-{n_sel} crystal is below the density "
                                f"gate den={params['den']:.4f}")
    return selected, mode_label, None


def gp_active_learning(ORG, POP, fitness):
    """Label a few structures of this generation, then re-predict them.

    Port of CalcFitness_310.m:114-422 with two things the MATLAB version lacks:

    * **Bootstrap**: when the database is still empty the best structures of
      the generation are labelled straight away, so a run no longer depends on
      a ready-made ``<data>`` directory.
    * **Real retraining**: the pickled models are dropped everywhere
      ``uspexkit`` wrote them (they are keyed by a kernel version stamp and
      nothing else), so the next prediction trains on the enlarged set.

    ``fitness`` comes back unchanged whenever no training row was added; every
    gate that stops the loop says which one it was.
    """
    fitness = np.asarray(fitness, dtype=float)
    if not gp_enabled(ORG):
        return fitness
    valid = [i for i in range(len(POP.POPULATION)) if fitness[i] < 100000]
    if not valid:
        console_print("  No valid structures this generation: GP training set unchanged")
        return fitness

    params = parse_gp_params(ORG)
    home, res_folder, data_dir = gp_data_paths(ORG)
    if not res_folder or not os.path.isdir(res_folder):
        console_print("  Warning: resFolder is missing, GP training set unchanged")
        return fitness

    # Step: the trajectory already exists -- WriteIndividualOutput_310
    # appends a frame to Individuals.traj for every structure that goes into
    # gatheredPOSCARS, so no conversion step is needed here.
    traj_file = os.path.join(res_folder, "Individuals.traj")
    if not os.path.isfile(traj_file):
        console_print("  Warning: Individuals.traj is missing in "
                      f"{res_folder}; this run directory predates the "
                      "per-structure trajectory, GP training set unchanged")
        return fitness

    numbers = [int(getattr(POP.POPULATION[i], "Number", 0)) for i in valid]
    table = metrics_table(POP, valid, numbers, os.path.join(home, "CalcFold1", "gp.csv"))
    rows_before = _db_rows(data_dir)

    if rows_before >= _MIN_DB_ROWS and table:
        selected, mode_label, reason = _select_by_ei(numbers, table, params)
        if reason is not None:
            console_print(f"  {reason}; GP training set unchanged")
            return fitness
        pred_ids = [n for n in numbers if n in table and table[n][1] <= 10]
    else:
        # Nothing to learn from yet: seed the database instead of waiting for a
        # generation that carries GP predictions.
        if rows_before < _MIN_DB_ROWS:
            console_print(f"  GP training set has only {rows_before} labelled "
                          f"row(s) in {data_dir}: seeding it with the best "
                          f"structures of generation "
                          f"{int(getattr(POP, 'generation', 0))}")
        else:
            console_print("  No GP metrics this generation, seeding the training "
                          "set with its best structures")
        entries = _pick_by_fitness(list(zip(valid, numbers)), fitness, params["nboot"])
        selected = [(number, float("nan"), float("nan"), float("nan"))
                    for _, number in entries]
        pred_ids = [number for _, number in entries]
        mode_label = "fitness (bootstrap)"

    if not selected:
        console_print("  Nothing selected for the reference calculation: "
                      "GP training set unchanged")
        return fitness

    if mode_label.endswith("(bootstrap)"):
        console_print(f"  Bootstrap: the best {len(selected)} structure(s) go to "
                      "the reference calculation")
        for rank, entry in enumerate(selected, start=1):
            number = entry[0]
            index = valid[numbers.index(number)]
            console_print(f"  {rank}: crystal {number}  fitness={fitness[index]:.6f}")
    else:
        console_print(f"{mode_label} top-{int(params['k'])} selection: "
                      f"{len(selected)} crystals above threshold "
                      f"({params['u']:.4f})")
        for rank, (number, sigma, mu, ei) in enumerate(selected, start=1):
            console_print(f"  {rank}: crystal {number}  EI={ei:.6f}  den_gp={mu:.4f}"
                          f"  sigma={sigma:.4f}")
    _write_ei_log(ORG, POP.generation, [s[0] for s in selected])

    ids_list = [int(s[0]) for s in selected]
    stale_dirs = _prune_stale_id_dirs(res_folder, ids_list)
    if stale_dirs:
        console_print("  Removed {} result folder(s) of a previous aborted "
                      "reference run: {}".format(
                          len(stale_dirs),
                          ", ".join(os.path.basename(p) for p in stale_dirs)))

    # calc() chdirs into <parent of cwd>/<dat> without creating it.
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError as exc:
        console_print(f"  Warning: cannot create GP data directory {data_dir}: {exc}")
        return fitness

    crystal_ids = " ".join(str(i) for i in ids_list)
    console_print(f"  Running: uspexkit.calc(ids='{crystal_ids}')")
    ok, _unused = _call("calc", cwd=res_folder, t="Individuals.traj", den=params["den"],
                        ids=crystal_ids, step=500, ncpu=int(params["ncpu"]),
                        dat=params["dat"], tolerance=0.01)
    if not ok:
        console_print("  Warning: uspexkit.calc failed: the reference calculation "
                      "needs `siesta` on PATH and <homePath>/Specific/*.psf")
        return fitness

    rows_after = _db_rows(data_dir)
    if rows_after <= rows_before:
        console_print(f"  Warning: uspexkit.calc ran {len(ids_list)} structure(s) but "
                      f"the training set is still {rows_after} row(s) in {data_dir};")
        console_print("           they were either already in the database, or the "
                      "reference calculation produced no row")
        return fitness

    # Step: retrain on the enlarged database -- the pickles have to go from
    # every directory uspexkit may have cached them in.
    dropped = _invalidate_pkl(home, res_folder, data_dir)
    console_print(f"  GP training set: {rows_before} -> {rows_after} rows; "
                  f"dropped {len(dropped)} cached model(s)")

    if not pred_ids:
        console_print("  No valid crystals with residual <= 10, skipping pred")
        return fitness

    pred_log = os.path.join(res_folder, "density_predict.log")
    if os.path.isfile(pred_log):
        os.remove(pred_log)
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
