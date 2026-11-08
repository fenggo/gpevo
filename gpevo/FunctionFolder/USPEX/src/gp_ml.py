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
import pickle
import random
import shutil

import numpy as np

from gpevo.log_utils import console_only, console_print
from gpevo.runtime import RESIDUAL_CUT

# Active-learning defaults of CalcFitness_310.m:124-129, used when the ORG has
# no INPUT.txt value (older pickles).
_EI_DEFAULTS = {"u": 0.04, "k": 1, "mode": "ei"}

# uspexkit reports the 95% confidence HALF-width 1.96*std in the
# ``uncertainty`` / ``uncertainty_eng`` fields (core.py:342,374).  EI's closed
# form needs the posterior standard deviation sigma, so the Python layer keeps
# both: ``s.uq`` stays 1.96*sigma for OUTPUT.txt / error bars / the ``u`` gate,
# ``s.sigma_gp`` is what the acquisition actually consumes.
UNCERT_SCALE = 1.96

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

    Resolve from ``core`` first.  ``gpevo/uspexkit/gp.py`` is a submodule whose
    name collides with ``core.gp()``: importing it (the gpytorch branch, or
    unpickling a cached GPyTorchRegressor) sets ``pkg.gp`` to the *module*,
    shadowing the PEP-562 ``__getattr__`` in ``__init__.py``.  A second
    uspexkit.gp() call then fails with "'module' object is not callable".
    """
    try:
        from gpevo.uspexkit import core as _core
    except Exception as exc:
        console_print(f"  Warning: gpevo.uspexkit is not importable ({exc})")
        return None
    func = getattr(_core, name, None)
    if func is None:
        # Names __init__.py maps to other modules (denevo, smd, md2pdf).
        from gpevo import uspexkit as _pkg
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


def _gp_backend(ORG):
    """GP training backend of INPUT.txt (``gpbackend``); sklearn by default."""
    return str(_opt(ORG, "gpBackend", "sklearn")).lower()


def _backend_ready(backend):
    """Fail fast when gpytorch is selected but not installed.

    ``_call`` swallows exceptions raised inside uspexkit, so a missing backend
    has to be caught *before* it -- otherwise a ``gpbackend : gpytorch`` run
    would silently produce sklearn results under the gpytorch setting.
    """
    if backend != "gpytorch":
        return True
    import importlib.util
    if importlib.util.find_spec("gpytorch") is not None:
        return True
    console_print("  ERROR: gpbackend : gpytorch selected but gpytorch is not installed;")
    console_print("         install it with `pip install gpytorch` (or `pip install gpevo[gpytorch]`).")
    console_print("         GP step skipped; NOT falling back to sklearn silently.")
    return False


def run_gp(ORG, cwd=None, id_=None):
    """Run ``uspexkit.gp(...)``; returns its metrics dict (None on failure)."""
    backend = _gp_backend(ORG)
    if not _backend_ready(backend):
        return None
    kwargs = {
        "n": _ncpu(ORG),
        "dat": str(getattr(ORG, "gpData", "data") or "data"),
        "den": float(getattr(ORG, "gpDen", 0.0) or 0.0),
        "optype": int(getattr(ORG, "optType", 11) or 11),
        "backend": backend,
        "training_iter": int(_opt(ORG, "gpTrainingIter", 200)),
        "learning_rate": float(_opt(ORG, "gpLr", 0.1)),
        "log_dir": getattr(ORG, "resFolder", None) or None,
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
    s.sigma_gp = _to_sigma(result.get("uncertainty"))
    # Energy prediction and its uncertainty, for the OUTPUT.txt table.
    s.energy_gp = _finite(result.get("energy_pred"))
    s.uq_eng = _finite(result.get("uncertainty_eng"))
    s.sigma_eng = _to_sigma(result.get("uncertainty_eng"))
    s.gp = dict(result)
    feature = result.get("feature")
    if isinstance(feature, (list, tuple)) and len(feature) == _FEATURE_LEN:
        s.gp_feature = [float(value) for value in feature]
        _cache_feature(getattr(s, "Number", None), s.gp_feature)
    return True


# The 10-D query feature gp() computed for each structure of the current
# generation, keyed by its global id.  The GP step and gp_active_learning run
# in the same process (SubmitJobs_310 dispatches jobs sequentially), so pred()
# can reuse them instead of re-running ReaxFF.  Pruned to one generation by
# gp_active_learning.
_FEATURE_LEN = 10
_FEATURE_CACHE = {}


def _features_reusable(ORG):
    """INPUT switch ``cache : 0`` falls back to recomputing every feature."""
    return int(_opt(ORG, "gpCache", 1) or 0) != 0


def _cache_feature(number, feature):
    """Remember gp()'s feature for *number* (None or NaN entries ignored)."""
    try:
        key = int(number)
    except (TypeError, ValueError):
        return
    if key <= 0 or not all(math.isfinite(value) for value in feature):
        return
    _FEATURE_CACHE[key] = list(feature)


def _prune_feature_cache(numbers):
    """Drop everything that is not part of the current generation."""
    keep = set(numbers)
    for key in [k for k in _FEATURE_CACHE if k not in keep]:
        del _FEATURE_CACHE[key]


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
    s.sigma_gp = _to_sigma(entry.get("uncertainty"))
    s.energy_gp = _finite(entry.get("energy_pred"))
    s.uq_eng = _finite(entry.get("uncertainty_eng"))
    s.sigma_eng = _to_sigma(entry.get("uncertainty_eng"))
    s.gp_pred = dict(entry)
    return True


# Row layout shared by ``_metrics_row`` and ``_read_gp_csv`` (the gp.csv
# columns of uspexkit core.py:223-224 with the leading id stripped):
#   0 nearest, 1 residual, 2 density_min, 3 density_rf, 4 density_gp,
#   5 sigma (posterior std), 6 energy_min, 7 energy_pred, 8 sigma_energy
# Both readers convert the 1.96*std columns of gp.csv / gp() into sigma.
_DEN_GP, _UNCERT = 4, 5
_ENG_GP, _UNCERT_ENG = 7, 8
_ROW_LEN = 9


def _to_sigma(uncertainty):
    """Convert uspexkit's 1.96*std half-width into a posterior std."""
    uq = _finite(uncertainty)
    return uq / UNCERT_SCALE if math.isfinite(uq) else float("nan")


def _metrics_row(s):
    """Row layout of ``_read_gp_csv``, columns 5/8 are posterior sigma."""
    sigma = _finite(getattr(s, "sigma_gp", None))
    if not math.isfinite(sigma):
        sigma = _to_sigma(getattr(s, "uq", None))
    sigma_eng = _finite(getattr(s, "sigma_eng", None))
    if not math.isfinite(sigma_eng):
        sigma_eng = _to_sigma(getattr(s, "uq_eng", None))
    return [float("nan"), _finite(getattr(s, "distance", None)), float("nan"),
            float("nan"), _finite(getattr(s, "density_gp", None)),
            sigma, float("nan"),
            _finite(getattr(s, "energy_gp", None)),
            sigma_eng]


def metrics_table(POP, valid, numbers, gp_file):
    """``{Number: row}`` from the values gp() returned; gp.csv as fallback.

    The file is only consulted when no structure carries the returned values
    (e.g. a run resumed after the switch, or carried out by the CLI).
    """
    table = {}
    for index, number in zip(valid, numbers):
        row = _metrics_row(POP.POPULATION[index])
        if (math.isfinite(row[1]) or math.isfinite(row[_DEN_GP])
                or math.isfinite(row[_ENG_GP])):
            table[number] = row
    if not table:
        table = _read_gp_csv(gp_file)
    return table


def parse_gp_params(ORG):
    """Active-learning parameters from the GP options of INPUT.txt.

    Port of CalcFitness_310.m:123-177.  Options: ``1 : gp``, ``24 : ncpu``,
    ``data : <dir>``, ``den : <density>``, ``u``, ``k``, ``mode``, ``nboot``,
    ``cache`` and ``explore``.
    """
    return {
        "ncpu": _ncpu(ORG),
        "dat": str(_opt(ORG, "gpData", "data")),
        "den": float(_opt(ORG, "gpDen", 0.0)),
        # ``u`` is a floor on the REPORTED 1.96*std uncertainty (gp.csv
        # column 6/10), not on the posterior std the EI formula uses.
        "u": float(_opt(ORG, "gpU", _EI_DEFAULTS["u"])),
        "k": int(_opt(ORG, "gpK", _EI_DEFAULTS["k"])),
        "mode": str(_opt(ORG, "gpMode", _EI_DEFAULTS["mode"])),
        # The sigma*phi(Z) term of EI.  On (the default) it explores, which is
        # what makes the acquisition an *expected improvement* at all; off it
        # ranks purely by the improvement term, so a wide posterior on a bad
        # crystal can no longer buy it a label.
        "explore": int(_opt(ORG, "gpExplore", 1) or 0) != 0,
        # Structures labelled to seed an empty database; two is the minimum:
        # uspexkit indexes the training array as data[:, 1:], which raises on
        # a single row.
        "nboot": int(_opt(ORG, "gpNboot", 2)),
        # GP training backend (INPUT.txt gpbackend / gpiter / gplr).  Passed to
        # both gp() and pred() so one generation never mixes two backends.
        "backend": _gp_backend(ORG),
        "training_iter": int(_opt(ORG, "gpTrainingIter", 200)),
        "learning_rate": float(_opt(ORG, "gpLr", 0.1)),
        # optType 1 minimises the enthalpy, so the acquisition runs on the GP
        # energy; every other optType keeps the density.
        "optType": int(getattr(ORG, "optType", 11) or 11),
        # 1 for a minimisation run, -1 for `-N : optType` (maximisation).
        "optSign": float(getattr(ORG, "opt_sign", 1) or 1),
    }


def gp_enabled(ORG):
    """True when INPUT.txt enabled the GP with ``1 : gp``."""
    return int(getattr(ORG, "gp", 0) or 0) == 1


def gp_plot_enabled(ORG):
    """True when INPUT.txt enabled the per-generation GP status figure.

    Off by default (``0 : gpplot`` or the option absent): no ``gp_gen<N>.png``
    is written, but ``gp_status_history.csv`` and the terminal GP status
    report still are, so the run keeps its diagnostics.
    """
    return int(getattr(ORG, "gpPlot", 0) or 0) == 1


def is_gp_step(ORG, step):
    """True for the extra GP step appended after the relaxation steps."""
    return gp_enabled(ORG) and int(step) == len(getattr(ORG, "abinitioCode", []) or [])


def _read_gp_csv(path):
    """Read CalcFold1/gp.csv into ``{crystal_id: row}`` (first row wins).

    Columns (CalcFitness_310.m:183-189):
    0 id, 1 nearest index, 2 residual, 3 density_min, 4 density_rf,
    5 density_gp, 6 uncertainty, 7 energy_min, 8 energy_pred, 9 uncertainty_eng

    The id is stripped, so the stored row starts at the nearest index (see
    ``_ROW_LEN``).  Columns 5/8 are converted from the stored 1.96*std to the
    posterior std.  Short or unreadable cells become NaN: an old gp.csv whose
    two last columns are still empty must not cost the whole row.
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
                except (TypeError, ValueError):
                    continue
                values = [_finite(value) for value in row[1:_ROW_LEN + 1]]
                values += [float("nan")] * (_ROW_LEN - len(values))
                # gp.csv stores 1.96*std in columns 6/10 (see header above);
                # the table's sigma columns must hold the posterior std.
                for col in (_UNCERT, _UNCERT_ENG):
                    if math.isfinite(values[col]):
                        values[col] /= UNCERT_SCALE
                table.setdefault(key, values)
    except OSError:
        return {}
    return table


def _write_ei_log(ORG, generation, crystals):
    """Append the EI-selected crystal ids to ``<resFolder>/ei.log``.

    The log belongs to the run that produced it, so it sits in that run's
    ``results<N>`` next to ``Individuals`` and the density logs, not in the
    home directory.  A following run opens its own ``results<N+1>`` and
    therefore starts with an empty log instead of inheriting the selections
    of the previous one.
    """
    res = (getattr(ORG, "resFolder", "") or getattr(ORG, "homePath", "")
           or os.getcwd())
    path = os.path.join(res, "ei.log")
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


def _expected_improvement(mu, sigma, f_best, explore=True):
    """EI for maximisation: (mu-f_best)*Phi(Z) + sigma*phi(Z).

    With ``explore`` off the second term is dropped.  That term is the only
    part that rewards a crystal for being *unknown*: a structure the GP has
    never seen anything like carries a wide posterior, so sigma*phi(Z) can
    outrank the improvement term even when mu is far worse than the
    incumbent -- which is how low-quality crystals get sent to an expensive
    reference calculation.  What is left is pure exploitation, the expected
    size of the improvement weighted by the probability of improving at all.
    """
    diff = mu - f_best
    if sigma < 1e-10:
        return max(0.0, diff)
    z = diff / sigma
    phi_z = 0.5 * math.erfc(-z / math.sqrt(2.0))
    if not explore:
        return diff * phi_z
    pdf_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return diff * phi_z + sigma * pdf_z


def _ei_target(params):
    """``(mu column, sigma column, name)`` of the quantity EI is computed on.

    optType 1 minimises the enthalpy, so the acquisition has to run on the GP
    energy prediction and on *its* uncertainty (gp.csv columns 8/9); every
    other optType keeps the density (columns 5/6), as the MATLAB reference
    does.
    """
    if int(params.get("optType", 11) or 11) == 1:
        return _ENG_GP, _UNCERT_ENG, "eng_gp"
    return _DEN_GP, _UNCERT, "den_gp"


def _usable(numbers, table, mu_col, sigma_col):
    """True when at least one valid crystal carries both predictions."""
    for number in numbers:
        row = table.get(number)
        if row is None or row[1] > RESIDUAL_CUT:
            continue
        if math.isfinite(row[mu_col]) and math.isfinite(row[sigma_col]):
            return True
    return False


def _best_observed(data_dir, column, sign):
    """Best observed physical value of one ``feature.csv`` column.

    ``sign < 0`` means the objective minimises that column (energy), so the
    best is the minimum; otherwise it is the maximum (density).  Returns the
    raw physical value, not the sign-flipped EI-space one.
    """
    if not data_dir:
        return None
    path = os.path.join(data_dir, "feature.csv")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fp:
            reader = csv.reader(fp)
            header = next(reader, None)
            if header is None:
                return None
            names = [t.strip() for t in header]
            try:
                col = names.index(column)
            except ValueError:
                return None
            values = []
            for row in reader:
                cell = row[col].strip() if col < len(row) else ""
                if cell:
                    try:
                        values.append(float(cell))
                    except ValueError:
                        continue
    except OSError:
        return None
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.min(arr)) if sign < 0 else float(np.max(arr))


def _observed_incumbent(data_dir, target, sign):
    """Best observed (DFT) value from ``<data_dir>/feature.csv``, in EI space.

    ``feature.csv`` holds the labelled ground truth the GP was trained on:
    ``etot`` (total energy) and ``density`` are the DFT-computed labels
    (uspexkit/core.py appends ``energy`` / ``density_`` there).  The chief
    defect of the previous EI was that ``f_best`` came from the GP's own
    prediction; a wrong incumbent shifts every Z score and biases the pick
    toward exploitation.  The honest incumbent is the observed value.

    Returns the incumbent in the *EI target space* (``sign * value``), or
    ``None`` when the file is missing, empty or lacks the needed column, in
    which case the caller falls back to the predicted value.
    """
    column = "etot" if target == "eng_gp" else "density"
    best = _best_observed(data_dir, column, sign)
    return None if best is None else sign * best


def observed_density_incumbent(ORG):
    """Raw max observed DFT density for the status report's EI preview.

    ``_ei_preview`` ranks on GP-predicted density, so it must use the same
    incumbent as ``_select_by_ei``; otherwise the report shows an EI top-1 the
    real selection never considered (ct22/results2: 1.8032 vs 1.8991).
    """
    _, _, data_dir = gp_data_paths(ORG)
    return _best_observed(data_dir, "density", 1)


def _select_by_ei(numbers, table, params, data_dir=None):
    """EI / uncertainty / random top-K selection of one generation.

    Returns ``(selected, mode_label, reason, target)``: *selected* holds
    ``(number, sigma, mu, ei)`` for every crystal that cleared all gates --
    *mu* being the GP energy for optType 1 and the GP density otherwise --
    and *reason* is a printable sentence naming the gate that stopped the
    selection (``None`` on success).
    """
    mu_col, sigma_col, target = _ei_target(params)
    # optType 1 on a run that carries no energy prediction at all (an old
    # gp.csv, or a gp() that failed) still has something to learn from.
    if target == "eng_gp" and not _usable(numbers, table, mu_col, sigma_col):
        console_print("  optType 1 but no GP energy prediction this generation: "
                      "the EI selection falls back to the density")
        mu_col, sigma_col, target = _DEN_GP, _UNCERT, "den_gp"

    # optType 1 minimises, so the acquisition runs on -energy_gp and f_best
    # is -min(energy_gp); the density is maximised and keeps mu as it is.
    # `-N : optType` inverts both (opt_sign = -1).
    opt_sign = float(params.get("optSign", 1) or 1)
    sign = -opt_sign if target == "eng_gp" else opt_sign

    observed = _observed_incumbent(data_dir, target, sign)
    # Step 0: f_best is the incumbent EI measures improvement against.
    # The observed best in feature.csv wins when present; only a missing or
    # empty database falls back to the GP's own (noisy) prediction.
    # ``best_density`` is always read off the predictions: the `den` gate is a
    # promise about this generation's predicted density, not the past.
    f_best = -float("inf") if observed is None else observed
    best_density = -float("inf")
    max_mu_ei = -float("inf")           # best candidate mean, in EI space
    filtered = 0
    for number in numbers:
        row = table.get(number)
        if row is None:
            continue
        if row[1] > RESIDUAL_CUT:                     # residual
            filtered += 1
            continue
        if math.isfinite(row[_DEN_GP]):
            best_density = max(best_density, row[_DEN_GP])
        mu, sigma = row[mu_col], row[sigma_col]
        if not (math.isfinite(mu) and math.isfinite(sigma)):
            continue
        max_mu_ei = max(max_mu_ei, sign * mu)
        if observed is None:                 # predicted fallback only here
            f_best = max(f_best, sign * mu)
    if filtered > 0:
        console_print(f"  Filtered out {filtered} crystals with residual > "
                      f"{RESIDUAL_CUT:g}")
    if observed is not None:
        # ``sign * observed`` is the physical value again (sign in {-1, 1}):
        # the lowest energy for optType 1, the highest density for optType 11.
        console_print(
            f"  EI incumbent: observed {target} = {sign * observed:.4f}"
            f"  (max predicted this generation = {max_mu_ei:.4f},"
            f" gap = {observed - max_mu_ei:+.4f})")
        if math.isfinite(max_mu_ei) and observed > max_mu_ei:
            # The observed incumbent lies outside the range the GP can predict
            # for these candidates, so every (mu - f_best) is negative and EI
            # collapses to its exploration term.  This is the ct22 symptom.
            console_print("  EI note: incumbent above every candidate mean;"
                          " the improvement term vanishes, EI ~ exploration")

    # `den` is a density, so it gates on the predicted density whatever the
    # acquisition target is; -inf means no crystal carries one at all.
    if math.isfinite(best_density) and best_density < params["den"]:
        return [], "", (f"density threshold gate: best density="
                        f"{best_density:.4f} < den={params['den']:.4f}"), target

    explore = bool(params.get("explore", True))
    # Step 1: EI of every valid crystal
    ei_list, crystal_list, uncert_list, mu_list = [], [], [], []
    for number in numbers:
        row = table.get(number)
        if row is None or row[1] > RESIDUAL_CUT:
            continue
        mu, sigma = row[mu_col], row[sigma_col]
        if not (math.isfinite(mu) and math.isfinite(sigma)):
            continue
        ei_list.append(_expected_improvement(sign * mu, sigma, f_best, explore))
        crystal_list.append(number)
        uncert_list.append(sigma)
        mu_list.append(mu)
    if not ei_list:
        return [], "", f"no crystal with a usable {target} prediction", target

    # Step 2: select top-K
    n_sel = min(int(params["k"]), len(ei_list))
    mode = str(params["mode"])
    ei_label = "EI" if explore else "EI (no exploration)"
    if mode == "uncertainty":
        order = sorted(range(len(ei_list)), key=lambda i: -uncert_list[i])
        mode_label = "uncertainty"
    elif mode == "random":
        order = list(range(len(ei_list)))
        random.shuffle(order)
        mode_label = "random"
    elif mode == "ei":
        order = sorted(range(len(ei_list)), key=lambda i: -ei_list[i])
        mode_label = ei_label
    else:
        console_print(f"  Unknown select_mode \"{mode}\", falling back to EI")
        order = sorted(range(len(ei_list)), key=lambda i: -ei_list[i])
        mode_label = ei_label

    candidates = [(crystal_list[i], uncert_list[i], mu_list[i], ei_list[i])
                  for i in order[:n_sel]]
    # ``u`` is calibrated on the reported 95% half-width, so compare the
    # reconstructed 1.96*sigma against it (EI itself used sigma above).
    above_u = [s for s in candidates
               if s[1] * UNCERT_SCALE > params["u"]]
    if not above_u:
        return [], mode_label, (f"no crystal above the uncertainty threshold "
                                f"({params['u']:.4f}) among the top-{n_sel}"), target
    # `den` gates on the predicted density, taken from the row itself: the
    # candidate holds the acquisition value, which is the energy for optType 1.
    selected = [s for s in above_u if not _below_den(table[s[0]], params["den"])]
    if not selected:
        return [], mode_label, (f"every top-{n_sel} crystal is below the density "
                                f"gate den={params['den']:.4f}"), target
    return selected, mode_label, None, target


def _below_den(row, den):
    """True when the crystal's predicted density is under the ``den`` gate."""
    density = row[_DEN_GP] if row is not None else float("nan")
    return math.isfinite(density) and density < den


# Leave-one-out costs one O(N^3) solve; above this many labelled rows the
# learning-curve line reports the database size only.
_LOO_MAX_ROWS = 2000


def _loo_rmse(gpr):
    """Exact leave-one-out RMSE of a fitted GPR, from its own Cholesky factor.

    Rasmussen & Williams eq. 5.12: the LOO residual is ``(K^-1 y)_i / (K^-1)_ii``,
    so no refit is needed.  ``alpha_`` is in the normalised target when the
    model was fitted with ``normalize_y=True``, hence the rescale -- the
    centring stays the one of the full training set, which is the model that
    is actually being scored (refitting each fold would re-centre it and
    answer a slightly different question).

    The hyperparameters are the ones fitted on all rows, so this is in-sample
    for them: expect a mild optimism, not a held-out error.
    """
    from scipy.linalg import cho_solve

    n = np.asarray(gpr.alpha_).size
    k_inv = cho_solve((gpr.L_, True), np.eye(n))
    resid = np.ravel(gpr.alpha_) / np.diag(k_inv)
    scale = float(getattr(gpr, "_y_train_std", 1.0) or 1.0)
    return float(np.sqrt(np.mean((resid * scale) ** 2)))


def _cached_models(data_dir):
    """``(gpr_density, gpr_energy)`` uspexkit left in *data_dir*, else None."""
    try:
        from gpevo.uspexkit import core as _core
    except Exception:
        return None
    version = getattr(_core, "_KERNEL_VERSION", None)
    models = []
    for name in ("gpr_density.pkl", "gpr_energy.pkl"):
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "rb") as fp:
                blob = pickle.load(fp)
        except Exception:
            return None
        if not isinstance(blob, dict):
            return None
        if version is not None and blob.get("version") != version:
            return None                  # stale cache: not this model
        model = blob.get("model")
        # The LOO hat matrix (_loo_rmse) needs sklearn's Cholesky factor L_ /
        # alpha_.  A gpytorch cache carries a different version tag anyway;
        # this guard also covers mixed caches so LOO degrades to `n/a` instead
        # of raising inside _loo_rmse.
        if not hasattr(model, "L_"):
            return None
        models.append(model)
    return models if all(models) else None


def _loo_rmses(data_dir):
    """``(density, energy)`` LOO RMSE of the cached models, None when unknown.

    With ``gpbackend : gpytorch`` the cached models have no sklearn Cholesky
    factor, so ``_cached_models`` returns None and the learning curve reports
    ``LOO RMSE: n/a`` -- the hat-matrix identity is sklearn-only.
    """
    try:
        from sklearn.preprocessing import StandardScaler

        feature = np.loadtxt(os.path.join(data_dir, "feature_mlp.csv"),
                             delimiter=",", skiprows=1)
        labels = np.loadtxt(os.path.join(data_dir, "feature.csv"),
                            delimiter=",", skiprows=1)
        feature = np.atleast_2d(feature)[:, 1:]
        labels = np.atleast_2d(labels)
        rows = min(len(feature), len(labels))
        if rows < 3 or rows > _LOO_MAX_ROWS:
            return None
        models = _cached_models(data_dir)
        if models is None:
            return None
        X = StandardScaler().fit_transform(feature[:rows])
        return tuple(_loo_rmse(model) for model, _ in
                     zip(models, (labels[:rows, -1], labels[:rows, 1])))
    except Exception:
        return None


def _log_learning_curve(ORG, POP, data_dir, rows_before):
    """One line per generation: is the training set -- and the model -- growing?

    ``rows`` is the only thing the active-learning loop can always report; the
    LOO RMSE says whether the labels bought anything, and is skipped when
    uspexkit left no (or a stale) model behind.
    """
    try:
        rows_after = _db_rows(data_dir)
        generation = int(getattr(POP, "generation", 0))
        line = (f"  GP learning curve (gen {generation}): rows "
                f"{rows_before} -> {rows_after} (+{rows_after - rows_before})")
        rmses = _loo_rmses(data_dir)
        if rmses is None:
            line += f"  LOO RMSE: n/a ({rows_after} labelled rows)"
        else:
            line += (f"  LOO RMSE density={rmses[0]:.4f} g/cm^3"
                     f"  energy={rmses[1]:.4f} eV")
        console_print(line)
    except Exception:
        pass                            # a log line must never break a run


# Pre-DFT GP means paired with the DFT densities they bought, keyed by the
# global crystal id.  Kept in memory for the whole process (EA_310 loops here);
# a restart starts a fresh accumulator, which is fine for a diagnostic.
_CALIBRATION = {}
_CALIBRATION_DIR = None


def _record_gp_calibration(res_folder, ids, pre_mu):
    """One-line GP-vs-DFT calibration for the crystals just labelled.

    ``density.log`` (written by uspexkit.calc) holds ``id density energy`` of
    the DFT labels; ``pre_mu`` holds the GP mean EI ranked *before* the label
    existed.  Using the pre-label mean keeps this honest: the post-label
    re-prediction in density_predict.log is in-sample and would flatter the GP.
    """
    global _CALIBRATION_DIR
    if _CALIBRATION_DIR != res_folder:
        _CALIBRATION.clear()
        _CALIBRATION_DIR = res_folder
    try:
        with open(os.path.join(res_folder, "density.log"), "r",
                  encoding="utf-8", errors="replace") as fp:
            for line in fp:
                parts = line.split()
                if len(parts) < 2 or parts[0].startswith("#"):
                    continue
                try:
                    cid, density = int(float(parts[0])), float(parts[1])
                except ValueError:
                    continue
                if (cid in ids and cid in pre_mu and cid not in _CALIBRATION
                        and math.isfinite(density) and math.isfinite(pre_mu[cid])):
                    _CALIBRATION[cid] = (pre_mu[cid], density)
    except OSError:
        return
    if len(_CALIBRATION) < 3:
        return
    mu = np.asarray([p[0] for p in _CALIBRATION.values()], dtype=float)
    y = np.asarray([p[1] for p in _CALIBRATION.values()], dtype=float)
    slope, intercept = np.polyfit(mu, y, 1)
    console_print(
        f"  GP calibration (n={len(mu)} labelled): y = {slope:.3f}*mu "
        f"+ {intercept:.3f}; mean(mu-y) = {float(np.mean(mu - y)):+.4f}, "
        f"RMSE = {float(np.sqrt(np.mean((mu - y) ** 2))):.4f} g/cm^3")


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

    The work happens in ``_gp_active_learning``; this wrapper only makes sure
    the learning-curve line is printed on every exit path, including the gates
    that return early without labelling anything.
    """
    if not gp_enabled(ORG):
        return fitness
    _, _res_folder, data_dir = gp_data_paths(ORG)
    rows_before = _db_rows(data_dir)
    try:
        return _gp_active_learning(ORG, POP, fitness)
    finally:
        _log_learning_curve(ORG, POP, data_dir, rows_before)


def _gp_active_learning(ORG, POP, fitness):
    """Body of :func:`gp_active_learning` (see its docstring)."""
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
    # Structure ids are global, so everything left in the cache belongs to an
    # earlier generation: keep the cache at one generation's worth of features.
    _prune_feature_cache(numbers)
    table = metrics_table(POP, valid, numbers, os.path.join(home, "CalcFold1", "gp.csv"))
    rows_before = _db_rows(data_dir)

    if rows_before >= _MIN_DB_ROWS and table:
        selected, mode_label, reason, target = _select_by_ei(numbers, table, params, data_dir)
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
        target = "fitness"

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
        # `-N : optType` is stored as optType=N with opt_sign=-1.
        opt_label = ("-" if float(params["optSign"]) < 0 else "") + str(
            int(params["optType"]))
        console_print(f"{mode_label} top-{int(params['k'])} selection on "
                      f"{target} (optType {opt_label}): "
                      f"{len(selected)} crystals above threshold "
                      f"({params['u']:.4f})")
        for rank, (number, sigma, mu, ei) in enumerate(selected, start=1):
            console_print(f"  {rank}: crystal {number}  EI={ei:.6f}  {target}={mu:.4f}"
                          f"  sigma={sigma:.4f} (95%={sigma * UNCERT_SCALE:.4f})")
    _write_ei_log(ORG, POP.generation, [s[0] for s in selected])

    ids_list = [int(s[0]) for s in selected]
    # GP means BEFORE the DFT labels exist: the honest input to the calibration
    # diagnostic below (pred() re-predicts with the enlarged database later).
    pre_mu = {n: table[n][_DEN_GP] for n in ids_list
              if n in table and math.isfinite(table[n][_DEN_GP])}
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
    _record_gp_calibration(res_folder, set(ids_list), pre_mu)
    dropped = _invalidate_pkl(home, res_folder, data_dir)
    console_print(f"  GP training set: {rows_before} -> {rows_after} rows; "
                  f"dropped {len(dropped)} cached model(s)")

    if not pred_ids:
        console_print(f"  No valid crystals with residual <= {RESIDUAL_CUT:g}, "
                      f"skipping pred")
        return fitness

    # uspexkit.pred appends to results1/density_predict.log and does not
    # rewrite it: that log is the only place that keeps the σ a crystal was
    # predicted with, and `gpevo ei-report` needs the σ of the crystals EI
    # labelled in *earlier* generations.  Crystal ids are unique across
    # generations, so rows never collide (and a repeat would just overwrite).
    # Reuse the features the GP step of this generation already computed: the
    # geometry pred() would re-optimise is the same relaxed one, so this skips
    # one ReaxFF optimisation and four GULP runs per crystal.
    features = {}
    if _features_reusable(ORG):
        features = {i: _FEATURE_CACHE[i] for i in pred_ids if i in _FEATURE_CACHE}
        if features:
            console_print(f"  pred: reusing the GP-step feature for "
                          f"{len(features)}/{len(pred_ids)} crystal(s), "
                          f"skipping their ReaxFF re-optimisation")
    if not _backend_ready(params["backend"]):
        return fitness
    console_print(f"  Running: uspexkit.pred(ids='{' '.join(str(i) for i in pred_ids)}')")
    ok, results = _call("pred", cwd=res_folder, t="Individuals.traj", step=100,
                        ncpu=int(params["ncpu"]), den=0.0,
                        ids=" ".join(str(i) for i in pred_ids), dat=params["dat"],
                        features=features or None,
                        backend=params["backend"],
                        training_iter=int(params["training_iter"]),
                        learning_rate=float(params["learning_rate"]),
                        log_dir=res_folder)
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
        if row is not None and row[1] > RESIDUAL_CUT:
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
                  f"(skipped {skipped} with residual > {RESIDUAL_CUT:g})")
    return fitness
