"""Shared MATLAB-style global workspace and lightweight array/struct helpers."""
from __future__ import annotations

import os
import sys
import math
import json
import pickle
import subprocess
import random as _random
from collections import defaultdict


# Cutoff on the GP residual: squared distance to the nearest labelled structure
# measured in the STANDARDISED 10-D ReaxFF feature space (the same space the GP
# is fitted in).  Single source of truth for uspexkit.core, gp_ml, gp_plot and
# the ei_report / ei_select tools.  Calibrated on the p99 of within-database
# nearest-neighbour distances: ~5.0 (example/data22) / ~3.2 (dnmt/data).
# See docs/algorithm_review.md §2.3.
RESIDUAL_CUT = 5.0


# ---------------------------------------------------------------------------
# Struct helpers
# ---------------------------------------------------------------------------

class AttrDict(dict):
    """A dict that allows matlab struct-ish ``x.field`` access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def copy(self):
        return AttrDict(super().copy())


def struct(fields=None, **kwargs):
    """Create an AttrDict.  ``fields`` is kept for matlab-like call sites."""
    out = AttrDict()
    if fields:
        for key in fields:
            if isinstance(key, str):
                out[key] = None
    out.update(kwargs)
    return out


def quickstart(template=None):
    """Return one empty element of a matlab ``struct('Field',{},...)`` array."""
    return AttrDict()


# ---------------------------------------------------------------------------
# Global workspace (mirrors MATLAB `global` variables).
# It is kept persistent across the main loop and saved/loaded as JSON/pickle.
# ---------------------------------------------------------------------------

CURRENT_FILE = "Current_POP.gpevo.json"
ORG_FILE = "Current_ORG.gpevo.json"
USPEX_FILE = "USPEX.gpevo.json"
POOL_FILE = "POOL.gpevo.json"

G = {
    "ORG_STRUC": None,
    "POP_STRUC": None,
    "USPEX_STRUC": None,
    "POOL_STRUC": None,
    "ANTISEEDS": None,
}

_HOME = None
_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))


def set_home(path: str) -> None:
    global _HOME
    _HOME = os.path.abspath(path)


def home_path() -> str:
    if _HOME is not None:
        return _HOME
    return os.getcwd()


def package_root() -> str:
    """Absolute path to the installed ``gpevo`` Python package root."""
    return _PACKAGE_ROOT


def resource_path(*parts: str) -> str:
    """Resolve a file bundled inside the ``gpevo`` package."""
    return os.path.join(_PACKAGE_ROOT, *parts)



def reset_workspace() -> None:
    global G
    G = {
        "ORG_STRUC": None,
        "POP_STRUC": None,
        "USPEX_STRUC": None,
        "POOL_STRUC": None,
        "ANTISEEDS": None,
    }


# ---------------------------------------------------------------------------
# File existence / mkdir / shell helpers
# ---------------------------------------------------------------------------

def fexist(path: str, kind: str = "file") -> bool:
    if kind == "dir":
        return os.path.isdir(path)
    return os.path.exists(path)


def exist(path: str, kind: str = "file") -> bool:
    """Small compatibility wrapper for calls such as exist('Seeds','dir')."""
    return fexist(path, kind)


def mkdir(path: str) -> tuple[int, str]:
    # MATLAB mkdir returns [status, message]; first call in createORGStruc uses
    # ``[nothing1,existfold] = mkdir(name)`` where a non-empty second return
    # means the folder already existed.
    if os.path.isdir(path):
        return 1, "directory already exists"
    try:
        os.makedirs(path, exist_ok=True)
        return 1, ""
    except OSError as exc:
        return 0, str(exc)


def unix(command: str, nargout: int | None = None) -> object:
    """Run a shell command and mimic Octave's ``unix``.

    Octave returns ``[status, output]`` when two outputs are requested and
    ``status`` when one output is requested.  In this Python port we simply
    return ``(status, output)``; call sites are written as Python tuples.
    """
    proc = subprocess.run(
        command,
        shell=True,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    return proc.returncode, proc.stdout


def safesave(filename: str, value, workdir: str | None = None) -> None:
    path = filename if os.path.isabs(filename) else os.path.join(workdir or os.getcwd(), filename)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(value, fh)
    os.replace(tmp, path)


def safeload(filename: str, workdir: str | None = None):
    path = filename if os.path.isabs(filename) else os.path.join(workdir or os.getcwd(), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# MATLAB-like functions used by the port
# ---------------------------------------------------------------------------

def isempty(value) -> bool:
    if value is None:
        return True
    try:
        import numpy as np
        return bool(np.size(value) == 0)
    except Exception:
        return len(value) == 0


def isfield(obj, field: str) -> bool:
    return isinstance(obj, (dict, AttrDict)) and field in obj


def isnumeric(value) -> bool:
    return isinstance(value, (int, float, complex, bool)) or isinstance(
        value, (list, tuple)
    ) if "numpy" not in sys.modules else bool(__import__("numpy").isscalar(value)) and isinstance(value, __import__("numpy").number)


def isnan(value) -> bool:
    return value is not None and isinstance(value, float) and math.isnan(value)


def num2str(value) -> str:
    if isinstance(value, float):
        if math.isclose(value, int(value)):
            return str(int(value))
        return repr(value)
    return str(value)


def str2num(text):
    """Parse a number/vector/matrix string into int/float/list.

    This intentionally returns ``None`` for unparsable text, like smop's
    behavior.  Supports whitespace/semicolon separated matrices.
    """
    import re
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if s.count(";") or s.count("\n") > 0:
        rows = [str2num(part) for part in re.split(r"[;\n]+", s)]
        rows = [r for r in rows if r is not None]
        return rows or None
    pieces = [piece for piece in s.replace(",", " ").split() if piece]
    if not pieces:
        return None
    try:
        vals = []
        for piece in pieces:
            try:
                vals.append(int(float(piece)))
            except ValueError:
                vals.append(float(piece))
        return vals[0] if len(vals) == 1 else vals
    except Exception:
        return None


def strfind(haystack, needle):
    hs = str(haystack)
    nd = str(needle)
    idx = hs.find(nd)
    if idx < 0:
        return []
    return [idx + 1]


def strcmp(a, b) -> bool:
    return str(a) == str(b)


def strcmpi(a, b) -> bool:
    return str(a).lower() == str(b).lower()


# ---------------------------------------------------------------------------
# Entry help
# ---------------------------------------------------------------------------

def quit(code: int = 0):
    raise SystemExit(code)


def error(message: str = ""):
    raise RuntimeError(message)


def warn(message: str = ""):
    print(message, file=sys.stderr)
