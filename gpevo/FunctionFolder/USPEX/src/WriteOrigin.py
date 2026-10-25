"""Per-structure ``origin`` file: ``WriteOUTPUT.m:78`` + ``Initialize.m:84``.

One line per finished structure, appended while the OUTPUT.txt row is written,
titled by a header written once at run start::

     ID    Origin    Enthalpy   Parent-E   Parent-ID
       1   Random      -0.829    -0.829  [         0]
       1   Seeds       -1.956    [          ]
     223 keptBest      -1.988    -1.988  [       151]

``Enthalpy`` is the enthalpy per atom (``Enthalpies(end)/sum(numIons)``) and
``Parent-E`` the per-atom enthalpy the parents had -- that is the unit
``Parents.enthalpy`` is stored in (Heredity_310.m:105, KeepBestStructures.m:38).

Three row shapes, mirroring the MATLAB ``fprintf`` exactly:

* no ``Parents`` at all (random structures) -> ``Parent-ID`` is ``0`` and
  ``Parent-E`` repeats the structure's own enthalpy;
* ``Parents`` present but without ids (seeds) -> MATLAB stores ``parent=[]``
  and leaves ``enthalpy`` unset; its ``fprintf`` then emits nothing for the
  numeric conversion and ten blanks for the id, i.e. a blank ``Parent-E``;
* ``Parents`` with ids (heredity / keptBest) -> the parent ids and enthalpy.

The file is what ``makeFigures.m`` reads to colour structures by the operator
that produced them.
"""
from __future__ import annotations

import os

import numpy as np

_HEADER = " ID    Origin    Enthalpy   Parent-E   Parent-ID"


def _origin_path(res_folder):
    return os.path.join(res_folder, "origin")


def _enthalpy(s):
    try:
        return float(np.asarray(s.Enthalpies, dtype=float).ravel()[-1])
    except (AttributeError, IndexError, TypeError, ValueError):
        return float("nan")


def _ions(s):
    try:
        total = int(sum(int(v) for v in np.asarray(s.numIons).ravel()))
    except (AttributeError, TypeError, ValueError):
        total = 0
    return total if total > 0 else 1


def _parents(s):
    """``(parent_ids, parent_enthalpy)``; ``(None, None)`` without any Parents.

    ``parent_enthalpy`` is None when the entry carries no number, which is how
    MATLAB represents the seeds case (``parent=[]``, ``enthalpy`` unset).
    """
    entries = getattr(s, "Parents", None)
    if not entries:
        return None, None
    first = entries[0]
    ids = str(getattr(first, "parent", "") or "").strip()
    if not ids:
        return "", None
    try:
        enthalpy = float(getattr(first, "enthalpy", None))
    except (TypeError, ValueError):
        enthalpy = None
    return ids, enthalpy


def origin_line(count, s):
    """The ``origin`` row of one finished structure."""
    howcome = str(getattr(s, "howCome", "N/A") or "N/A")
    enth_atom = _enthalpy(s) / _ions(s)
    ids, parent_enthalpy = _parents(s)
    if ids is None:                       # no Parents at all
        ids, parent_enthalpy = "0", enth_atom
    head = f"{int(count):4d} {howcome:<11s} {enth_atom:8.3f}"
    if parent_enthalpy is None:
        # MATLAB's '%8.3f  %8.3f  [' emits nothing for the empty enthalpy but
        # keeps both separators, hence four blanks before the bracket.
        return f"{head}    [{ids:>10s}]"
    return f"{head}  {parent_enthalpy:8.3f}  [{ids:>10s}]"


def append_origin(res_folder, count, s):
    """Append one row to ``resFolder/origin``, creating the header if needed."""
    path = _origin_path(res_folder)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", encoding="utf-8") as fp:
            if new:
                fp.write(_HEADER + "\n")
            fp.write(origin_line(count, s) + "\n")
    except OSError as exc:
        from gpevo.log_utils import console_print

        console_print(f"  Warning: cannot append to origin: {exc}")
        return False
    return True
