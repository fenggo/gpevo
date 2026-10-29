"""Clean a USPEX run directory.

This is the packaged version of the ``clean.py`` that used to sit next to
``INPUT.txt``: it removes what a new run must not inherit (calculation folders,
checkpoints, the "still running" flags) and keeps what a run cannot do without
(``Specific/``, seeds, the results and the GP training set).

Rules, in one place instead of three slightly different copies:

* directories   ``AntiSeeds``, ``CalcFold*`` (every one of them, not just
  ``CalcFold1``) and the per-structure ``results*/<ID>`` scratch folders;
* files         ``*.pkl``, ``*.mat``, ``*.backup``, plus the state files
  ``still_running``, ``USPEX_IS_DONE``, ``octave-workspace``, ``Warnings`` and
  the ``ei.log`` of the GP selection -- it is appended to, so a new run would
  otherwise report the crystals an earlier one labelled.

``--all`` additionally removes ``results*/``.  Nothing else is ever touched:
``Specific``, ``Seeds``, ``INPUT.txt``, ``MOL_*`` and the GP data directory are
assets, not scratch.
"""
from __future__ import annotations

import glob
import os
import shutil

# Deleted by a plain `gpevo clean`.
# ``ei.log`` is appended to by every generation that labels something, so it
# belongs with the other per-run state: leaving it behind mixes the selections
# of two runs in one file.
_STATE_FILES = ("still_running", "USPEX_IS_DONE", "octave-workspace", "Warnings",
                "ei.log")
_FILE_PATTERNS = ("*.pkl", "*.mat", "*.backup")
_DIR_PATTERNS = ("AntiSeeds", "CalcFold*")

# Printed when the directory does not look like a run directory.
_MARKERS = ("INPUT.txt", "Current_ORG.pkl", "CalcFold1")


def _is_run_directory(root):
    return any(os.path.exists(os.path.join(root, name)) for name in _MARKERS) \
        or bool(glob.glob(os.path.join(root, "CalcFold*")))


def _plan(root, all_):
    """``(directories, files)`` that would be removed, most specific first."""
    directories = []
    for pattern in _DIR_PATTERNS:
        directories.extend(glob.glob(os.path.join(root, pattern)))
    # results*/<ID>: the folders uspexkit.calc creates when it labels a
    # structure.  Only the numeric ones -- AuxiliaryFiles and generationN stay.
    for results_dir in sorted(glob.glob(os.path.join(root, "results*"))):
        for entry in sorted(glob.glob(os.path.join(results_dir, "*"))):
            if os.path.isdir(entry) and os.path.basename(entry).isdigit():
                directories.append(entry)
    if all_:
        directories.extend(sorted(glob.glob(os.path.join(root, "results*"))))

    files = []
    for pattern in _FILE_PATTERNS:
        files.extend(sorted(glob.glob(os.path.join(root, pattern))))
    for name in _STATE_FILES:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            files.append(path)
    return directories, files


def clean(path=".", all_=False, force=False, dry_run=False):
    """Clean the run directory *path*.

    Returns the number of removed entries.  Raises ``SystemExit(2)`` when
    *path* does not look like a run directory and ``force`` was not given --
    deleting a wrong directory is not recoverable.
    """
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        raise SystemExit(f"clean: {root} is not a directory")
    if not force and not _is_run_directory(root):
        raise SystemExit(
            "clean: {} does not look like a USPEX run directory "
            "(no {})\n       re-run with --force if that is really it".format(
                root, " / ".join(_MARKERS)))

    directories, files = _plan(root, all_)
    for entry in directories + files:
        print(("would remove " if dry_run else "removing   ") +
              os.path.relpath(entry, root))
        if dry_run:
            continue
        try:
            if os.path.isdir(entry) and not os.path.islink(entry):
                shutil.rmtree(entry)
            else:
                os.remove(entry)
        except OSError as exc:
            print(f"  Warning: cannot remove {entry}: {exc}")

    removed = len(directories) + len(files)
    print("clean: {} entr{} {}".format(removed, "y" if removed == 1 else "ies",
                                       "would be removed" if dry_run else "removed"))
    return removed
