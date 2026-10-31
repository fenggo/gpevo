"""Clean a USPEX run directory.

This is the packaged version of the ``clean.py`` that used to sit next to
``INPUT.txt``: it removes what a new run must not inherit (calculation folders,
checkpoints, the "still running" flags) and keeps what a run cannot do without
(``Specific/``, seeds, the results and the GP training set).

Rules, in one place instead of three slightly different copies:

* directories   ``AntiSeeds``, ``CalcFold*`` (every one of them, not just
  ``CalcFold1``);
* files         ``*.pkl``, ``*.mat``, ``*.backup``, plus the state files
  ``still_running``, ``USPEX_IS_DONE``, ``octave-workspace``, ``Warnings`` and
  the ``ei.log`` of the GP selection -- it is appended to, so a new run would
  otherwise report the crystals an earlier one labelled.  Current runs write
  that log into their own ``results<N>/``; only the home-directory copy of a
  run finished before that move is left to remove.

The per-structure ``results*/<ID>`` folders -- what ``uspexkit.calc`` writes
when it labels a structure -- are **kept**: they are the DFT record of that
run, sitting next to its ``Individuals`` and ``ei.log``, and a following run
opens its own ``results<N+1>`` anyway.  ``--all`` removes them with the rest of
``results*/``.

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
# of two runs in one file.  It is written to ``results<N>/`` nowadays, so this
# only picks up the home-directory copy an older run left behind.
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
    # results*/<ID> -- the folders uspexkit.calc creates when it labels a
    # structure -- are the DFT record of that run, so they stay: a new run
    # writes into its own results<N+1> and never inherits them.  Only --all,
    # which drops results*/ wholesale, takes them along.
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
