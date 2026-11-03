"""Python equivalent of USPEX/Start.m dispatcher."""
from __future__ import annotations

import os
import re
import time
from importlib import import_module as _im

import numpy as np

from gpevo.runtime import G, reset_workspace, set_home
from gpevo.log_utils import configure_log
from gpevo.FunctionFolder.USPEX.src.createORGStruc import createORGStruc
from gpevo.FunctionFolder.USPEX.src.Initialize import Initialize
from gpevo.FunctionFolder.sys.CreateCalcFolder import CreateCalcFolder
from gpevo.log_utils import console_print


def calcTypeStr():
    ORG = G["ORG_STRUC"]
    return f"{ORG.dimension}{ORG.molecule}{ORG.varcomp}"


def _setup_rng(ORG):
    # Seed numpy and random modules similar to USPEX's clock-based seeding.
    now = time.time()
    seed = int((now * 1000000) % (2**31))
    np.random.seed(seed)
    import random as _random
    _random.seed(seed)


# Attributes a restart must keep from the checkpoint even though INPUT.txt
# also sets them.  Two groups:
#   * fixed when the run is initialised -- changing the composition or the
#     population size mid-run would contradict the population the checkpoint
#     restores;
#   * written back by the loop -- latVolume is re-averaged from the population
#     every generation, firstGeneSplit* carries the composition split, and
#     update_STUFF zeroes / rebuilds the operator fractions and tournament.
_RESTART_KEEP = (
    "dimension", "molecule", "varcomp", "optType", "opt_sign",
    "populationSize", "numGenerations", "initialPopSize", "numIons", "numMols",
    "resFolder", "homePath", "specificFolder", "log_file",
    "latVolume", "tournament", "softMutOnly", "fracTrans", "howManyTrans",
    "fracPerm", "fracGene", "minAt", "maxAt",
    "firstGeneSplit", "firstGeneSplitAll",
    "correlation_coefficient", "cor_dir",
)


def _report_changes(before, ORG, limit=12):
    """Print the options INPUT.txt actually changed on this restart."""
    changed = []
    for name, value in sorted(ORG.items()):
        if name in _RESTART_KEEP or name not in before:
            continue
        old = before[name]
        if isinstance(value, (int, float, str)) and not isinstance(value, bool) \
                and isinstance(old, (int, float, str)) and not isinstance(old, bool) \
                and value != old:
            changed.append(f"{name}: {old} -> {value}")
    if not changed:
        return
    for line in changed[:limit]:
        console_print(f"  INPUT.txt (re-read on restart) {line}")
    if len(changed) > limit:
        console_print(f"  ... and {len(changed) - limit} more")


def _refresh_input_options(input_file="INPUT.txt"):
    """Re-read INPUT.txt into the ORG a restart just restored.

    A restart loads ``Current_ORG.pkl``, so edits made to INPUT.txt while the
    run was stopped were silently ignored -- only the fingerprint block used
    to be re-parsed.  The option parsers are pure (they only set attributes on
    ``G["ORG_STRUC"]``), so running them again on the restored ORG refreshes
    everything tunable, the GP block included, while *_RESTART_KEEP* stays as
    the checkpoint had it.

    A parse failure must not kill a calculation that is already running, so
    the checkpoint is put back and the restart continues with it.
    """
    import copy

    ORG = G["ORG_STRUC"]
    if not os.path.isfile(input_file):
        return ORG
    from gpevo.FunctionFolder.USPEX.src.createORG_System import createORG_System
    from gpevo.FunctionFolder.USPEX.src.createORG_AbinitCode import \
        createORG_AbinitCode
    from gpevo.FunctionFolder.USPEX.src.createORG_Symmetry import createORG_Symmetry
    from gpevo.FunctionFolder.USPEX.src.createORG_EA import createORG_EA
    from gpevo.FunctionFolder.USPEX.src.createORG_Molecule import createORG_Molecule
    from gpevo.FunctionFolder.USPEX.src.createORG_Fingerprint import \
        createORG_Fingerprint

    snapshot = copy.deepcopy(dict(ORG))
    try:
        createORG_System(input_file)
        createORG_AbinitCode(input_file)
        createORG_Symmetry(input_file)
        createORG_EA(input_file)
        if os.path.exists(os.path.join(ORG.homePath, "MOL_1")):
            createORG_Molecule(input_file)
        createORG_Fingerprint(input_file)
    except Exception as exc:
        console_print(f"  Warning: INPUT.txt could not be re-read on restart "
                      f"({exc}); the checkpoint's settings are kept")
        G["ORG_STRUC"].clear()
        G["ORG_STRUC"].update(snapshot)
        return G["ORG_STRUC"]

    ORG = G["ORG_STRUC"]
    _report_changes(snapshot, ORG)
    for name in _RESTART_KEEP:
        if name in snapshot:
            setattr(ORG, name, snapshot[name])
    return ORG


def _results_dir(home):
    """The newest results<N> of home, or '' when there is none."""
    best, number = "", -1
    try:
        names = os.listdir(home)
    except OSError:
        return ""
    for name in names:
        match = re.fullmatch(r"results(\d+)", name)
        if match and int(match.group(1)) > number:
            path = os.path.join(home, name)
            if os.path.isdir(path):
                number = int(match.group(1))
                best = path
    return best


def _uspex_snapshots(POP):
    """[(generation, path), ...] of the per-generation USPEX checkpoints."""
    res = getattr(POP, "resFolder", None) or ""
    if not os.path.isdir(res):
        # A run moved to another machine keeps the absolute resFolder of the
        # machine it started on; the results folder of the current directory is
        # then the only one that can hold its snapshots.
        res = _results_dir(os.getcwd())
    if not res:
        return []
    found = []
    for name in os.listdir(res):
        match = re.fullmatch(r"generation(\d+)", name)
        if not match:
            continue
        path = os.path.join(res, name, "USPEX.pkl")
        if os.path.isfile(path):
            found.append((int(match.group(1)), path))
    return sorted(found)


def _restore_uspex_struc(POP):
    """The cumulative record a restart has to continue from, or None.

    USPEX_STRUC.POPULATION is the one place every structure ever relaxed is
    remembered: WriteIndividual rewrites Individuals from it.  It is checkpointed
    per generation (generation<N>/USPEX.pkl), not in Current_POP.pkl, so a
    restart that never loaded it started with an empty record -- and the next
    WriteIndividual then dropped every generation before the restart from the
    table.

    The newest snapshot that still starts at structure 1 is the base (a snapshot
    written after such a restart holds only its own generation), and the
    structures the newer snapshots know about are merged back in, so a
    generation that was already finished before the restart is not lost either.
    """
    from gpevo.runtime import safeload

    snapshots = _uspex_snapshots(POP)
    if not snapshots:
        return None
    generation = int(getattr(POP, "generation", 0) or 0)
    # a snapshot of the generation being restarted may still be half-written
    usable = [(n, p) for n, p in snapshots if n < generation] or snapshots

    base = None
    for number, path in reversed(usable):
        uspex = safeload(path)
        population = list(getattr(uspex, "POPULATION", []) or [])
        if population and population[0] is not None:
            base = (number, uspex)
            break
    if base is None:
        return None

    number, uspex = base
    merged = list(uspex.POPULATION)
    for later, path in usable:
        if later <= number:
            continue
        other = safeload(path)
        for index, record in enumerate(getattr(other, "POPULATION", []) or []):
            if record is None:
                continue
            while len(merged) <= index:
                merged.append(None)
            merged[index] = record
    uspex.POPULATION = merged
    return uspex


def Start():
    _setup_rng(None)

    # The run directory comes from the current working directory.
    # Bundled resources are resolved from the package itself.
    working = os.getcwd()
    set_home(working)

    ORG = G["ORG_STRUC"]
    console_print("  ")

    if os.path.exists("Current_POP.pkl"):
        from gpevo.runtime import safeload
        POP = safeload("Current_POP.pkl")
        ORG = safeload("Current_ORG.pkl")
        if not getattr(POP, "resFolder", None):
            POP.resFolder = ORG.resFolder
        G["POP_STRUC"] = POP
        G["ORG_STRUC"] = ORG
        # Re-read INPUT.txt: the checkpoint is the only source of options on a
        # restart, so anything edited while the run was stopped (the GP gates
        # above all) would otherwise be ignored.  Fingerprint weights are
        # refreshed here too -- older checkpoints were saved before the
        # molecular weight bug was fixed and may carry an atom-type weight
        # vector sized for numMols instead of numIons.
        ORG = _refresh_input_options("INPUT.txt")
        G["ORG_STRUC"] = ORG
        # The cumulative record lives in the per-generation snapshots only, so
        # a restart has to put it back or Individuals loses every generation
        # before the restart the next time it is written.
        # Prefer the top-level checkpoint written by WriteGenerationBackup; the
        # per-generation merge (_restore_uspex_struc) is only a fallback for
        # runs that predate Current_USPEX.pkl.
        USPEX = None
        if os.path.exists("Current_USPEX.pkl"):
            USPEX = safeload("Current_USPEX.pkl")
        if USPEX is None or not getattr(USPEX, "POPULATION", None):
            USPEX = _restore_uspex_struc(POP)
        if USPEX is not None:
            G["USPEX_STRUC"] = USPEX
        # Reprocess already-relaxed individuals from a finished checkpoint so
        # restarted runs no longer keep N/A descriptor columns.
        from gpevo.FunctionFolder.sys.fingerprint_utils import compute_structure_metrics
        for _s in POP.POPULATION:
            if getattr(_s, "symg", None) is None or str(getattr(_s, "symg", "")).strip().upper() == "N/A":
                _s.symg = 1
            if int(getattr(_s, "Done", 0)) == 1:
                try:
                    compute_structure_metrics(_s, ORG)
                except Exception:
                    pass
        from gpevo.FunctionFolder.USPEX.src.pickAntiSeeds import (
            ANTISEEDS_FILE,
            pickAntiSeeds,
        )

        if os.path.exists(ANTISEEDS_FILE):
            try:
                G["ANTISEEDS"] = safeload(ANTISEEDS_FILE)
            except Exception:
                pickAntiSeeds()
        else:
            pickAntiSeeds()
        # Work in result folder (matches POP structure)
        if POP.resFolder and os.path.exists(POP.resFolder):
            os.makedirs(POP.resFolder, exist_ok=True)
        CreateCalcFolder()
    else:
        ORG = createORGStruc("INPUT.txt")
        G["ORG_STRUC"] = ORG
        CreateCalcFolder(1)
        Initialize()
        CreateCalcFolder()
        # AntiSeedsCorrection.m only ever sees anti-seeds on a restart in the
        # reference (Start.m:57 sits in the restart branch), but reading them
        # here too makes a user-supplied AntiSeeds/POSCARS work for a fresh run.
        if os.path.isfile(os.path.join(ORG.homePath, "AntiSeeds", "POSCARS")):
            from gpevo.FunctionFolder.USPEX.src.pickAntiSeeds import pickAntiSeeds

            pickAntiSeeds()

    for name in ("ORG_STRUC", "POP_STRUC"):
        if name not in G or G[name] is None:
            raise RuntimeError(f"{name} is not initialized")

    ORG = G["ORG_STRUC"]
    configure_log(os.path.join(ORG.resFolder, ORG.log_file or "OUTPUT.txt"))
    # Save current state before entering generation loop (restartable).
    from gpevo.runtime import safesave
    safesave("Current_ORG.pkl", G["ORG_STRUC"])
    safesave("Current_POP.pkl", G["POP_STRUC"])
    if G.get("USPEX_STRUC") is not None:
        safesave("Current_USPEX.pkl", G["USPEX_STRUC"])

    ORG = G["ORG_STRUC"]
    if (ORG.dimension == 3 and ORG.molecule == 1) or \
       (ORG.dimension == 1 and ORG.molecule == 1 and ORG.varcomp == 0):
        for name in os.listdir(ORG.homePath):
            if name.startswith("MOL_"):
                shutil_copy = __import__("shutil").copy
                shutil_copy(os.path.join(ORG.homePath, name), os.path.join(ORG.resFolder, name))
    elif ORG.dimension == 2 and ORG.molecule == 0:
        src = os.path.join(ORG.homePath, "POSCAR_SUBSTRATE")
        if os.path.exists(src):
            shutil_copy(os.path.join(ORG.homePath, "POSCAR_SUBSTRATE"),
                        os.path.join(ORG.resFolder, "POSCAR_SUBSTRATE"))

    calc_type = calcTypeStr()
    module = _im("gpevo.FunctionFolder.USPEX." + calc_type + ".EA_" + calc_type)
    EA = getattr(module, "EA_" + calc_type)
    EA()

    ORG = G["ORG_STRUC"]
    if getattr(ORG, "repeatForStatistics", 1) > 1:
        console_print("Repeat statistics mode is not implemented in this Python port yet")
    else:
        if os.path.exists("still_running"):
            os.remove("still_running")

    console_print("USPEX IS DONE!")
    with open("USPEX_IS_DONE", "w") as fh:
        fh.write("1\n")
