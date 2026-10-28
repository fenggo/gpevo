"""Python equivalent of ReadJobs_310.m (non-parallel, GULP/molecular path)."""
from __future__ import annotations

import csv
import math
import os
from importlib import import_module

import numpy as np

from gpevo.runtime import G
from gpevo.FunctionFolder.Submission.checkStatusC import checkStatusC
from gpevo.FunctionFolder.USPEX.src.Reading import Reading
from gpevo.log_utils import console_print
WriteIndividualOutput_310 = import_module(
    "gpevo.FunctionFolder.USPEX.310.WriteIndividualOutput_310"
).WriteIndividualOutput_310


def _float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_gp_metrics(s, ORG):
    """Best-effort read of gp.csv columns into an individual's GP fields.

    Since ``uspexkit.gp()`` returns its metrics, submitJob already stores them
    on the structure; the file is only consulted when they are missing (e.g. a
    run started before the switch, or a structure evaluated by the CLI).
    Columns: 2 residual, 5 density_gp, 6 uncertainty, 8 energy_pred,
    9 uncertainty_eng.
    """
    for field in ("residual", "distance", "density_gp", "uq", "energy_gp", "uq_eng"):
        if not hasattr(s, field):
            setattr(s, field, float("nan"))
    if all(math.isfinite(float(getattr(s, field))) for field in
           ("distance", "density_gp", "uq")):
        return

    gp_file = os.path.join(getattr(ORG, "homePath", os.getcwd()), "CalcFold1", "gp.csv")
    if not os.path.isfile(gp_file):
        return

    try:
        with open(gp_file, "r", encoding="utf-8", errors="replace") as fp:
            reader = csv.reader(fp)
            next(reader, None)
            matched = None
            target = int(getattr(s, "Number", 0))
            for row in reader:
                if len(row) < 7:
                    continue
                try:
                    if int(float(row[0])) != target:
                        continue
                except (TypeError, ValueError):
                    continue
                matched = row
            if matched is not None:
                s.residual = _float_or_nan(matched[2])
                s.distance = _float_or_nan(matched[2])
                s.density_gp = _float_or_nan(matched[5])
                s.uq = _float_or_nan(matched[6])
                if len(matched) >= 10:
                    s.energy_gp = _float_or_nan(matched[8])
                    s.uq_eng = _float_or_nan(matched[9])
    except Exception:
        # A malformed or missing gp.csv must not break job reading; keep
        # inherited values (already initialized above) for this individual.
        return


def ReadJobs_310():
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    for indic in range(1, ORG.numParallelCalcs + 1):
        which = [i for i, s in enumerate(POP.POPULATION) if s.Folder == indic]
        if not which:
            continue
        idx = which[0]
        s = POP.POPULATION[idx]
        step = int(s.Step)
        console_print(f"Structure{idx + 1} step{step} at CalcFold{indic}")
        if getattr(s, "JobID", 0) == 0:
            continue
        done_or = checkStatusC(idx)
        if done_or and getattr(ORG, "platform", 0) > 0:
            marker = os.path.join(ORG.homePath, "CalcFoldTemp", str(getattr(s, "JobID", 0)))
            try:
                if os.path.exists(marker):
                    os.remove(marker)
            except OSError:
                pass
        from gpevo.FunctionFolder.sys.fingerprint_utils import compute_structure_metrics

        if done_or:
            if s.JobID == 0.01:
                s.Step = len(ORG.abinitioCode) + 1
            elif s.JobID == 0.02:
                s.Step = step + 1
            else:
                _ = Reading(ORG.abinitioCode[step - 1], idx, indic)

            s.JobID = 0
            if s.Error > ORG.maxErrors:
                s.Done = 1
                s.ToDo = 0
                s.Folder = 0
            elif s.Step > len(ORG.abinitioCode):
                s.Done = 1
                s.ToDo = 0
                POP.bodyCount += 1
                s.Number = POP.bodyCount
                s.Folder = 0
                console_print("Relaxation is done.")
                _load_gp_metrics(s, ORG)
                POP.DoneOrder[idx] = POP.bodyCount
                try:
                    compute_structure_metrics(s, ORG)
                    if getattr(s, "symg", None) is None or str(getattr(s, "symg", "")).strip().upper() == "N/A":
                        s.symg = getattr(s, "symg", 1)
                except Exception as exc:
                    console_print(f"WARNING: structure metrics failed for structure {idx + 1}: {exc}")
                WriteIndividualOutput_310(idx)
            from gpevo.runtime import safesave
            safesave("Current_POP.pkl", POP)
        else:
            console_print(f"Relaxation is not done for step: {idx + 1}")
    G["POP_STRUC"] = POP
