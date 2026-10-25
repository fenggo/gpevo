"""Python equivalent of EA_310.m."""
from __future__ import annotations

import os
import time
import math
from importlib import import_module as _im

import numpy as np

from gpevo.runtime import G, struct, safesave
from gpevo.log_utils import write_log
ReadJobs_310 = _im("gpevo.FunctionFolder.USPEX.310.ReadJobs_310").ReadJobs_310
SubmitJobs_310 = _im("gpevo.FunctionFolder.USPEX.310.SubmitJobs_310").SubmitJobs_310
CalcFitness_310 = _im("gpevo.FunctionFolder.USPEX.310.CalcFitness_310").CalcFitness_310
from gpevo.FunctionFolder.USPEX.src.Correlation import Correlation
from gpevo.FunctionFolder.USPEX.src.AntiSeedsCorrection import AntiSeedsCorrection
from gpevo.FunctionFolder.USPEX.src.FitnessRankingCorrection import FitnessRankingCorrection
from gpevo.FunctionFolder.USPEX.src.StopRun import StopRun
from gpevo.FunctionFolder.USPEX.src.Finish import Finish
from gpevo.FunctionFolder.USPEX.src.WriteGenerationOutput_310 import WriteGenerationOutput_310
from gpevo.FunctionFolder.USPEX.src.WriteGenerationBackup import WriteGenerationBackup
from gpevo.FunctionFolder.USPEX.src.WriteGenerationStart import WriteGenerationStart
from gpevo.FunctionFolder.USPEX.src.pick_Seeds import pick_Seeds
from gpevo.FunctionFolder.sys.CreateCalcFolder import CreateCalcFolder
from gpevo.FunctionFolder.sys.alignLine import alignLine
from gpevo.log_utils import console_print


def _op(name):
    return getattr(_im("gpevo.FunctionFolder.USPEX.310." + name), name)


def EA_310():
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    while POP.generation < ORG.numGenerations + 0.5:
        # Local optimization loop, matching the MATLAB inner ``while 1`` loop.
        # For scheduler platforms this Python port polls the queue in-process
        # instead of exiting and restarting from Current_POP.mat.
        while True:
            ReadJobs_310()
            SubmitJobs_310()
            done_flags = [bool(s.Done) for s in POP.POPULATION]
            if all(done_flags):
                break
            if ORG.platform > 0 or ORG.numParallelCalcs > 1:
                interval = getattr(ORG, "pollingInterval", 20)
                time.sleep(interval)

        console_print("status = Local optimization finished")
        console_print(" ")
        write_log("Local optimization finished\n\n")
        write_log(alignLine(f"SUMMARY of Generation {int(POP.generation)}") + "\n")

        good = sum(1 for s in POP.POPULATION if np.asarray(s.Enthalpies)[-1] < 90000)
        total = max(1, len(POP.POPULATION))
        if good < max(1, total // 4):
            write_log("Too many structures have errors or failed the constraints after optimization.\n"
                      "Please check the input files. The calculation has to stop.\n"
                      "Possible reasons: badly tuned optimization parameters or unreasonable contraints.\n")
            # EA_310.m:41-45 writes this to the log only (fprintf), so unlike the
            # disp() messages it does not belong on the terminal.
            raise SystemExit(1)
        elif good / total < ORG.bestFrac:
            write_log("Some structures have errors or failed the constraints after optimisation,\n"
                      "bestFrac parameter will be lowered for this generation to discard such structures.\n"
                      "Please check the input files. Results may not be reliable. \n"
                      "Possible reasons: high bestFrac, bad optimization parameters or unreasonable contraints.\n")
            # EA_310.m:46-51: log only as well.

        # Fitness and ranking.  POP.ranking is published by
        # FitnessRankingCorrection (MATLAB does the same); ReRank only reorders
        # POPULATION inside Start_POP_310.
        fitness = CalcFitness_310()
        POP.fitness = np.asarray(fitness)
        cor = Correlation(fitness)
        write_log(alignLine(f"Correlation coefficient = {cor:.4f}") + "\n")
        fitness = AntiSeedsCorrection(fitness)
        fitness = FitnessRankingCorrection(fitness)
        from gpevo.FunctionFolder.USPEX.src.compositionStatistic import compositionStatistic

        compositionStatistic(ORG.firstGeneSplitAll)
        WriteGenerationOutput_310(fitness)

        to_stop, message = StopRun(fitness, POP.generation, ORG.numGenerations, ORG.stopCrit, ORG.stopFitness)
        if to_stop:
            write_log(message + "\n")
            Finish()
            return

        write_log("Proceeding to Selection Process\n\n")
        WriteGenerationBackup()

        # Operator fractions / tournament for the coming generation
        # (EA_310.m:75).  The input file is re-read here, like MATLAB does.
        from gpevo.FunctionFolder.USPEX.src.update_STUFF import update_STUFF

        update_STUFF("INPUT.txt", good / max(1, len(POP.POPULATION)), POP.ranking)
        ORG = G["ORG_STRUC"]

        # Keep/rebuild next generation.
        CreateCalcFolder()
        OFF = struct()
        OFF.POPULATION = []
        OFF.generation = POP.generation
        OFF.bodyCount = POP.bodyCount
        OFF.resFolder = getattr(POP, "resFolder", ORG.resFolder)
        OFF.ranking = list(POP.ranking)
        OFF.DoneOrder = []
        G["OFF_STRUC"] = OFF

        # Variation operators are invoked in the same order and with the same
        # counts as the MATLAB EA_310.
        operations = ["Heredity_310", "Random_310", "Permutation_310", "Rotation_310",
                      "LatMutation_310", "SoftModeMutation_310"]
        counts = [ORG.howManyOffsprings, ORG.howManyRand, ORG.howManyPermutations,
                  ORG.howManyRotations, ORG.howManyMutations, ORG.howManyAtomMutations]
        count = 0
        for i in range(len(operations)):
            for j in range(max(0, int(counts[i]))):
                # Ind_No runs over the whole generation, like EA_310.m:88-95,
                # so the "Structure N generated by ..." lines are numbered
                # continuously.
                count = count + 1
                try:
                    _op(operations[i])(count)
                except Exception as exc:
                    console_print(f"Variation operator {operations[i]}[{j + 1}] failed: {exc}")
                    _op("Random_310")(count)

        console_print(" ")
        console_print("Variation operators applied, applying elitist scheme")
        write_log(alignLine("Variation operators applied") + "\n")

        # Elitist scheme: carry over the best structures.
        import gpevo.FunctionFolder.USPEX.src.KeepBestStructures as KBS
        addon_diff = KBS.KeepBestStructures()
        write_log(f"            {int(ORG.howManyOffsprings):4d} structures produced by heredity     \n")
        write_log(f"            {int(ORG.howManyRand):4d} structures produced by random       \n")
        write_log(f"            {int(ORG.howManyAtomMutations):4d} structures produced by softmutation \n")
        write_log(f"            {int(ORG.howManyPermutations):4d} structures produced by permutation  \n")
        write_log(f"            {int(ORG.howManyMutations):4d} structures produced by latmutation  \n")
        write_log(f"            {int(ORG.howManyRotations):4d} structures produced by rotmutation  \n")
        write_log(f"            {int(addon_diff):4d} structures kept as best from the previous generation\n")

        # No trimming/padding: like EA_310.m, every offspring produced by the
        # variation operators survives, with the kept-best appended after them.
        POP = OFF
        POP.generation = POP.generation + 1
        POP.bodyCount = POP.bodyCount
        G["POP_STRUC"] = POP

        num_seeds = pick_Seeds()
        write_log(f"            {int(num_seeds):4d} Seeds structures are added from Seeds/POSCARS\n")

        # Reinitialize job bookkeeping for the new generation.
        write_log(alignLine(f"Proceeding to the generation {POP.generation}") + "\n")
        write_log(f"     {int(ORG.numParallelCalcs):4d} parallel calculations are performed simutaneously\n")
        write_log(alignLine(f"Generation {POP.generation}") + "\n")
        write_log("  ID   Origin     Composition  Enthalpy(eV)  Volume(A^3)  KPOINTS  SYMMETRY\n")
        WriteGenerationStart()
        _im("gpevo.FunctionFolder.USPEX.310.Start_POP_310").Start_POP_310()

        safesave("Current_POP.pkl", POP)
        safesave("Current_ORG.pkl", ORG)
        console_print(" ")
        console_print("New generation produced")
    Finish()
