"""Python port of WriteBest.m.

Rewrites ``resFolder/BESTIndividuals`` with one row per generation's best
structure(s), taken from ``USPEX_STRUC.GENERATION(i).BestID``.
"""
from __future__ import annotations

from gpevo.runtime import G
from gpevo.FunctionFolder.USPEX.src.WriteIndividual import HEADER, UNITS, _row


def WriteBest(resFolder, energyUnits=None):
    USPEX = G.get("USPEX_STRUC")
    if USPEX is None:
        return
    population = list(getattr(USPEX, "POPULATION", []) or [])
    generations = list(getattr(USPEX, "GENERATION", []) or [])
    path = f"{resFolder}/BESTIndividuals"
    try:
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(HEADER + "\n")
            fp.write(UNITS + "\n")
            for entry in generations:
                for number in list(entry.get("BestID", []) or []):
                    number = int(number)
                    if 0 < number <= len(population) and population[number - 1] is not None:
                        fp.write(_row(number, population[number - 1]) + "\n")
    except OSError:
        return
