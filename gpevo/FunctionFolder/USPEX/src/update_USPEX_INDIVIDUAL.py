"""Python port of update_USPEX_INDIVIDUAL.m.

Records a finished structure in the cumulative ``USPEX_STRUC.POPULATION``
(indexed by the global structure ``Number``); that record is what the
generation history and the composition statistics work on.
"""
from __future__ import annotations

import copy

import numpy as np

from gpevo.runtime import G, AttrDict, safesave

COMMON_FIELDS = [
    "COORDINATES", "LATTICE", "numIons", "FINGERPRINT", "order", "Parents",
    "S_order", "symg", "K_POINTS", "Enthalpies", "struc_entr", "howCome",
]
PRIVATE_FIELDS = [
    "cell", "numMols", "MOLECULES", "numBlocks", "Surface_numIons", "gap",
    "hardness", "powerfactor", "mag_moment", "magmom_ions", "magmom_ini",
    "ldaU", "dielectric_tensor", "elasticMatrix", "elasticProperties", "soap_fp",
]


def update_USPEX_INDIVIDUAL(individual, resFolder, gen, atomType):
    USPEX = G.get("USPEX_STRUC")
    if USPEX is None:
        USPEX = AttrDict(POPULATION=[], GENERATION=[])
        G["USPEX_STRUC"] = USPEX

    number = int(getattr(individual, "Number", 0) or 0)
    if number <= 0:
        return USPEX
    population = list(getattr(USPEX, "POPULATION", []) or [])
    while len(population) < number:
        population.append(None)
    while len(population) > number:
        population.pop()

    record = AttrDict()
    record.gen = gen
    record.ToCount = 1
    record.Fitness = 1000
    for field in COMMON_FIELDS:
        record[field] = copy.deepcopy(getattr(individual, field, None))
    for field in PRIVATE_FIELDS:
        if hasattr(individual, field):
            record[field] = copy.deepcopy(getattr(individual, field))

    if hasattr(individual, "Vol"):
        volume = individual.Vol
    else:
        try:
            volume = float(np.linalg.det(np.asarray(individual.LATTICE, dtype=float).reshape(3, 3)))
        except Exception:
            volume = 0.0
    record.Vol = volume

    from gpevo.FunctionFolder.sys.elementMass import calc_density

    try:
        record.density = calc_density(individual.numIons, atomType, volume)
    except Exception:
        record.density = None

    population[number - 1] = record
    USPEX.POPULATION = population
    G["USPEX_STRUC"] = USPEX
    return USPEX
