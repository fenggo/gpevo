"""Python equivalent of update_STUFF_old.m (static operator fractions)."""
from __future__ import annotations

from gpevo.runtime import G


def _frac(ORG, name, default=0.0):
    value = getattr(ORG, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def update_STUFF_old(inputFile, goodFrac=None, ranking=None):
    ORG = G["ORG_STRUC"]

    if len(ORG.numIons) == 1:
        ORG.fracPerm = 0.0
        ORG.fracGene = _frac(ORG, "fracGene") + ORG.fracPerm

    population = int(ORG.populationSize)
    how_gene = int(round(population * _frac(ORG, "fracGene")))
    how_perm = int(round(population * _frac(ORG, "fracPerm")))
    how_soft = int(round(population * _frac(ORG, "fracAtomsMut")))
    how_rand = int(round(population * _frac(ORG, "fracRand")))
    how_rot = int(round(population * _frac(ORG, "fracRotMut")))
    how_trans = int(round(population * _frac(ORG, "fracTrans")))
    how_switch = int(round(population * _frac(ORG, "fracSecSwitch")))
    how_border = int(round(population * _frac(ORG, "fracShiftBorder")))

    total = how_gene + how_perm + how_soft + how_rand + how_rot + how_trans + how_switch + how_border
    if total > population and total > 0:
        scale = population / total
        how_gene = int(round(how_gene * scale))
        how_perm = int(round(how_perm * scale))
        how_soft = int(round(how_soft * scale))
        how_rand = int(round(how_rand * scale))
        how_rot = int(round(how_rot * scale))
        how_trans = int(round(how_trans * scale))
        how_switch = int(round(how_switch * scale))
        how_border = int(round(how_border * scale))
        total = how_gene + how_perm + how_soft + how_rand + how_rot + how_trans + how_switch + how_border

    left = max(population - total, 0)
    if not int(getattr(ORG, "constLattice", 0)):
        how_lat = left
    else:
        how_gene += left
        how_lat = 0

    ORG.howManyOffsprings = how_gene
    ORG.howManyPermutations = how_perm
    ORG.howManyAtomMutations = how_soft
    ORG.howManyRand = how_rand
    ORG.howManyRotations = how_rot
    ORG.howManyMutations = how_lat
    ORG.howManyTrans = how_trans
    ORG.howManySecSwitch = how_switch
    ORG.howManyShiftBorder = how_border
    G["ORG_STRUC"] = ORG
    return ORG
