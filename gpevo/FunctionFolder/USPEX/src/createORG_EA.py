"""Python equivalent of createORG_EA.m (310 path, AutoFrac/static)."""
from __future__ import annotations

import math

from gpevo.runtime import G
from gpevo.inputparser import InputParser


def _flo(value, default):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def createORG_EA(inputFile):
    ORG = G["ORG_STRUC"]
    p = InputParser(inputFile)

    raw = p.scalar("numGenerations")
    if raw is not None:
        ORG.numGenerations = max(1, int(float(raw)))

    raw = p.scalar("stopCrit")
    if raw is not None:
        ORG.stopCrit = float(raw)
    else:
        ORG.stopCrit = float(sum(ORG.numIons)) * (3 if ORG.molecule else 1)

    raw = p.scalar("populationSize")
    if raw is None:
        pS = 10 * round(2 * sum(ORG.numIons) / 10)
        pS = min(60, max(10, pS))
    else:
        pS = int(float(raw))
    ORG.populationSize = pS

    ORG.bestFrac = _flo(p.scalar("bestFrac"), ORG.bestFrac)
    raw = p.scalar("AutoFrac")
    if raw is not None:
        ORG.AutoFrac = int(float(raw))

    ORG.fracGene = _flo(p.scalar("fracGene"), ORG.fracGene)
    ORG.fracRand = _flo(p.scalar("fracRand"), ORG.fracRand)
    raw = p.scalar("fracPerm")
    if raw is None:
        raw = 0.1 if len(ORG.atomType) > 1 and not ORG.varcomp else 0.0
    ORG.fracPerm = _flo(raw, ORG.fracPerm)
    ORG.fracLatMut = _flo(p.scalar("fracLatMut"), 0.0 if ORG.constLattice else 0.1)
    ORG.fracAtomsMut = _flo(p.scalar("fracAtomsMut"), ORG.fracAtomsMut)
    if ORG.molecule == 1:
        ORG.fracRotMut = _flo(p.scalar("fracRotMut"), 0.1)
    else:
        ORG.fracRotMut = 0.0

    ORG.howManyMut = _flo(p.scalar("mutationDegree"), ORG.howManyMut)
    ORG.mutationRate = _flo(p.scalar("mutationRate"), ORG.mutationRate)

    # Rigid-body swarm move.  `p.option` accepts both "0.1 : fracSwarm" and
    # "fracSwarm : 0.1"; all three default to the values set by
    # createORGDefault, i.e. disabled.
    ORG.fracSwarm = _flo(p.option("fracSwarm"), ORG.fracSwarm)
    ORG.swarmStep = _flo(p.option("swarmStep"), ORG.swarmStep)
    ORG.swarmAngle = _flo(p.option("swarmAngle"), ORG.swarmAngle)
    ORG.swarmInertia = _flo(p.option("swarmInertia"), ORG.swarmInertia)
    ORG.swarmCognitive = _flo(p.option("swarmCognitive"), ORG.swarmCognitive)

    # Continuous mutation amplitudes.  They are fixed when the RL bandit is
    # off and become the *starting point* of the RL inner parameters when it
    # is on (RL_agent._fresh maps them onto the unit interval).  The angle is
    # written in degrees; ORG.rotationAngleMax itself is radians.
    raw = p.option("rotationAngleMax")
    if raw is not None and raw != "":
        ORG.rotationAngleMax = math.radians(float(raw))
    ORG.translationMax = _flo(p.option("translationMax"), ORG.translationMax)
    ORG.softStepScale = _flo(p.option("softStepScale"), ORG.softStepScale)

    raw = p.scalar("softMutTill")
    ORG.softMutTill = int(float(raw)) if raw is not None else ORG.numGenerations
    ORG.softMutOnly = [0] * (ORG.numGenerations + 1)

    raw = p.scalar("numberparents")
    if raw is not None:
        ORG.numParents = int(float(raw))
    if ORG.numParents > 2:
        ORG.manyParents = 1

    ORG.minSlice = int(_flo(p.scalar("minSlice"), 2))
    ORG.maxSlice = int(_flo(p.scalar("maxSlice"), 8))
    ORG.percSliceShift = _flo(p.scalar("percSliceShift"), 1.0)
    ORG.maxDistHeredity = _flo(p.scalar("maxDistHeredity"), 0.5)

    raw = p.scalar("keepBestHM")
    ORG.keepBestHM = int(_flo(raw, max(1, round(0.15 * ORG.populationSize))))
    ORG.keepBestHM = max(1, ORG.keepBestHM)

    raw = p.scalar("RL_strategy")
    if raw is not None:
        ORG.RL_strategy = int(float(raw))
    # Which continuous amplitudes the inner bandit may move: "angle" leaves
    # translationMax / mutationRate / softStepScale at their INPUT.txt value,
    # which is what a single-parameter ablation needs.  Free text, so it is
    # read with p.option (both "angle : RL_inner" and "RL_inner : angle" work).
    raw = p.option("RL_inner")
    if raw is not None:
        ORG.RL_inner = str(raw)
    for name, cast in (
        ("RL_forget", float), ("RL_w1", float), ("RL_w2", float),
        ("RL_armCap", float), ("RL_kappa", float), ("RL_autoExport", int),
    ):
        raw = p.scalar(name)
        if raw is None:
            continue
        try:
            setattr(ORG, name, cast(float(raw)))
        except (TypeError, ValueError):
            pass

    # Static / AutoFrac count calculation (update_STUFF_old logic)
    N = ORG.populationSize
    how_atom = round(N * ORG.fracAtomsMut)
    how_perm = round(N * ORG.fracPerm)
    how_gene = round(N * ORG.fracGene)
    how_rand = round(N * ORG.fracRand)
    how_rot = round(N * ORG.fracRotMut)
    how_trans = 0
    total = how_atom + how_perm + how_gene + how_rand + how_rot + how_trans
    if total > N:
        def scale(k):
            return int(round(k * N / total))
        how_atom, how_perm, how_gene, how_rand, how_rot, how_trans = map(scale, [how_atom, how_perm, how_gene, how_rand, how_rot, how_trans])
        total = sum([how_atom, how_perm, how_gene, how_rand, how_rot, how_trans])
    how_left = max(N - total, 0)
    if ORG.constLattice:
        how_gene += how_left
        how_mut = 0
    else:
        how_mut = how_left

    ORG.howManyOffsprings = how_gene
    ORG.howManyRand = how_rand
    ORG.howManyPermutations = how_perm
    ORG.howManyRotations = how_rot
    ORG.howManyMutations = how_mut
    ORG.howManyAtomMutations = how_atom
    ORG.howManyTrans = how_trans
    # Swarm offspring are additional: they are not part of the `total` split
    # above, so enabling fracSwarm never changes the existing counts.
    ORG.howManySwarm = int(round(N * max(0.0, float(ORG.fracSwarm or 0.0))))
    G["ORG_STRUC"] = ORG
