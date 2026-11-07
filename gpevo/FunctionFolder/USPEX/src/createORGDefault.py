"""Python equivalent of FunctionFolder/USPEX/src/createORGDefault.m."""
from __future__ import annotations

import math
import os

from gpevo.runtime import G, home_path, package_root, struct


def createORGDefault():
    ORG = struct()
    ORG.homePath = home_path()
    ORG.packageRoot = package_root()

    ORG.fixRndSeed = 0
    ORG.collectForces = 0
    ORG.PhaseDiagram = 0
    ORG.checkMolecules = 1
    ORG.symmetrize = 0
    ORG.cor_dir = 1
    ORG.correlation_coefficient = 1
    ORG.averageFitness = 1000000
    ORG.constLattice = 0
    ORG.varcomp = 0
    ORG.ldaU = 0
    ORG.softMutOnly = 0
    ORG.splitN = 0
    ORG.vacuumSize = 0
    ORG.maxErrors = 2
    ORG.sym_coef = 1
    ORG.chargeNeutrality = 0
    ORG.splitInto = []
    ORG.repeatForStatistics = 1
    ORG.ordering = 1
    ORG.minAngle = 55
    ORG.minDiagAngle = 30
    ORG.bestFrac = 0.7
    ORG.alpha = -1
    ORG.mutationRate = 0.08
    ORG.numGenerations = 100
    ORG.percSliceShift = 1.0
    ORG.dynamicalBestHM = 2
    ORG.reoptOld = 0
    ORG.pickUpYN = 0
    ORG.pickUpGen = 1
    ORG.pickUpFolder = 1
    ORG.pickUpNCount = 0
    ORG.remote = 0
    ORG.numParallelCalcs = 1
    ORG.pluginType = 0
    ORG.startNextGen = 1
    ORG.currentGenDone = 0
    ORG.log_file = "OUTPUT.txt"
    ORG.specificFolder = "Specific"
    ORG.abinitioCode = [1]
    ORG.platform = 0
    ORG.optType = 1
    # Gaussian-process (uspexkit gp) options: `1 : gp`, `24 : ncpu`,
    # `data : data`, `den : 1.8`, `0.04 : u`, `2 : k`, `ei : mode`
    ORG.gp = 0
    ORG.gpData = "data"
    ORG.gpDen = 0.0
    ORG.gpNcpu = 0          # 0 = fall back to % numProcessors
    ORG.gpU = 0.04          # uncertainty threshold of the active learning
    ORG.gpK = 1             # top-K crystals selected per generation
    ORG.gpMode = "ei"       # ei | random | uncertainty
    ORG.gpNboot = 2         # structures labelled to seed an empty training set
    ORG.gpExplore = 1       # 0 drops the sigma*phi(Z) term of EI: no exploration
    ORG.opt_sign = 1
    ORG.dimension = 3
    ORG.molecule = 0
    ORG.varcomp = 0
    ORG.maxDistHeredity = 0.5
    ORG.SGtolerance = 0.10
    ORG.RmaxFing = 10
    ORG.deltaFing = 0.08
    ORG.sigmaFing = 0.03
    ORG.toleranceFing = 0.008
    ORG.toleranceBestHM = 0.02
    ORG.erf_table = [math.erf((i - 402) / 100) for i in range(803)]
    ORG.antiSeedsMax = 0
    ORG.antiSeedsSigma = 0.001
    ORG.antiSeedsActivation = 5000
    ORG.reconstruct = 1
    ORG.thicknessS = 2.0
    ORG.thicknessB = 3.0
    ORG.manyParents = 0
    ORG.numParents = 2
    ORG.minSlice = 2
    ORG.maxSlice = 8
    ORG.AutoFrac = 0
    ORG.RL_strategy = 0
    ORG.RL_forget = 0.9
    ORG.RL_w1 = 1.0
    ORG.RL_w2 = 0.5
    ORG.RL_armCap = 0.6
    ORG.RL_kappa = 1.0
    ORG.RL_autoExport = 1
    ORG.RL_inner = "all"          # inner amplitudes the bandit may move
    ORG.rotationAngleMax = math.pi / 2
    ORG.translationMax = 0.5
    # Rigid-body swarm move (`fracSwarm : ...`): 0 disables it, so the default
    # operator mix is unchanged.  Its offspring are produced *in addition* to
    # populationSize, like the elitist additions.
    # `cache : 0` makes pred() recompute every 10-D feature instead of reusing
    # the one the GP step already computed (1 = reuse, the default).
    ORG.gpCache = 1
    ORG.fracSwarm = 0.0
    ORG.howManySwarm = 0
    ORG.swarmStep = 0.5          # fraction of the displacement toward the best
    ORG.swarmAngle = 30.0        # max orientation slerp toward g, degrees
    ORG.swarmInertia = 0.0       # PSO inertia weight w (0 = no momentum)
    ORG.swarmCognitive = 0.0     # PSO cognitive pull toward pbest, c1 (0 = off)
    ORG.softStepScale = 1.0
    ORG.fracGene = 0.5
    ORG.fracRand = 0.2
    ORG.fracAtomsMut = 0.1
    ORG.fracLatMut = 0.1
    ORG.fracRotMut = 0.1
    ORG.fracTrans = 0.1
    ORG.fracPerm = 0.1
    ORG.mlPeerFilter = 0
    ORG.firstGeneSplitAll = []
    ORG.minAt = 0
    ORG.maxAt = 0
    ORG.stopFitness = None
    ORG.stopCrit = 10
    ORG.populationSize = 30
    ORG.initialPopSize = 30
    ORG.keepBestHM = None
    ORG.howManyOffsprings = 6
    ORG.howManyRand = 2
    ORG.howManyPermutations = 3
    ORG.howManyRotations = 2
    ORG.howManyMutations = 3
    ORG.howManyAtomMutations = 4
    ORG.howManyMut = 3
    ORG.howManySwaps = 0
    ORG.numMols = []
    ORG.STDMOL = []
    ORG.CenterminDistMatrice = []
    ORG.minDistMatrice = []
    ORG.goodBonds = None       # set from calcDefaultGoodBonds in createORG_System
    ORG.valences = []          # per-species formal valences (soft modes)
    ORG.NvalElectrons = []     # per-species valence electron counts (soft modes)
    ORG.nsym = []
    ORG.resFolder = ""
    ORG.pickedUP = 0
    ORG.weight = []
    ORG.doFing = 1
    ORG.ExternalPressure = 0.0
    ORG.commandExecutable = []
    ORG.remoteFolder = ""
    ORG.Kresol = []
    ORG.numProcessors = []
    # Set by createORG_system
    return ORG
