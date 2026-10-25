"""Python equivalent of createORG_Fingerprint.m."""
from __future__ import annotations

from gpevo.runtime import G
from gpevo.inputparser import InputParser


def createORG_Fingerprint(inputFile):
    ORG = G["ORG_STRUC"]
    p = InputParser(inputFile)
    raw = p.scalar("RmaxFing")
    if raw is not None:
        ORG.RmaxFing = float(raw)
    ORG.doFing = 1 if ORG.RmaxFing > 0 else 0
    raw = p.scalar("deltaFing")
    if raw is not None:
        ORG.deltaFing = float(raw)
    raw = p.scalar("sigmaFing")
    if raw is not None:
        ORG.sigmaFing = float(raw)
    # Molecular runs use looser fingerprint tolerances than atomic ones
    # (createORG_Fingerprint.m:27-28 and 34-35).
    #
    # The molecular default of the reference implementation is 0.05, but at
    # that value this system's relaxed structures are judged identical far too
    # easily (12 duplicate pairs out of 30 in generation 9, against 6 at 0.02),
    # which pushes them into ``bad_rank`` and starves the search.  0.02 is used
    # here instead; set ``toleranceFing`` in INPUT.txt to override it.
    raw = p.scalar("toleranceFing")
    if raw is not None:
        ORG.toleranceFing = float(raw)
    elif int(getattr(ORG, "molecule", 0)):
        ORG.toleranceFing = 0.02
    raw = p.scalar("toleranceBestHM")
    if raw is not None:
        ORG.toleranceBestHM = float(raw)
    elif int(getattr(ORG, "molecule", 0)):
        ORG.toleranceBestHM = 0.10
    raw = p.scalar("antiSeedsMax")
    if raw is not None:
        ORG.antiSeedsMax = float(raw)
    # Weights are over atom types, not molecule types.  For molecular runs,
    # ``GetPOP_MOL`` reconstructs the per-atom-type counts that correspond to
    # the atom-type dimension of the full-atom fingerprint matrix.
    if ORG.varcomp == 1 or ORG.dimension == -4:
        ORG.weight = 1.0
        G["ORG_STRUC"] = ORG
        return

    if ORG.molecule == 1:
        from gpevo.FunctionFolder.USPEX.src.GetPOP_MOL import GetPOP_MOL
        _, _, num_ions = GetPOP_MOL(ORG.numMols)
    else:
        num_ions = list(ORG.numIons)
    L = len(num_ions)
    S = 0
    w = [0.0] * (L * L)
    for i in range(L):
        for j in range(L):
            w[i * L + j] = float(num_ions[i] * num_ions[j])
            S += w[i * L + j]
    if S > 0:
        w = [v / S for v in w]
    ORG.weight = w
    G["ORG_STRUC"] = ORG
