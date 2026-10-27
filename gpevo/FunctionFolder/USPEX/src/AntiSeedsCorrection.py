"""Python equivalent of AntiSeedsCorrection.m.

Anti-seeds repel the search from regions of fingerprint space that are already
over-sampled: every structure whose fingerprint is close to an anti-seed gets
a Gaussian bump added to its fitness.

Direct port of ``../uspex-gp/FunctionFolder/USPEX/src/AntiSeedsCorrection.m``
(line numbers below refer to that file).  The MATLAB global ``ANTISEEDS`` is
``G["ANTISEEDS"]``: a list of ``{"FINGERPRINT", "Sigma", "Max"}`` entries.

Note that this is inert unless ``antiSeedsMax`` is set in INPUT.txt — with the
default 0 every derived ``Max`` is 0 and the fitness vector comes back
unchanged, which is how the minimal port behaved.
"""

from __future__ import annotations

import math

import numpy as np

from gpevo.log_utils import console_print
from gpevo.runtime import G, safesave

def _is_empty_fp(f):
    """True for missing/empty fingerprints — MATLAB's ``isempty``."""
    if f is None:
        return True
    arr = np.asarray(f, dtype=np.float64)
    return arr.size == 0


def _cos_dist(f1, f2, weight):
    """Cosine distance between two fingerprints, 1 when either is empty.

    ``weight`` may be the scalar ``1.0`` (``varcomp == 1``) or the ORG weight
    vector; it is reshaped to the fingerprint row count the same way
    ``KeepBestStructures._distance`` does, because ``_cosine_distance`` runs it
    through ``np.diag``.
    """
    if _is_empty_fp(f1) or _is_empty_fp(f2):
        return 1.0
    from gpevo.FunctionFolder.sys.fingerprint_utils import _cosine_distance

    rows = np.asarray(f1, dtype=np.float64).shape[0]
    w = np.asarray(weight, dtype=np.float64)
    if w.ndim == 0 or w.size != rows:
        w = np.ones(rows, dtype=np.float64)
    return float(_cosine_distance(f1, f2, w))


def AntiSeedsCorrection(fitness):
    """Add the anti-seed penalty to ``fitness`` and possibly learn new ones."""
    ORG = G.get("ORG_STRUC")
    POP = G.get("POP_STRUC")
    if ORG is None or POP is None:
        return np.asarray(fitness, dtype=np.float64)

    population = list(getattr(POP, "POPULATION", []) or [])
    generation = int(getattr(POP, "generation", 0) or 0)
    anti_seeds = G.get("ANTISEEDS") or []

    out = np.asarray(fitness, dtype=np.float64).copy()

    activation = int(getattr(ORG, "antiSeedsActivation", 0) or 0)
    can_append = activation != 0 and generation >= abs(activation)
    # Nothing to penalise and nothing to learn -> identical to the reference.
    if not anti_seeds and not can_append:
        return out

    n = len(population)
    if n == 0:
        return out

    # :5  how many of the best structures define the current best-region
    fit_len = int(round(len(out) * float(getattr(ORG, "bestFrac", 0.7))))
    fit_len = max(0, min(fit_len, len(out)))

    # :6  adaptive amplitude: proportional to the fitness spread of the best part
    body = out[:fit_len] if fit_len else np.array([], dtype=np.float64)
    min_fit = float(np.min(out)) if len(out) else 0.0
    if fit_len:
        anti_seeds_max = float(getattr(ORG, "antiSeedsMax", 0.0)) * (
            float(np.sum(body)) / fit_len - min_fit
        )
    else:
        anti_seeds_max = 0.0

    # :7-11 never let the amplitude decay faster than half its previous value
    if anti_seeds and generation > activation + 1:
        floor = 0.5 * float(anti_seeds[-1].get("Max", 0.0) or 0.0)
        if anti_seeds_max < floor:
            anti_seeds_max = floor

    # :14-35 width from the mean pairwise distance among the best structures
    weight = 1.0 if int(getattr(ORG, "varcomp", 0) or 0) == 1 else getattr(ORG, "weight", 1.0)
    count = 0
    anti_seeds_sigma = 0.0
    for fit1 in range(fit_len - 1):
        f1 = getattr(population[fit1], "FINGERPRINT", None)
        for fit2 in range(fit1 + 1, fit_len):
            f2 = getattr(population[fit2], "FINGERPRINT", None)
            try:
                anti_seeds_sigma += _cos_dist(f1, f2, weight)
            except Exception:
                anti_seeds_sigma += 1.0
            count += 1
    if fit_len < 2 or anti_seeds_sigma < 1e-15:
        anti_seeds_sigma = 0.001
        count = 1
    anti_seeds_sigma = float(getattr(ORG, "antiSeedsSigma", 0.001)) * anti_seeds_sigma / count

    # :36-41 first generation fixes the scale of every pre-loaded anti-seed
    if generation == 1:
        for entry in anti_seeds:
            entry["Sigma"] = anti_seeds_sigma
            entry["Max"] = anti_seeds_max

    # :42-61 the penalty itself
    if anti_seeds and not _is_empty_fp(anti_seeds[0].get("FINGERPRINT")):
        warned_shape = False
        for entry in anti_seeds:
            f1 = entry.get("FINGERPRINT")
            sigma = float(entry.get("Sigma", 0.0) or 0.0)
            max_val = float(entry.get("Max", 0.0) or 0.0)
            if sigma <= 0.0 or max_val == 0.0:
                continue                # :57 would contribute nothing anyway
            denom = 2.0 * sigma * sigma
            for i in range(n):
                f2 = getattr(population[i], "FINGERPRINT", None)
                try:
                    cos_dist = _cos_dist(f1, f2, weight)
                except Exception:
                    cos_dist = 1.0
                    if not warned_shape:
                        console_print(
                            "  Warning: fingerprint shapes differ, anti-seed "
                            "penalty treated as maximal for those pairs"
                        )
                        warned_shape = True
                out[i] += max_val * math.exp(-(cos_dist * cos_dist) / denom)

    # :62-80 learn new anti-seeds once the generation is past the activation
    if can_append:
        if activation > 0:
            for individual in population:
                anti_seeds.append(
                    {
                        "FINGERPRINT": getattr(individual, "FINGERPRINT", None),
                        "Sigma": anti_seeds_sigma,
                        "Max": anti_seeds_max,
                    }
                )
        else:
            best = int(np.argmin(out)) if len(out) else 0
            anti_seeds.append(
                {
                    "FINGERPRINT": getattr(population[best], "FINGERPRINT", None),
                    "Sigma": anti_seeds_sigma,
                    "Max": anti_seeds_max,
                }
            )

    # :81-86 drop the empty placeholder and persist
    if anti_seeds and _is_empty_fp(anti_seeds[0].get("FINGERPRINT")):
        anti_seeds.pop(0)

    G["ANTISEEDS"] = anti_seeds
    home = getattr(ORG, "homePath", None)
    try:
        from gpevo.FunctionFolder.USPEX.src.pickAntiSeeds import ANTISEEDS_FILE

        safesave(ANTISEEDS_FILE, anti_seeds, workdir=home)
    except Exception:
        pass

    return out
