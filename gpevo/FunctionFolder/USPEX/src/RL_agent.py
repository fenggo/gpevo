"""Python port of RL_agent.m (hierarchical contextual bandit, L2).

State lives in ``rl_agent_state.pkl`` next to the run; the cross-task prior is
``RLPriors/operator_prior.pkl`` then ``~/.config/uspex-gp/RLPriors/operator_prior.pkl``.
"""
from __future__ import annotations

import math
import os
import pickle
import re

import numpy as np

from gpevo.runtime import G
from gpevo.log_utils import console_print

STATE_FILE = "rl_agent_state.pkl"
PRIOR_NAME = "operator_prior.pkl"

KEYS = ["Heredity", "Random", "Permutate", "Rotate", "LatMutate", "softmutate"]
NP = [0, 0, 0, 2, 1, 1]
K = 6

ANGLE_LO = math.pi / 12
ANGLE_HI = math.pi / 2
TRANS_LO = 0.1
TRANS_HI = 0.8
MUT_LO = 0.05
MUT_HI = 0.30
LOG_SCALE_LO = math.log(0.5)
LOG_SCALE_HI = math.log(2.0)
INNER_BASE_SD = 0.05

# The four continuous amplitudes the inner sampler can move, as (arm, inner
# index): arm 4 is Rotate (rotation angle, rigid translation), arm 5 LatMutate
# (strain), arm 6 softmutate (step scale).  RL_inner names them one by one, so a
# run can let the bandit move just the rotation angle and keep the other three
# at whatever INPUT.txt set -- that is what makes a single-parameter ablation
# possible.
INNER_TOKENS = {
    "angle": [(4, 0)],
    "rotation": [(4, 0)],
    "rotationangle": [(4, 0)],
    "rotate": [(4, 0), (4, 1)],
    "translation": [(4, 1)],
    "trans": [(4, 1)],
    "latmut": [(5, 0)],
    "latmutate": [(5, 0)],
    "mutation": [(5, 0)],
    "mutationrate": [(5, 0)],
    "strain": [(5, 0)],
    "softmut": [(6, 0)],
    "softmutate": [(6, 0)],
    "softstep": [(6, 0)],
    "softstepscale": [(6, 0)],
}
INNER_ALL = {(4, 0), (4, 1), (5, 0), (6, 0)}
INNER_LABEL = {(4, 0): "rotationAngleMax", (4, 1): "translationMax",
               (5, 0): "mutationRate", (6, 0): "softStepScale"}


def _inner_params(ORG):
    """The (arm, inner index) pairs the sampler is allowed to move.

    RL_inner : angle restricts it to the rotation angle; the amplitudes left out
    keep the value INPUT.txt gave them.  Without RL_inner (or with all) every
    amplitude is sampled, which is the behaviour up to now.
    """
    raw = getattr(ORG, "RL_inner", None)
    if raw is None:
        return set(INNER_ALL)
    text = str(raw).strip().lower()
    if not text or text in ("all", "1", "on"):
        return set(INNER_ALL)
    if text in ("none", "off", "0"):
        return set()
    chosen = set()
    for token in re.split(r"[,;\s+]+", text):
        if not token:
            continue
        pairs = INNER_TOKENS.get(token)
        if pairs is None:
            console_print("  RL bandit: RL_inner token '" + token + "' unknown "
                          "(angle | translation | mutation | softstep | rotate |"
                          " latmut | softmut | all | none)")
            continue
        chosen.update(pairs)
    return chosen or set(INNER_ALL)


def which_arm(how_come):
    how_come = str(how_come or "")
    for index, key in enumerate(KEYS):
        if key in how_come:
            return index + 1
    return 0


def _u(value, lo, hi):
    """Map a physical value onto the unit interval, clamped."""
    try:
        u = (float(value) - lo) / (hi - lo)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.5
    return min(1.0, max(0.0, u))


def _fresh():
    # Start from the amplitudes configured in INPUT.txt (createORG_EA), so
    # `rotationAngleMax` / `translationMax` / `mutationRate` / `softStepScale`
    # set both the value used when the bandit is off and the point the bandit
    # starts exploring from.
    ORG = G.get("ORG_STRUC")
    if ORG is None:
        angle, trans, mut, scale = ANGLE_HI, 0.5, 0.5, 1.0
    else:
        angle = float(getattr(ORG, "rotationAngleMax", ANGLE_HI) or ANGLE_HI)
        trans = float(getattr(ORG, "translationMax", 0.5) or 0.5)
        mut = float(getattr(ORG, "mutationRate", 0.08) or 0.08)
        scale = float(getattr(ORG, "softStepScale", 1.0) or 1.0)

    inner = [
        [{"m": 0.5, "s": 0.25, "neff": 1.0} for _ in range(2)]
        for _ in range(K)
    ]
    inner[3][0]["m"] = _u(angle, ANGLE_LO, ANGLE_HI)
    inner[3][1]["m"] = _u(trans, TRANS_LO, TRANS_HI)
    inner[4][0]["m"] = _u(mut, MUT_LO, MUT_HI)
    inner[5][0]["m"] = _u(math.log(max(scale, 1e-12)), LOG_SCALE_LO, LOG_SCALE_HI)
    return {
        "alpha": np.ones(K, dtype=float),
        "beta": np.ones(K, dtype=float),
        "inner": inner,
        "lastU": np.full((K, 2), np.nan),
        "history": [],
        "version": 1,
    }


def _read_prior_file():
    home = os.path.expanduser("~")
    candidates = [
        os.path.join("RLPriors", PRIOR_NAME),
        os.path.join(home, ".config", "uspex-gp", "RLPriors", PRIOR_NAME),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as fh:
                prior = pickle.load(fh)
            if isinstance(prior, dict) and "alpha" in prior:
                return prior, path
        except Exception:
            continue
    return None, ""


def _fuse_prior(state):
    prior, source = _read_prior_file()
    if prior is None:
        return state
    try:
        state["alpha"] = np.asarray(state["alpha"], dtype=float) + np.maximum(
            0.0, np.asarray(prior["alpha"], dtype=float) - 1.0
        )
        state["beta"] = np.asarray(state["beta"], dtype=float) + np.maximum(
            0.0, np.asarray(prior["beta"], dtype=float) - 1.0
        )
        prior_inner = prior.get("inner")
        if prior_inner:
            for a in range(min(K, len(prior_inner))):
                for k in range(min(2, len(prior_inner[a]))):
                    state["inner"][a][k]["m"] = float(prior_inner[a][k]["m"])
                    state["inner"][a][k]["neff"] += float(prior_inner[a][k]["neff"])
        console_print(f"RL bandit: loaded cross-task prior from {source}")
    except Exception as exc:
        console_print(f"RL bandit: prior ignored ({exc})")
    return state


def safe_fractions(f, cap):
    """Normalise, cap every arm and redistribute the excess (RL_safe_fractions)."""
    f = np.asarray(f, dtype=float)
    f = f / f.sum()
    for _ in range(50):
        over = f > cap + 1e-12
        if not np.any(over):
            break
        excess = float(np.sum(f[over] - cap))
        f[over] = cap
        under = ~over
        room = float(np.sum(f[under]))
        if room <= 0:
            f[0] += excess
            break
        f[under] = f[under] + excess * (f[under] / room)
    return f / f.sum()


def RL_agent(popSize):
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]

    state = None
    if os.path.isfile(STATE_FILE):
        try:
            with open(STATE_FILE, "rb") as fh:
                loaded = pickle.load(fh)
            if isinstance(loaded, dict) and loaded.get("version") == 1:
                state = loaded
        except Exception:
            state = None
    if state is None:
        state = _fuse_prior(_fresh())

    lam = float(getattr(ORG, "RL_forget", 0.9))
    w1 = float(getattr(ORG, "RL_w1", 1.0))
    w2 = float(getattr(ORG, "RL_w2", 0.5))
    arm_cap = float(getattr(ORG, "RL_armCap", 0.6))
    kappa = float(getattr(ORG, "RL_kappa", 1.0))

    # ---- settle the generation that just finished -----------------------
    enthalpies = []
    for individual in POP.POPULATION:
        value = float(np.asarray(individual.Enthalpies, dtype=float).ravel()[-1])
        if value < 90000:
            enthalpies.append(value)
    if enthalpies:
        h_mean = float(np.mean(enthalpies))
        h_std = float(np.std(enthalpies))
    else:
        h_mean, h_std = 0.0, 1.0
    if not h_std > 0:
        h_std = 1.0

    ranking = [int(i) for i in getattr(POP, "ranking", None) or range(len(POP.POPULATION))]
    n_pro = max(1, int(round(len(POP.POPULATION) * float(getattr(ORG, "bestFrac", 0.5)))))
    elite = set(ranking[: min(n_pro, len(ranking))])

    trials = np.zeros(K, dtype=float)
    wins = np.zeros(K, dtype=float)
    last_u = np.asarray(state.get("lastU"), dtype=float)

    for index, individual in enumerate(POP.POPULATION):
        arm = which_arm(getattr(individual, "howCome", ""))
        if arm == 0:
            continue
        arm_index = arm - 1
        value = float(np.asarray(individual.Enthalpies, dtype=float).ravel()[-1])
        reward = 0.0
        if value < 90000:
            in_elite = 1.0 if index in elite else 0.0
            gain = min(1.0, max(0.0, (h_mean - value) / h_std))
            reward = w1 * in_elite + w2 * gain
        trials[arm_index] += 1.0
        wins[arm_index] += reward
        if reward > 0 and arm in (4, 5, 6):
            for k in range(2):
                u = float(last_u[arm_index, k])
                if math.isnan(u):
                    continue
                cell = state["inner"][arm_index][k]
                n_old = lam * float(cell["neff"])
                m_old = float(cell["m"])
                n_new = n_old + reward
                cell["m"] = (m_old * n_old + u * reward) / n_new
                cell["neff"] = n_new

    # Beta-Bernoulli needs successes in [0, trials].  ``reward`` is a real
    # score in [0, w1+w2] (elite flag + clipped enthalpy gain), so raw ``wins``
    # can exceed ``trials``; ``beta`` then goes negative, is clamped to 1e-3,
    # and theta ~ 1 makes that arm dominate forever.  Normalise to fractional
    # successes before the update (the inner Gaussian above keeps raw reward).
    scale = max(w1 + w2, 1e-12)
    successes = np.minimum(wins / scale, trials)
    state["alpha"] = np.maximum(
        lam * np.asarray(state["alpha"], dtype=float) + successes, 1e-3)
    state["beta"] = np.maximum(
        lam * np.asarray(state["beta"], dtype=float) + (trials - successes),
        1e-3)

    # ---- draw the next generation's fractions ---------------------------
    count = np.zeros(K, dtype=float)
    for _ in range(int(popSize)):
        theta = np.zeros(K, dtype=float)
        for a in range(K):
            x = np.random.gamma(state["alpha"][a])
            y = np.random.gamma(state["beta"][a])
            theta[a] = x / (x + y) if (x + y) > 0 else 0.0
        count[int(np.argmax(theta))] += 1.0
    f = count / max(int(popSize), 1)

    f[0] = max(f[0], 0.10)   # heredity
    f[1] = max(f[1], 0.10)   # random
    f[5] = max(f[5], 0.10)   # softmutate
    f = safe_fractions(f, arm_cap)

    # ---- inner continuous Thompson sampling for arms 4/5/6 ---------------
    # Only the amplitudes RL_inner selected; the others stay at the value
    # INPUT.txt set, so a run can ablate one of them on its own.
    inner_params = _inner_params(ORG)
    next_u = np.full((K, 2), np.nan)
    for arm, k in sorted(inner_params):
        cell = state["inner"][arm - 1][k]
        sd = kappa / math.sqrt(max(float(cell["neff"]), 1e-12)) + INNER_BASE_SD
        value = float(cell["m"]) + sd * float(np.random.randn())
        next_u[arm - 1, k] = min(1.0, max(0.0, value))
    state["lastU"] = next_u

    if (4, 0) in inner_params:
        ORG.rotationAngleMax = ANGLE_LO + float(next_u[3, 0]) * (ANGLE_HI - ANGLE_LO)
    if (4, 1) in inner_params:
        ORG.translationMax = TRANS_LO + float(next_u[3, 1]) * (TRANS_HI - TRANS_LO)
    if (5, 0) in inner_params:
        ORG.mutationRate = MUT_LO + float(next_u[4, 0]) * (MUT_HI - MUT_LO)
    if (6, 0) in inner_params:
        ORG.softStepScale = math.exp(LOG_SCALE_LO + float(next_u[5, 0])
                                     * (LOG_SCALE_HI - LOG_SCALE_LO))
    if inner_params != INNER_ALL:
        names = ", ".join(sorted(INNER_LABEL[pair] for pair in inner_params))
        console_print("  RL bandit: inner = " + (names if names else "none"))

    state.setdefault("history", []).append(
        {
            "gen": int(getattr(POP, "generation", 0)),
            "trials": trials.copy(),
            "wins": wins.copy(),          # raw reward sum, before normalising
            "successes": successes.copy(),  # what actually fed the Beta update
            "f": f.copy(),
            "u": next_u.copy(),
        }
    )
    state["version"] = 1

    from gpevo.runtime import safesave

    safesave(STATE_FILE, state)
    if int(getattr(ORG, "RL_autoExport", 0)):
        RL_export_prior(state)

    console_print("RL bandit: f = [" + " ".join(f"{v:.2f}" for v in f) + " ]")
    return f


def RL_export_prior(state):
    """Python equivalent of RL_export_prior.m."""
    try:
        directory = "RLPriors"
        os.makedirs(directory, exist_ok=True)
        prior = {
            "alpha": np.asarray(state["alpha"], dtype=float),
            "beta": np.asarray(state["beta"], dtype=float),
            "inner": state["inner"],
            "version": 1,
        }
        path = os.path.join(directory, PRIOR_NAME)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(prior, fh)
        os.replace(tmp, path)
    except Exception as exc:
        console_print(f"RL bandit: export failed ({exc})")
