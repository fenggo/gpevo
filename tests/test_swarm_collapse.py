"""Regression tests for the swarm-collapse fix.

results2 gen2/gen3 produced 19/18 IDENTICAL Swarm offspring.  Root cause:
a full interpolation between two packings failed `valid_molecules` ~95% of
the time, so the only accepted "move" was parent == ranking[0] with zero
displacement -- an exact clone of the best structure.  The fix is threefold:

* enabled PSO coefficients by default (0.6 / 0.5);
* a per-offspring jitter independent of d_g / d_p, so a particle at g still
  moves, plus a hard guard that rejects any byte-identical child;
* a damped retry (1, 0.5, 0.25, 0.125) of the SAME transforms when the
  validity gate rejects a move, so genuinely moved offspring from different
  parents survive instead of only clones of g.
"""
import copy
import importlib

import numpy as np

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault
from gpevo.FunctionFolder.USPEX.src.createORG_EA import createORG_EA

_swarm = importlib.import_module(
    "gpevo.FunctionFolder.USPEX.310.SwarmMutation_310")

LATTICE = np.diag([10.0, 10.0, 10.0])


def _mol(coords):
    c = np.asarray(coords, dtype=float)
    return AttrDict(MOLCOORS=c.copy(), MOLCENTER=c.mean(0), ZMATRIX=c.copy())


def _ind(offset=0.0):
    # Four non-collinear atoms: Kabsch has a well-defined proper rotation.
    base = np.array([[0.0, 0.0, 0.0],
                     [1.2, 0.0, 0.0],
                     [0.0, 1.3, 0.0],
                     [0.0, 0.0, 1.4]], dtype=float)
    return AttrDict(
        MOLECULES=[_mol(base + offset), _mol(base + offset + np.array([5.0, 5.0, 5.0]))],
        MtypeLIST=[1, 1],
        LATTICE=LATTICE.copy(),
    )


def _wire(monkeypatch, population, ranking, parent_idx, *, jitter=0.05,
          inertia=0.6, cognitive=0.5, step=0.5, angle=30.0, valid=None):
    produced = []
    monkeypatch.setitem(G, "ORG_STRUC", AttrDict(
        swarmStep=step, swarmAngle=angle,
        swarmInertia=inertia, swarmCognitive=cognitive,
        swarmJitter=jitter))
    monkeypatch.setitem(G, "POP_STRUC",
                        AttrDict(POPULATION=population, ranking=ranking))
    monkeypatch.setitem(G, "OFF_STRUC", AttrDict(POPULATION=[]))
    monkeypatch.setattr(_swarm, "pick_parent", lambda: parent_idx)
    monkeypatch.setattr(_swarm, "copy_individual", lambda ind: copy.deepcopy(ind))
    monkeypatch.setattr(_swarm, "recompute_zmatrix", lambda *a, **k: None)
    monkeypatch.setattr(_swarm, "finalize_offspring",
                        lambda child, how: produced.append(child))
    monkeypatch.setattr(_swarm, "valid_molecules",
                        valid if valid is not None else (lambda *a, **k: True))
    _rand = importlib.import_module(
        "gpevo.FunctionFolder.USPEX.310.Random_310")
    rand_calls = []
    monkeypatch.setattr(_rand, "Random_310",
                        lambda count: rand_calls.append(count))
    return produced, rand_calls


def _sig(ind):
    return tuple(np.concatenate([
        np.asarray(m.MOLCOORS, dtype=float).ravel()
        for m in ind.MOLECULES]).round(6))


def _org(monkeypatch):
    """A default ORG with the few fields createORG_System normally supplies."""
    ORG = createORGDefault()
    ORG.numIons = [8]
    ORG.numMols = [8]
    ORG.atomType = [6, 8, 7, 1]
    ORG.molecule = 1
    monkeypatch.setitem(G, "ORG_STRUC", ORG)
    return ORG


# ---------------------------------------------------------------------------
# Defaults / INPUT parsing
# ---------------------------------------------------------------------------

def test_defaults_enable_pso_and_jitter():
    ORG = createORGDefault()
    assert ORG.swarmInertia == 0.6
    assert ORG.swarmCognitive == 0.5
    assert ORG.swarmJitter == 0.05
    assert ORG.swarmStep == 0.5          # unchanged, doc derivation anchor


def test_createORG_EA_parses_all_swarm_options(tmp_path, monkeypatch):
    inp = tmp_path / "INPUT.txt"
    inp.write_text(
        "0.02 : swarmJitter\n"
        "0.4  : swarmInertia\n"
        "0.3  : swarmCognitive\n"
        "0.8  : swarmStep\n")
    ORG = _org(monkeypatch)
    createORG_EA(str(inp))
    assert (ORG.swarmJitter, ORG.swarmInertia,
            ORG.swarmCognitive, ORG.swarmStep) == (0.02, 0.4, 0.3, 0.8)


def test_explicit_zero_still_disables(tmp_path, monkeypatch):
    inp = tmp_path / "INPUT.txt"
    inp.write_text("0 : swarmInertia\n0 : swarmCognitive\n0 : swarmJitter\n")
    ORG = _org(monkeypatch)
    createORG_EA(str(inp))
    assert (ORG.swarmInertia, ORG.swarmCognitive, ORG.swarmJitter) == (0, 0, 0)


def test_legacy_checkpoint_missing_fields_gets_jitter(tmp_path, monkeypatch):
    """A checkpoint saved before swarmJitter existed still gets the fix."""
    inp = tmp_path / "INPUT.txt"
    inp.write_text("")
    ORG = _org(monkeypatch)
    del ORG["swarmJitter"]
    createORG_EA(str(inp))
    assert ORG.swarmJitter == 0.05


def test_restart_keeps_explicit_checkpoint_coefficients(tmp_path, monkeypatch):
    """Documented caveat: a checkpoint that already has 0/0 keeps it.

    Restarts must add `0.6 : swarmInertia` / `0.5 : swarmCognitive` to
    INPUT.txt to pick up the enabled coefficients.
    """
    inp = tmp_path / "INPUT.txt"
    inp.write_text("")
    ORG = _org(monkeypatch)
    ORG.swarmInertia = 0.0
    ORG.swarmCognitive = 0.0
    createORG_EA(str(inp))
    assert ORG.swarmInertia == 0.0 and ORG.swarmCognitive == 0.0


# ---------------------------------------------------------------------------
# No exact clones
# ---------------------------------------------------------------------------

def test_jitter_makes_siblings_of_the_best_distinct(monkeypatch):
    """parent == g, fresh state, all PSO terms zero -> clones before the fix."""
    ind = _ind()
    produced, rand_calls = _wire(
        monkeypatch, [ind], [0], 0,
        jitter=0.05, inertia=0.0, cognitive=0.0, step=0.0)
    np.random.seed(2024)
    for i in range(8):
        _swarm.SwarmMutation_310(i + 1)
    assert len(produced) == 8
    assert len({_sig(p) for p in produced}) == 8
    assert not rand_calls                 # no Random fallback needed


def test_zero_jitter_zero_guidance_never_emits_a_clone(monkeypatch):
    """The clone guard must reject zero-displacement children outright."""
    ind = _ind()
    produced, rand_calls = _wire(
        monkeypatch, [ind], [0], 0,
        jitter=0.0, inertia=0.0, cognitive=0.0, step=0.0)
    np.random.seed(11)
    _swarm.SwarmMutation_310(1)
    assert produced == []                 # exact clone rejected at every damp
    assert rand_calls == [1]             # falls back to Random once


# ---------------------------------------------------------------------------
# Damped retry
# ---------------------------------------------------------------------------

def test_damped_retry_reuses_same_move_at_lower_amplitude(monkeypatch):
    """A rejected full move is retried with damp<1, not thrown away."""
    parent = _ind()
    best = _ind(offset=np.array([1.0, 0.5, -0.7]))
    base = [np.asarray(m.MOLCOORS, dtype=float).copy()
            for m in parent.MOLECULES]
    seen = []

    def _valid(child, tol):
        d = max(float(np.max(np.abs(np.asarray(m.MOLCOORS, dtype=float) - base[k])))
                for k, m in enumerate(child.MOLECULES))
        seen.append(d)
        return len(seen) >= 3           # reject damp 1.0 and 0.5

    produced, rand_calls = _wire(
        monkeypatch, [parent, best], [1], 0, jitter=0.0,
        inertia=0.0, cognitive=0.0, step=0.5, angle=30.0, valid=_valid)
    np.random.seed(5)
    _swarm.SwarmMutation_310(1)
    assert len(produced) == 1
    assert not rand_calls
    assert len(seen) == 3
    assert seen[-1] < seen[0], seen      # accepted move is damped


def test_full_move_unchanged_when_valid_on_first_try(monkeypatch):
    """damp=1.0 must reproduce the original single-pass behaviour."""
    parent = _ind()
    best = _ind(offset=np.array([1.0, 0.5, -0.7]))
    seen = []

    def _valid(child, tol):
        seen.append(child)
        return True

    produced, _ = _wire(
        monkeypatch, [parent, best], [1], 0, jitter=0.0,
        inertia=0.0, cognitive=0.0, step=0.5, angle=30.0, valid=_valid)
    np.random.seed(5)
    _swarm.SwarmMutation_310(1)
    assert len(produced) == 1
    assert len(seen) == 1                # accepted at damp 1.0, no retries
    centre = np.asarray(produced[0].MOLECULES[0].MOLCOORS, dtype=float).mean(0)
    assert np.linalg.norm(centre - np.asarray(parent.MOLECULES[0].MOLCENTER)) > 1e-6
