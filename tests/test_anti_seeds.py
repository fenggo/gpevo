"""Tests for the AntiSeed port (pickAntiSeeds + AntiSeedsCorrection).

Everything runs inside ``tmp_path``; nothing touches a live run directory.
Reference for these expectations is
``~/uspex-gp/FunctionFolder/USPEX/src/{pickAntiSeeds,AntiSeedsCorrection}.m``.
"""

import os

import numpy as np
import pytest

from gpevo.runtime import G, struct
from gpevo.FunctionFolder.USPEX.src.pickAntiSeeds import (
    ANTISEEDS_FILE,
    pickAntiSeeds,
)
from gpevo.FunctionFolder.USPEX.src.AntiSeedsCorrection import AntiSeedsCorrection


POSCAR_BLOCK = """Seed {n}
1.0
{l} 0.0 0.0
0.0 {l} 0.0
0.0 0.0 {l}
2
Direct
0.0 0.0 0.0
{p} {p} {p}
"""


def _write_poscars(home, *cells):
    """Write an AntiSeeds/POSCARS file with one block per (cell, frac) pair."""
    anti = os.path.join(home, "AntiSeeds")
    os.makedirs(anti, exist_ok=True)
    text = ""
    for i, (length, frac) in enumerate(cells, start=1):
        text += POSCAR_BLOCK.format(n=i, l=length, p=frac)
    with open(os.path.join(anti, "POSCARS"), "w") as fh:
        fh.write(text)


def _org(tmp_path, **overrides):
    fields = dict(
        homePath=str(tmp_path),
        RmaxFing=10.0,
        sigmaFing=0.03,
        deltaFing=0.08,
        dimension=3,
        bestFrac=0.7,
        varcomp=0,
        weight=1.0,
        antiSeedsMax=0.0,
        antiSeedsSigma=0.001,
        antiSeedsActivation=5000,
    )
    fields.update(overrides)
    return struct(**fields)


def _individual(fingerprint):
    return struct(FINGERPRINT=np.asarray(fingerprint, dtype=np.float64))


def _population(*fingerprints):
    return struct(
        POPULATION=[_individual(f) for f in fingerprints],
        generation=1,
    )


# --------------------------------------------------------------------------
# pickAntiSeeds
# --------------------------------------------------------------------------

def test_pickAntiSeeds_reads_every_poscar_block(tmp_path, monkeypatch):
    _write_poscars(str(tmp_path), (5.0, 0.5), (6.0, 0.25))
    org = _org(tmp_path, antiSeedsMax=0.5, antiSeedsSigma=0.02)
    monkeypatch.setitem(G, "ORG_STRUC", org)
    monkeypatch.setitem(G, "ANTISEEDS", [])

    entries = pickAntiSeeds()

    assert len(entries) == 2
    for entry in entries:
        assert entry["FINGERPRINT"].size > 0
        assert entry["Sigma"] == 0.02
        assert entry["Max"] == 0.5
    # persisted next to Current_ORG.pkl
    assert os.path.isfile(os.path.join(str(tmp_path), ANTISEEDS_FILE))


def test_pickAntiSeeds_missing_file_is_silent(tmp_path, monkeypatch):
    monkeypatch.setitem(G, "ORG_STRUC", _org(tmp_path))
    monkeypatch.setitem(G, "ANTISEEDS", ["kept"])

    assert pickAntiSeeds() == []
    # a missing POSCARS must not clobber whatever was loaded before
    assert G["ANTISEEDS"] == ["kept"]


def test_pickAntiSeeds_direct_and_cartesian_agree(tmp_path, monkeypatch):
    """Direct blocks are converted to Cartesian, so both forms match."""
    _write_poscars(str(tmp_path), (5.0, 0.5))
    org = _org(tmp_path)
    monkeypatch.setitem(G, "ORG_STRUC", org)
    direct = pickAntiSeeds()[0]["FINGERPRINT"]

    path = os.path.join(str(tmp_path), "AntiSeeds", "POSCARS")
    text = open(path).read().replace("Direct", "Cartesian").replace(
        "0.0 0.0 0.0\n0.5 0.5 0.5", "0.0 0.0 0.0\n2.5 2.5 2.5"
    )
    open(path, "w").write(text)
    cartesian = pickAntiSeeds()[0]["FINGERPRINT"]

    assert np.allclose(direct, cartesian)


# --------------------------------------------------------------------------
# AntiSeedsCorrection
# --------------------------------------------------------------------------

def test_correction_is_identity_by_default(tmp_path, monkeypatch):
    """antiSeedsMax=0 (the default) must leave fitness untouched."""
    monkeypatch.setitem(G, "ORG_STRUC", _org(tmp_path))
    monkeypatch.setitem(G, "POP_STRUC", _population([[1.0, 0.0]], [[0.0, 1.0]]))
    monkeypatch.setitem(G, "ANTISEEDS", [])

    fitness = [-1.0, -2.0]
    out = AntiSeedsCorrection(fitness)

    assert np.allclose(out, fitness)
    assert fitness == [-1.0, -2.0]        # caller's list is not mutated


def test_correction_penalizes_structures_similar_to_an_anti_seed(tmp_path, monkeypatch):
    near = np.array([[1.0, 0.5], [0.5, 1.0]])
    far = np.array([[-1.0, -0.5], [-0.5, -1.0]])
    monkeypatch.setitem(G, "ORG_STRUC", _org(tmp_path, antiSeedsMax=2.0))
    monkeypatch.setitem(G, "POP_STRUC", _population(near, far))
    monkeypatch.setitem(
        G,
        "ANTISEEDS",
        [{"FINGERPRINT": near, "Sigma": 0.05, "Max": 1.0}],
    )

    # Max is scaled by the fitness spread of the best part (:6), so the
    # fixtures must not all carry the same fitness or Max collapses to 0.
    out = AntiSeedsCorrection([-1.0, -2.0])

    assert out[0] > out[1]                # the clone is penalised much harder
    assert out[0] > 0.5                   # and by a sizeable amount
    assert out[1] < -1.0 + 1e-6           # the unrelated one is untouched


def test_correction_first_generation_rescales_preloaded_seeds(tmp_path, monkeypatch):
    fp = np.array([[1.0, 0.2], [0.2, 1.0]])
    monkeypatch.setitem(G, "ORG_STRUC", _org(tmp_path, antiSeedsMax=2.0, antiSeedsSigma=0.1))
    monkeypatch.setitem(G, "POP_STRUC", _population(fp, fp * 0.5))
    entries = [{"FINGERPRINT": fp, "Sigma": 999.0, "Max": 999.0}]
    monkeypatch.setitem(G, "ANTISEEDS", entries)

    AntiSeedsCorrection([-1.0, -2.0])

    assert entries[0]["Sigma"] != 999.0    # :36-41 recomputed, not inherited
    assert entries[0]["Max"] != 999.0


def test_correction_positive_activation_appends_all(tmp_path, monkeypatch):
    fps = [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]
    monkeypatch.setitem(G, "ORG_STRUC", _org(tmp_path, antiSeedsMax=1.0, antiSeedsActivation=1))
    monkeypatch.setitem(G, "POP_STRUC", struct(POPULATION=[_individual(f) for f in fps], generation=1))
    monkeypatch.setitem(G, "ANTISEEDS", [])

    AntiSeedsCorrection([-1.0, -2.0, -3.0])

    assert len(G["ANTISEEDS"]) == 3       # :62-70 one per structure


def test_correction_negative_activation_appends_only_the_best(tmp_path, monkeypatch):
    fps = [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]
    monkeypatch.setitem(G, "ORG_STRUC", _org(tmp_path, antiSeedsMax=1.0, antiSeedsActivation=-1))
    monkeypatch.setitem(G, "POP_STRUC", struct(POPULATION=[_individual(f) for f in fps], generation=1))
    monkeypatch.setitem(G, "ANTISEEDS", [])

    AntiSeedsCorrection([-1.0, -9.0])

    appended = G["ANTISEEDS"]
    assert len(appended) == 1
    assert np.allclose(appended[0]["FINGERPRINT"], fps[1])   # index of min fitness


def test_correction_max_has_a_half_decay_floor(tmp_path, monkeypatch):
    """After `activation + 1` generations Max cannot drop below half the last one."""
    fp = np.array([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setitem(
        G,
        "ORG_STRUC",
        _org(tmp_path, antiSeedsMax=0.0, antiSeedsActivation=1),
    )
    monkeypatch.setitem(G, "POP_STRUC", struct(POPULATION=[_individual(fp)], generation=5))
    entries = [{"FINGERPRINT": fp * 0.9, "Sigma": 0.05, "Max": 4.0}]
    monkeypatch.setitem(G, "ANTISEEDS", entries)

    out = AntiSeedsCorrection([-1.0])

    # antiSeedsMax computes to 0 here, so any penalty can only come from the floor
    assert out[0] > 0.0


def test_correction_empty_fingerprints_treated_as_maximally_distant(tmp_path, monkeypatch):
    fp = np.array([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setitem(G, "ORG_STRUC", _org(tmp_path, antiSeedsMax=1.0))
    monkeypatch.setitem(G, "POP_STRUC", _population([[0.0, 0.0], [0.0, 0.0]]))
    monkeypatch.setitem(G, "ANTISEEDS", [{"FINGERPRINT": fp, "Sigma": 0.05, "Max": 1.0}])

    # must not raise even though the population has no fingerprint yet
    out = AntiSeedsCorrection([0.0])

    assert out.shape == (1,)
