import contextlib
import importlib
import io

import numpy as np

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.sys.mol_bond_table import intramol_bonds, radii_for

_rot = importlib.import_module("gpevo.FunctionFolder.USPEX.310.Rotation_310")
_twist_dihedrals = _rot._twist_dihedrals
_bond_graph = _rot._bond_graph


def _setup_atom_types():
    # intramol_bonds needs ORG_STRUC.atomType to resolve covalent radii.
    G["ORG_STRUC"] = AttrDict(atomType=[6])  # all carbons


def test_bond_graph_matches_intramol_bonds():
    _setup_atom_types()
    coords = np.array([[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 0, 0]], dtype=float)
    types = [1, 1, 1, 1]
    adj = _bond_graph(coords, radii_for(types))
    bonds = set(tuple(sorted(p)) for p in intramol_bonds(coords, types))
    edges = set()
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            edges.add(tuple(sorted((i, j))))
    assert edges == bonds


def test_twist_no_flex_returns_identical():
    _setup_atom_types()
    coords = np.array([[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 0, 0]], dtype=float)
    out, moved = _twist_dihedrals(coords, [], [1, 1, 1, 1], None)
    assert moved is False
    assert np.allclose(out, coords)


def test_twist_rotates_subtree_keeps_connectivity():
    _setup_atom_types()
    # Bent C-C-C-C chain (atom 3 off the 1-2 rotation axis); flex around
    # bond (1,2), outer atom 3 rotates out of plane.
    coords = np.array([[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 1.0, 0]], dtype=float)
    types = [1, 1, 1, 1]
    ref_bonds = intramol_bonds(coords, types)
    flex = [(1, 2, 3)]
    # Try several seeds; a small-angle twist should be accepted.
    accepted = None
    for _ in range(50):
        out, moved = _twist_dihedrals(coords, flex, types, ref_bonds, max_iter=5)
        if moved and not np.allclose(out, coords):
            accepted = out
            break
    assert accepted is not None, "expected at least one accepted dihedral twist"
    # Connectivity preserved.
    assert set(tuple(sorted(p)) for p in intramol_bonds(accepted, types)) == \
        set(tuple(sorted(p)) for p in ref_bonds)
    # Atom 3 (the only off-axis one) actually moved; atoms 0,1 stay put.
    assert not np.allclose(accepted[3], coords[3])
    assert np.allclose(accepted[0], coords[0])
    assert np.allclose(accepted[1], coords[1])
    # Atom 2 lies on the rotation axis -> unchanged.
    assert np.allclose(accepted[2], coords[2])


# ---------------------------------------------------------------------------
#  Regression: STDMOL / molRefBonds are keyed by molecule TYPE, not position.
#  A cell with MtypeLIST=[1,1,2,2] has the flex-declaring type (2) at
#  positions 2,3 -- beyond len(STDMOL)=2.  Indexing by position skipped them
#  entirely and made the group-rotation ratio permanently zero (ct22/results3).
# ---------------------------------------------------------------------------

def _two_type_offspring():
    """2-atom type-1 molecules (no flex) + 4-atom type-2 molecules (flex)."""
    t1 = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    t2 = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                   [3.0, 0.0, 0.0], [4.5, 1.0, 0.0]])

    def mol(c):
        return AttrDict(MOLCOORS=c.copy(), MOLCENTER=c.mean(0),
                        ZMATRIX=c.copy())

    child = AttrDict()
    child.MOLECULES = [mol(t1), mol(t1), mol(t2), mol(t2)]
    child.MtypeLIST = [1, 1, 2, 2]
    child.LATTICE = np.diag([60.0, 60.0, 60.0])
    return child, t1, t2


def test_group_rotation_reaches_molecules_past_the_template_count(monkeypatch,
                                                                  capsys):
    """Type-2 copies at positions 2,3 must get their flexible-dihedral twist."""
    child, t1, t2 = _two_type_offspring()
    t1_std = AttrDict(types=[1, 1], flex_dihedral=[],
                      format=np.zeros((2, 3), dtype=int))
    t2_std = AttrDict(types=[1, 1, 1, 1], flex_dihedral=[(1, 2, 3)],
                      format=np.zeros((4, 3), dtype=int))
    org = AttrDict(
        atomType=[6], STDMOL=[t1_std, t2_std],
        molRefBonds=[intramol_bonds(t1, [1, 1]),
                     intramol_bonds(t2, [1, 1, 1, 1])],
        rotationAngleMax=np.pi / 2, translationMax=0.0, maxDihedralIter=200,
    )
    monkeypatch.setitem(G, "ORG_STRUC", org)
    monkeypatch.setitem(G, "POP_STRUC", AttrDict(POPULATION=[AttrDict()]))
    monkeypatch.setitem(G, "OFF_STRUC", AttrDict(POPULATION=[]))
    monkeypatch.setattr(_rot, "pick_parent", lambda: 0)
    monkeypatch.setattr(_rot, "copy_individual",
                        lambda ind: _two_type_offspring()[0])
    monkeypatch.setattr(_rot, "valid_molecules", lambda *a, **k: True)
    monkeypatch.setattr(_rot, "recompute_zmatrix", lambda *a, **k: None)
    monkeypatch.setattr(_rot, "finalize_offspring", lambda *a, **k: None)
    monkeypatch.setattr(_rot, "_group_rot_gen", 0)
    monkeypatch.setattr(_rot, "_group_rot_total", 0)

    _rot.Rotation_310(1)

    assert _rot._group_rot_gen >= 1, (
        "the two type-2 (flex) molecules at positions 2,3 were never twisted")
    assert "[group rotation]" in capsys.readouterr().out


def test_type_lookup_does_not_apply_wrong_template(monkeypatch):
    """A type-1 molecule must NOT receive the type-2 template's flex.

    Before the fix, mi=1 (type 1) was indexed as std_mols[1] (the flex
    template).  Give type 1 a 2-atom geometry: the flex tuple (1,2,3) is
    out of range, so a wrong lookup yields no twist -- but with the correct
    lookup, tidx=0 has no flex at all.  Both give no twist for those two, so
    assert the successful offspring carries exactly the two type-2 twists.
    """
    child, t1, t2 = _two_type_offspring()
    t1_std = AttrDict(types=[1, 1], flex_dihedral=[],
                      format=np.zeros((2, 3), dtype=int))
    t2_std = AttrDict(types=[1, 1, 1, 1], flex_dihedral=[(1, 2, 3)],
                      format=np.zeros((4, 3), dtype=int))
    org = AttrDict(
        atomType=[6], STDMOL=[t1_std, t2_std],
        molRefBonds=[intramol_bonds(t1, [1, 1]),
                     intramol_bonds(t2, [1, 1, 1, 1])],
        rotationAngleMax=np.pi / 2, translationMax=0.0, maxDihedralIter=200,
    )
    produced = []

    def capture(child, how_come):
        produced.append(child)

    monkeypatch.setitem(G, "ORG_STRUC", org)
    monkeypatch.setitem(G, "POP_STRUC", AttrDict(POPULATION=[AttrDict()]))
    monkeypatch.setitem(G, "OFF_STRUC", AttrDict(POPULATION=[]))
    monkeypatch.setattr(_rot, "pick_parent", lambda: 0)
    monkeypatch.setattr(_rot, "copy_individual",
                        lambda ind: _two_type_offspring()[0])
    monkeypatch.setattr(_rot, "valid_molecules", lambda *a, **k: True)
    monkeypatch.setattr(_rot, "recompute_zmatrix", lambda *a, **k: None)
    monkeypatch.setattr(_rot, "finalize_offspring", capture)
    monkeypatch.setattr(_rot, "_group_rot_gen", 0)
    monkeypatch.setattr(_rot, "_group_rot_total", 0)

    _rot.Rotation_310(1)

    assert produced, "no offspring was produced"
    off = produced[0]
    # The two type-1 molecules (positions 0,1) must be untouched by flex: they
    # only have 2 atoms, so their coordinates can never gain a twist.
    assert off.MOLECULES[0].MOLCOORS.shape == (2, 3)
    assert off.MOLECULES[1].MOLCOORS.shape == (2, 3)
    # group_rotated came from the two type-2 molecules, not from a wrong
    # template applied to a type-1 one.
    assert _rot._group_rot_gen == 1


def _install_twist_offspring(monkeypatch, flex=((1, 2, 3),)):
    """Common setup: a 4-atom type-2 molecule whose twist succeeds."""
    child, t1, t2 = _two_type_offspring()
    t1_std = AttrDict(types=[1, 1], flex_dihedral=[],
                      format=np.zeros((2, 3), dtype=int))
    t2_std = AttrDict(types=[1, 1, 1, 1], flex_dihedral=list(flex),
                      format=np.zeros((4, 3), dtype=int))
    org = AttrDict(
        atomType=[6], STDMOL=[t1_std, t2_std],
        molRefBonds=[intramol_bonds(t1, [1, 1]),
                     intramol_bonds(t2, [1, 1, 1, 1])],
        rotationAngleMax=np.pi / 2, translationMax=0.0, maxDihedralIter=200,
    )
    monkeypatch.setitem(G, "ORG_STRUC", org)
    monkeypatch.setitem(G, "POP_STRUC", AttrDict(POPULATION=[AttrDict()]))
    monkeypatch.setitem(G, "OFF_STRUC", AttrDict(POPULATION=[]))
    monkeypatch.setattr(_rot, "pick_parent", lambda: 0)
    monkeypatch.setattr(_rot, "copy_individual",
                        lambda ind: _two_type_offspring()[0])
    monkeypatch.setattr(_rot, "recompute_zmatrix", lambda *a, **k: None)
    monkeypatch.setattr(_rot, "finalize_offspring", lambda *a, **k: None)
    monkeypatch.setattr(_rot, "_group_rot_gen", 0)
    monkeypatch.setattr(_rot, "_group_rot_total", 0)
    return org


def test_rigid_rotation_fallback_when_twisted_cell_is_invalid(monkeypatch, capsys):
    """A twisted group that collides falls back to rigid rotation, not Random.

    This is the operator-level fallback: valid_molecules is called twice (once
    on the twisted child, once after undoing the twists) and the offspring is
    still a rotmutation, with no [group rotation] tag.
    """
    _install_twist_offspring(monkeypatch)
    calls = {"n": 0, "coords": []}

    def fake_valid(ind, adj):
        calls["n"] += 1
        # snapshot a type-2 molecule's geometry so we can see the fallback
        calls["coords"].append(np.asarray(ind.MOLECULES[2].MOLCOORS).copy())
        return 0 if calls["n"] == 1 else 1   # clash, then rigid-only ok

    monkeypatch.setattr(_rot, "valid_molecules", fake_valid)
    _rot.Rotation_310(1)

    assert calls["n"] == 2, "expected one failed check + one after fallback"
    # The fallback restored the rigid-rotation geometry (coords changed).
    assert not np.allclose(calls["coords"][0], calls["coords"][1])
    assert _rot._group_rot_gen == 0, "fallback offspring is not group-rotated"
    assert "[group rotation]" not in capsys.readouterr().out
    assert len(G["OFF_STRUC"].POPULATION) == 1, "no Random replacement"


def test_no_fallback_when_twisted_cell_is_valid(monkeypatch, capsys):
    """When the twisted child passes, no fallback and the tag is kept."""
    _install_twist_offspring(monkeypatch)
    calls = {"n": 0}

    def fake_valid(ind, adj):
        calls["n"] += 1
        return 1

    monkeypatch.setattr(_rot, "valid_molecules", fake_valid)
    _rot.Rotation_310(1)

    assert calls["n"] == 1, "no second check when the first passes"
    assert _rot._group_rot_gen == 1
    assert "[group rotation]" in capsys.readouterr().out


def test_fallback_still_fails_retries_then_uses_random(monkeypatch):
    """If rigid-only is also invalid, the attempt is retried (not accepted)."""
    _install_twist_offspring(monkeypatch)
    monkeypatch.setattr(_rot, "valid_molecules", lambda *a, **k: 0)
    # Stub the Random fallback so the test does not need a CalcFoldTemp dir.
    rnd = {"n": 0}
    monkeypatch.setattr(
        __import__("importlib").import_module(
            "gpevo.FunctionFolder.USPEX.310.Random_310"),
        "Random_310", lambda count: rnd.__setitem__("n", rnd["n"] + 1))
    with contextlib.redirect_stdout(io.StringIO()):
        _rot.Rotation_310(1)
    assert rnd["n"] == 1
    assert len(G["OFF_STRUC"].POPULATION) == 0
