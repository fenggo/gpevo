import importlib
import numpy as np

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.sys.mol_bond_table import intramol_bonds

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
    adj = _bond_graph(coords, types)
    bonds = set(tuple(sorted(p)) for p in intramol_bonds(coords, types))
    edges = set()
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            edges.add(tuple(sorted((i, j))))
    assert edges == bonds


def test_twist_no_flex_returns_identical():
    _setup_atom_types()
    coords = np.array([[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 0, 0]], dtype=float)
    out = _twist_dihedrals(coords, [], [1, 1, 1, 1], None)
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
        out = _twist_dihedrals(coords, flex, types, ref_bonds, max_iter=5)
        if not np.allclose(out, coords):
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
