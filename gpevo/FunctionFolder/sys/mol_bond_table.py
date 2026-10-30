"""Intramolecular bond tables for molecule-damage detection.

At the start of a calculation we build a *reference* bond table for every
molecule type from its standard (template) geometry.  During the run each
relaxed individual is checked: if a molecule carries *more* intramolecular
bonds than the reference -- i.e. an atom pair that is not bonded in the
template has come within bonding distance inside the molecule -- the molecule
is flagged as damaged.  This catches an unwanted bond having formed (e.g. a
dimerisation or a ring closure that the intended chemistry does not allow),
which the existing geometry / intermolecular-distance checks can miss.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G
from gpevo.FunctionFolder.sys.covalentRadius import covalentRadius

# Cordero-style single-bond criterion: two atoms are bonded if their distance
# is below the sum of their covalent radii plus this tolerance (Angstrom).
BOND_TOL = 0.45


def _rcov(type_idx):
    """Covalent radius (Angstrom) for a 1-based ORG.atomType index."""
    ORG = G["ORG_STRUC"]
    try:
        z = ORG.atomType[type_idx - 1]
    except (IndexError, TypeError, AttributeError):
        return 1.20
    r = covalentRadius(z)
    try:
        return float(r)
    except (TypeError, ValueError):
        return 1.20


def intramol_bonds(coords, type_idxs):
    """Return the set of ``(i, j)`` (i < j) bonded atom pairs within one molecule.

    ``coords`` are Cartesian coordinates of the molecule's atoms (Angstrom);
    ``type_idxs`` are 1-based indices into ``ORG.atomType`` for each atom.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    bonds = set()
    if n < 2:
        return bonds
    for i in range(n):
        ri = _rcov(type_idxs[i])
        ci = coords[i]
        for j in range(i + 1, n):
            d = float(np.linalg.norm(ci - coords[j]))
            if d < ri + _rcov(type_idxs[j]) + BOND_TOL:
                bonds.add((i, j))
    return bonds


def build_reference_bond_tables():
    """Build and store ``ORG.molRefBonds`` from the standard molecules (STDMOL).

    Called automatically at the start of a calculation (from ``read_molecules``)
    and lazily from the molecule check if the table is missing (e.g. after a
    restart that discarded it).
    """
    ORG = G["ORG_STRUC"]
    std = getattr(ORG, "STDMOL", None)
    if not std:
        ORG.molRefBonds = []
        return
    ref = []
    for m in std:
        coords = np.asarray(getattr(m, "molecule", np.zeros((0, 3))), dtype=float)
        types = [int(t) for t in getattr(m, "types", [])]
        ref.append(intramol_bonds(coords, types))
    ORG.molRefBonds = ref
