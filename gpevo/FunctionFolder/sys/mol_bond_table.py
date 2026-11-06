"""Intramolecular bond tables for molecule-damage detection.

At the start of a calculation we build a *reference* bond table for every
molecule type from its standard (template) geometry.  During the run each
relaxed individual is checked: if a molecule carries *more* intramolecular
bonds than the reference -- i.e. an atom pair that is not bonded in the
template has come within bonding distance inside the molecule -- the molecule
is flagged as damaged.  This catches an unwanted bond having formed (e.g. a
dimerisation or a ring closure that the intended chemistry does not allow),
which the existing geometry / intermolecular-distance checks can miss.

The pairwise scan is compiled in ``bondtable_cy.pyx``.  A vectorised numpy
implementation is kept as the fallback for source checkouts where the
extension has not been built; ``tests/test_bondtable_cy.py`` asserts the two
agree bit-for-bit with the original double-loop reference.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G
from gpevo.FunctionFolder.sys.covalentRadius import covalentRadius

try:
    from gpevo.FunctionFolder.sys.bondtable_cy import bonds as _cython_bonds
except Exception:                       # extension not built / incompatible
    _cython_bonds = None


# Single-bond criterion matching USPEX (find_pair.m): two atoms are bonded if
# their distance is below 1.2 * (sum of their covalent radii).  The multiplicative
# 1.2 factor scales with atom size, so light atoms (e.g. H-H) are not spuriously
# bonded the way a fixed additive tolerance would.
BOND_SCALE = 1.2


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


def radii_for(type_idxs):
    """Resolve covalent radii once for a molecule's atom-type list.

    ``_rcov`` does a ``G``/dict lookup per atom; callers that scan many
    candidate geometries for the SAME molecule (e.g. ``_twist_dihedrals``)
    should call this once and pass the result to ``bonds_from_radii``.
    """
    return np.array([_rcov(t) for t in type_idxs], dtype=np.float64)


def _vectorized_bonds(coords, radii):
    """Pure-numpy fallback: pairwise distances via broadcasting."""
    n = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    iu = np.triu_indices(n, k=1)
    hit = (d < BOND_SCALE * (radii[:, None] + radii[None, :]))[iu]
    return set(zip(iu[0][hit].tolist(), iu[1][hit].tolist()))


def _reference_bonds(coords, type_idxs):
    """Original double-loop implementation, kept as the parity oracle."""
    n = coords.shape[0]
    bonds = set()
    for i in range(n - 1):
        ri = _rcov(type_idxs[i])
        ci = coords[i]
        for j in range(i + 1, n):
            d = float(np.linalg.norm(ci - coords[j]))
            if d < BOND_SCALE * (ri + _rcov(type_idxs[j])):
                bonds.add((i, j))
    return bonds


def bonds_from_radii(coords, radii):
    """Return the set of bonded ``(i, j)`` pairs using precomputed radii.

    ``radii`` comes from ``radii_for(type_idxs)``.  Uses the Cython kernel when
    available, otherwise the vectorised numpy fallback.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if n < 2:
        return set()
    r = np.ascontiguousarray(radii, dtype=np.float64)
    if _cython_bonds is not None:
        return set(_cython_bonds(
            np.ascontiguousarray(coords, dtype=np.float64), r,
            BOND_SCALE, n))
    return _vectorized_bonds(coords, r)


def intramol_bonds(coords, type_idxs):
    """Return the set of ``(i, j)`` (i < j) bonded atom pairs within one molecule.

    ``coords`` are Cartesian coordinates of the molecule's atoms (Angstrom);
    ``type_idxs`` are 1-based indices into ``ORG.atomType`` for each atom.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if n < 2:
        return set()
    return bonds_from_radii(coords, radii_for(type_idxs[:n]))


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
