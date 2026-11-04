"""Python equivalent of newMolCheck.m (molecular intermolecular distance checks).

The pairwise distance scan is compiled in ``molcheck_cy.pyx``.  The original
pure-numpy routine is kept below as a reference oracle (and a fallback for
source checkouts where the extension has not been built); the parity tests in
tests/test_molcheck_cy.py assert the two agree on random structures.
"""
from __future__ import annotations

import itertools

import numpy as np

from gpevo.runtime import G

try:
    from gpevo.FunctionFolder.sys.molcheck_cy import molcheck as _cython_molcheck
except Exception:                       # extension not built / incompatible
    _cython_molcheck = None


def _prepare(molecules, mtype_list, min_matrix):
    """Build per-molecule coordinate/type lists; None means an invalid template."""
    ORG = G["ORG_STRUC"]
    stdmol = ORG.STDMOL
    mol_coords, mol_types = [], []
    for i, mol in enumerate(molecules):
        std = stdmol[int(mtype_list[i]) - 1]
        classes = [int(t) for t in std.types]
        # std.types holds 1-based positions into ORG.atomType (read_molecules.py
        # records an absent element as 0).  Indexing min_matrix with it would
        # silently wrap to the last row/column, so reject the structure instead.
        if any(t <= 0 or t > len(ORG.atomType) for t in classes):
            return None
        mol_coords.append(np.ascontiguousarray(mol.MOLCOORS, dtype=np.float64))
        mol_types.append(classes)
    return (mol_coords, mol_types,
            np.ascontiguousarray(min_matrix, dtype=np.float64))


def _cython_call(mol_coords, mol_types, lat, inv_lat, min_matrix):
    """Flatten the molecules into contiguous buffers for the Cython kernel."""
    starts = [0]
    coords_parts, frac_parts, types_parts = [], [], []
    for coords, classes in zip(mol_coords, mol_types):
        coords_parts.append(coords)
        frac_parts.append(coords @ inv_lat)
        types_parts.append(np.asarray(classes, dtype=np.int32) - 1)
        starts.append(starts[-1] + len(coords))
    if not coords_parts:
        return 1
    offsets = np.asarray(
        [p for p in itertools.product(range(-2, 3), repeat=3)
         if p != (0, 0, 0)], dtype=np.float64) @ lat
    return int(_cython_molcheck(
        np.ascontiguousarray(np.vstack(frac_parts), dtype=np.float64),
        np.ascontiguousarray(np.vstack(coords_parts), dtype=np.float64),
        np.asarray(starts, dtype=np.int32),
        np.ascontiguousarray(np.concatenate(types_parts), dtype=np.int32),
        np.ascontiguousarray(lat, dtype=np.float64),
        min_matrix,
        np.ascontiguousarray(offsets, dtype=np.float64),
    ))


def _python_molcheck(mol_coords, mol_types, lat, inv_lat, min_matrix):
    """Original pure-numpy implementation, kept as the reference oracle."""
    n_mols = len(mol_types)
    # Intermolecular checks
    for i in range(n_mols - 1):
        ci = mol_coords[i]
        ti = mol_types[i]
        if len(ci) == 0:
            continue
        frac_i = ci @ inv_lat
        for k in range(i + 1, n_mols):
            ck = mol_coords[k]
            tk = mol_types[k]
            if len(ck) == 0:
                continue
            frac_k = ck @ inv_lat
            vectors = []
            for a in range(len(ci)):
                for b in range(len(ck)):
                    dfrac = (frac_i[a] - frac_k[b]) - np.round(frac_i[a] - frac_k[b])
                    vectors.append(dfrac @ lat)
            if not vectors:
                continue
            vect = np.asarray(vectors)
            dists = np.linalg.norm(vect, axis=1)
            flatthr = np.array([min_matrix[ti[a] - 1, tk[b] - 1]
                                for a in range(len(ci)) for b in range(len(ck))])
            if np.any(dists <= flatthr):
                return 0

    # Intramolecular periodic-image checks (same original style, non-matrix loop)
    for ind in range(n_mols):
        coords = mol_coords[ind]
        types = mol_types[ind]
        n_atoms = len(coords)
        if n_atoms < 2:
            continue
        offsets = np.array([p for p in itertools.product(range(-2, 3), repeat=3)
                            if p != (0, 0, 0)]) @ lat
        for n in range(1, n_atoms):
            images = coords[n] + offsets
            thresh = np.array([min_matrix[types[m] - 1, types[n] - 1]
                               for m in range(n)])
            for m in range(n):
                dist = np.min(np.linalg.norm(images - coords[m], axis=1))
                if dist <= thresh[m]:
                    return 0
    return 1


def newMolCheck(molecules, lattice, mtype_list, min_matrix):
    lat = np.asarray(lattice, dtype=float)
    inv_lat = np.linalg.inv(lat)
    prepared = _prepare(molecules, mtype_list, min_matrix)
    if prepared is None:
        return 0
    mol_coords, mol_types, min_matrix = prepared
    if _cython_molcheck is not None:
        return _cython_call(mol_coords, mol_types, lat, inv_lat, min_matrix)
    return _python_molcheck(mol_coords, mol_types, lat, inv_lat, min_matrix)
