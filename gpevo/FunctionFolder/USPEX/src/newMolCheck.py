"""Python equivalent of newMolCheck.m (molecular intermolecular distance checks)."""
from __future__ import annotations

import itertools
import numpy as np

from gpevo.runtime import G


def newMolCheck(molecules, lattice, mtype_list, min_matrix):
    ORG = G["ORG_STRUC"]
    stdmol = ORG.STDMOL
    lat = np.asarray(lattice, dtype=float)
    inv_lat = np.linalg.inv(lat)
    n_mols = len(mtype_list)
    min_matrix = np.asarray(min_matrix, dtype=float)
    if min_matrix.shape != (len(ORG.atomType), len(ORG.atomType)):
        if len(ORG.atomType) < min_matrix.shape[0]:
            pass  # callers may pass larger matrices; index by type works
    mol_coords = []
    mol_types = []
    for i, mol in enumerate(molecules):
        mt = mtype_list[i]
        std = stdmol[mt - 1]
        mol_coords.append(np.asarray(mol.MOLCOORS, dtype=float))
        classes = [int(t) for t in std.types]
        # ``std.types`` holds 1-based positions into ORG.atomType (see
        # read_molecules.py:116 / GetPOP_MOL.py:19).  A value out of that
        # range means the molecule was built from an element absent from
        # atomType (read_molecules.py records it as 0): the code can not look
        # up its distance threshold, and indexing min_matrix with it would
        # silently wrap to the last row/column.  Reject the structure instead.
        if any(t <= 0 or t > len(ORG.atomType) for t in classes):
            return 0
        mol_types.append(classes)

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
            # thresh lookup
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
        offsets = np.array([p for p in itertools.product(range(-2, 3), repeat=3) if p != (0, 0, 0)]) @ lat
        for n in range(1, n_atoms):
            images = coords[n] + offsets
            thresh = np.array([min_matrix[types[m] - 1, types[n] - 1] for m in range(n)])
            for m in range(n):
                dist = np.min(np.linalg.norm(images - coords[m], axis=1))
                if dist <= thresh[m]:
                    return 0
    return 1
