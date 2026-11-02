"""Soft-mode machinery for the 310 molecular variation operators.

Pure-Python/numpy port of the MATLAB/Octave code in the ``uspex0925`` reference
tree:

* ``USPEX/src/calcSoftModes_molecules.m`` and its Cython translation
  ``sys/uspex_softmode_core.pyx`` -> :func:`calc_soft_modes`
* ``USPEX/src/move_along_SoftMode_molMutation.m`` -> :func:`move_along_soft_mode`
* ``USPEX/src/move_all_mol_Mutation.m`` -> :func:`move_all_mol_mutation`
* ``sys/valence.m``, ``sys/valenceElectronsNumber.m``,
  ``sys/calcDefaultGoodBonds.m`` -> the tables and ``default_good_bonds``

``calc_soft_modes`` returns ``(freq, eigvector)`` where ``freq`` is the 1-D
vector ``abs(real(eig(D)))`` (i.e. MATLAB's ``diag(freq)``) and ``eigvector`` is
the real (3N x 3N) matrix whose columns are the eigenvectors.  Bond search and
the colour union-find are vectorised / done with a disjoint-set forest, but the
sequence of operations is the same as in the original sources.
"""
from __future__ import annotations

import itertools

import numpy as np

# valence.m -- 1-based by atomic number (Z=1..110); beyond the table -> 3.0.
_VALENCE = [
    1.0, 0.5, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.5, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.5, 1.0, 2.0,
    3.0, 4.0, 4.0, 3.0, 4.0, 3.0, 3.0, 2.0, 2.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.5, 1.0, 2.0, 3.0, 4.0,
    5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.5, 1.0, 2.0, 3.0, 4.0, 3.0, 3.0,
    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 2.0,
    3.0, 4.0, 3.0, 2.0, 1.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
    4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
]

# valenceElectronsNumber.m -- 1-based by Z (Z=1..105); beyond -> 3.
_VALENCE_ELECTRONS = [
    1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2,
    3, 4, 5, 6, 5, 3, 3, 3, 2, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4,
    5, 6, 5, 3, 3, 3, 2, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 4, 5, 6, 7, 8, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3,
]

# atomicGoodBonds.m -- 1-based by Z (Z=1..118); beyond -> 0.10.
_ATOMIC_GOOD_BONDS = [
    0.20, 0.05, 0.10, 0.20, 0.30, 0.50, 0.50, 0.30, 0.10, 0.05, 0.05, 0.10, 0.20, 0.30, 0.30, 0.20, 0.10, 0.05, 0.05, 0.10,
    0.20, 0.30, 0.30, 0.25, 0.30, 0.25, 0.25, 0.15, 0.10, 0.10, 0.25, 0.50, 0.35, 0.20, 0.10, 0.05, 0.05, 0.10, 0.20, 0.30,
    0.35, 0.30, 0.30, 0.30, 0.30, 0.30, 0.05, 0.10, 0.20, 0.30, 0.20, 0.20, 0.10, 0.05, 0.05, 0.10, 0.20, 0.30, 0.20, 0.20,
    0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.30, 0.40, 0.30, 0.30, 0.30,
    0.30, 0.30, 0.05, 0.10, 0.20, 0.30, 0.20, 0.20, 0.10, 0.05, 0.05, 0.10, 0.20, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30,
    0.30, 0.30, 0.30, 0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
]


def valence(atomic_number):
    """valence.m -- formal valence used by the soft-mode force constants."""
    z = int(atomic_number)
    return _VALENCE[z - 1] if 1 <= z <= len(_VALENCE) else 3.0


def valence_electrons(atomic_number):
    """valenceElectronsNumber.m -- used for the electronegativity scale."""
    z = int(atomic_number)
    return float(_VALENCE_ELECTRONS[z - 1]) if 1 <= z <= len(_VALENCE_ELECTRONS) else 3.0


def default_good_bonds(atom_type):
    """calcDefaultGoodBonds.m -- sqrt(gB_i * gB_j), symmetric."""
    z = [int(v) for v in atom_type]
    g = np.array([_ATOMIC_GOOD_BONDS[v - 1] if 1 <= v <= len(_ATOMIC_GOOD_BONDS) else 0.10
                  for v in z], dtype=float)
    return np.sqrt(np.outer(g, g))


class _DSU:
    """Disjoint-set forest over the (5,5,5,N) colour labels.

    The MATLAB code keeps a ``colors1`` array and implements a union by
    ``colors1[colors1 == cj] = ci``.  That is exactly a disjoint-set union; a
    forest keeps the same equivalence classes without scanning the whole array.
    """

    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int64)

    def find(self, x):
        p = self.p
        r = int(x)
        while p[r] != r:
            r = int(p[r])
        while p[x] != r:
            p[x], x = r, int(p[x])
        return r

    def find_many(self, arr):
        arr = np.asarray(arr, dtype=np.int64)
        r = self.p[arr].copy()
        while True:
            r2 = self.p[r]
            if np.array_equal(r2, r):
                break
            r = r2
        self.p[arr] = r
        return r

    def union_pairs(self, cj, ci):
        for a, b in zip(np.asarray(cj, dtype=np.int64).ravel(),
                        np.asarray(ci, dtype=np.int64).ravel()):
            ra, rb = self.find(a), self.find(b)
            if ra != rb:
                self.p[ra] = rb

    def union_many(self, labels):
        roots = self.find_many(labels)
        base = int(roots[0])
        for r in np.unique(roots):
            if int(r) != base:
                self.p[int(r)] = base

    def snapshot(self):
        return self.p.copy()


def _matlab_round(x):
    """MATLAB ``round`` (ties away from zero), unlike np.round (ties to even)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _find_many(parent, arr):
    arr = np.asarray(arr, dtype=np.int64)
    r = parent[arr].copy()
    while True:
        r2 = parent[r]
        if np.array_equal(r2, r):
            break
        r = r2
    return r


def calc_soft_modes(coords, lat, at_types, R_val, N_val, val_arr, goodBonds):
    """Port of ``calcSoftModes_molecules.m`` / ``uspex_softmode_core.pyx``.

    Parameters mirror ``fast_calcSoftModes.m``: ``coords`` are fractional
    coordinates in species-blocked order (``POPULATION(i).COORDINATES``),
    ``lat`` is the 3x3 lattice, ``at_types`` is the 0-based species index per
    atom, ``R_val``/``N_val``/``val_arr`` are per-species covalent radii /
    valence-electron counts / valences, and ``goodBonds`` is the (n,n) matrix.

    Returns ``(freq, eigvector)`` with ``freq`` = ``abs(real(eigvals(D)))`` and
    ``eigvector`` = real eigenvectors (columns), matching MATLAB's ``eig(D)``.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    lat = np.asarray(lat, dtype=float).reshape(3, 3)
    at_types = np.asarray(at_types, dtype=int).ravel()
    R_val = np.asarray(R_val, dtype=float).ravel()
    N_val = np.asarray(N_val, dtype=float).ravel()
    val_arr = np.asarray(val_arr, dtype=float).ravel()
    goodBonds = np.asarray(goodBonds, dtype=float)
    n_types = R_val.size
    if goodBonds.shape != (n_types, n_types):
        goodBonds = np.ones((n_types, n_types))

    N = coords.shape[0]
    n3 = 3 * N
    same_bond, max_bond, nL, nL1 = 0.05, 5.0, 2, 5
    sb = -0.37 * np.log(np.maximum(goodBonds, 1e-12))

    # -- 1. Bond search (calcSoftModes_molecules.m:35-89) -------------------
    # MATLAB loops i=1:N, j=i:N, k1..k3=-1:1 and insertion-sorts by delta;
    # np.argsort(..., kind='stable') gives the same order.
    i_idx, j_idx = np.triu_indices(N)
    shifts = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)
    I = np.repeat(i_idx, shifts.shape[0])
    J = np.repeat(j_idx, shifts.shape[0])
    K = np.tile(shifts, (i_idx.size, 1))
    keep = ~((I == J) & np.all(K == 0, axis=1))
    I, J, K = I[keep], J[keep], K[keep]
    vect = coords[I] + K - coords[J]
    dR = vect @ lat
    delta = np.linalg.norm(dR, axis=1) - R_val[at_types[I]] - R_val[at_types[J]]
    keep = delta < max_bond
    I, J, K, delta = I[keep], J[keep], K[keep], delta[keep]
    M = delta.size
    if M == 0:
        return np.zeros(n3), np.eye(n3)
    order = np.argsort(delta, kind="stable")
    I, J, K, delta = I[order], J[order], K[order], delta[order]
    bonds = np.zeros((M, 7), dtype=float)
    bonds[:, 0] = I + 1           # 1-based atom i
    bonds[:, 1] = J + 1           # 1-based atom j
    bonds[:, 2] = delta
    bonds[:, 4:7] = K

    # -- 2. Bond-type classification (:90-112) ------------------------------
    # Bonds within ``same_bond`` in delta and with the same (type_i, type_j)
    # pair form one bond type; bonds_sorted is built in that order.
    bt_of = np.zeros(M, dtype=np.int64)
    sorted_rows = []
    bond_types = 0
    for bb in range(M):
        if bt_of[bb] != 0:
            continue
        bond_types += 1
        bt_of[bb] = bond_types
        sorted_rows.append(bb)
        for bbb in range(bb + 1, M):
            if bonds[bbb, 2] > bonds[bb, 2] + same_bond:
                break
            if (bt_of[bbb] == 0
                    and at_types[int(bonds[bbb, 0]) - 1] == at_types[int(bonds[bb, 0]) - 1]
                    and at_types[int(bonds[bbb, 1]) - 1] == at_types[int(bonds[bb, 1]) - 1]):
                bt_of[bbb] = bond_types
                sorted_rows.append(bbb)
    bs = bonds[sorted_rows].copy()
    bs[:, 3] = bt_of[sorted_rows]
    n_bonds = bs.shape[0]

    # -- 3. Colour union-find / connectivity (:36-222) ----------------------
    # labels[cell, atom] with cell = 25*k1 + 5*k2 + k3, matching
    # colors(k1,k2,k3,i) = i + 25*N*k1 + 5*N*k2 + N*k3 (0-based).
    labels = np.arange(nL1 ** 3 * N, dtype=np.int64).reshape(nL1 ** 3, N)
    center_cell = (nL * nL1 + nL) * nL1 + nL  # (2,2,2) -> 25*2+5*2+2 = 62
    dsu = _DSU(labels.size)
    committed = dsu.snapshot()
    cells = np.arange(nL1 ** 3)
    m1, m2, m3 = cells // 25, (cells // 5) % 5, cells % 5

    def sync_colors():
        # MATLAB: for i<j with the same central colour, union the corresponding
        # periodic-image colours cell-by-cell.  Grouping atoms by their central
        # root gives the same equivalence classes in a single pass.
        for _ in range(3):
            roots = dsu.find_many(labels[center_cell, :])
            changed = False
            for g in np.unique(roots):
                atoms = np.nonzero(roots == g)[0]
                if atoms.size < 2:
                    continue
                for c in range(nL1 ** 3):
                    group = labels[c, atoms]
                    if np.unique(dsu.find_many(group)).size > 1:
                        dsu.union_many(group)
                        changed = True
            if not changed:
                break

    it = -1
    for bt in range(1, bond_types + 1):
        while it + 1 < n_bonds and int(bs[it + 1, 3]) == bt:
            it += 1
            a1, a2 = int(bs[it, 0]), int(bs[it, 1])
            a_type = at_types[a1 - 1]
            b_type = at_types[a2 - 1]
            k1, k2, k3 = (int(bs[it, 4]), int(bs[it, 5]), int(bs[it, 6]))
            bond_used = 0
            # The MATLAB condition inside the 125-cell loop depends only on the
            # bond's own shift, so it is evaluated once (:123-151).
            already = (
                _find_many(committed, [labels[center_cell, a2 - 1]])[0]
                == _find_many(committed, [labels[(nL + k1) * 25 + (nL + k2) * 5 + (nL + k3), a1 - 1]])[0]
                and bs[it, 2] > sb[a_type, b_type]
            )
            if not already:
                bond_used = 1
                shift1, shift2, shift3 = m1 + k1, m2 + k2, m3 + k3
                valid = ((shift1 >= 0) & (shift1 < 5) & (shift2 >= 0) & (shift2 < 5)
                         & (shift3 >= 0) & (shift3 < 5))
                cp = shift1 * 25 + shift2 * 5 + shift3
                cj = labels[cells[valid], a2 - 1]
                ci = labels[cp[valid], a1 - 1]
                dsu.union_pairs(cj, ci)
            if bond_used == 0:
                bs[it, 0] = 0
                bs[it, 1] = 0
            if it + 1 >= n_bonds:
                break
            sync_colors()
            if it + 1 < n_bonds and bs[it + 1, 2] < float(np.max(sb)):
                continue
            committed = dsu.snapshot()
            # connected: every sampled colour is c1 or at most one other value.
            c1 = _find_many(committed, [labels[0, 0]])[0]
            sampled = np.concatenate([
                _find_many(committed, labels[(k1 * 25 + k2 * 5 + k3), :])
                for k1 in (nL - 1, nL, nL + 1)
                for k2 in (nL - 1, nL, nL + 1)
                for k3 in (nL - 1, nL, nL + 1)
            ])
            connected = np.unique(sampled[sampled != c1]).size <= 1
            if connected or it >= n_bonds - 1:
                bs[it + 1:, 0] = 0
                bs[it + 1:, 1] = 0
                break

    # -- 4. nu_factor (:223-232) -------------------------------------------
    good = (bs[:, 0] > 0) & (bs[:, 1] > 0)
    gb = bs[good]
    nu_full = np.zeros(N)
    if gb.size:
        w = np.exp(-gb[:, 2] / 0.37)
        np.add.at(nu_full, gb[:, 0].astype(int) - 1, w)
        np.add.at(nu_full, gb[:, 1].astype(int) - 1, w)
    nu_factor = val_arr[at_types] / np.maximum(nu_full, 1e-12)

    # -- 5. Dynamical matrix D (:233-322) ---------------------------------
    D = np.zeros((n3, n3))
    for row in bs:
        a1, a2 = int(row[0]), int(row[1])
        if a1 == 0 or a2 == 0:
            continue
        a_type, b_type = at_types[a1 - 1], at_types[a2 - 1]
        d = float(row[2])
        R_a = R_val[a_type] + d / 2
        R_b = R_val[b_type] + d / 2
        if R_a < 0.05:
            R_b = R_b - 0.05 + R_a
            R_a = 0.05
        if R_b < 0.05:
            R_a = R_a - 0.05 + R_b
            R_b = 0.05
        nu = np.exp(-d / 0.37)
        EN_a = 0.481 * N_val[a_type] / R_a
        EN_b = 0.481 * N_val[b_type] / R_b
        CN_a = val_arr[a_type] / (nu * nu_factor[a1 - 1])
        CN_b = val_arr[b_type] / (nu * nu_factor[a2 - 1])
        denom = np.sqrt(max(EN_a * EN_b, 1e-12))
        f_ab = 0.25 * abs(EN_a - EN_b) / denom
        X_ab = np.sqrt(max(EN_a * EN_b / max(CN_a * CN_b, 1e-12), 0.0))
        H = X_ab * np.exp(-2.7 * f_ab)
        # MATLAB swaps a/b here (:263-264): b = atom1, a = atom2.
        b, a = a1, a2
        k1, k2, k3 = int(row[4]), int(row[5]), int(row[6])
        v = coords[b - 1] + np.array([k1, k2, k3], dtype=float) - coords[a - 1]
        dr = v @ lat
        norm = np.linalg.norm(dr)
        if norm <= 0 or a == b:
            continue
        c = _matlab_round(dr / norm * 1e6) / 1e6
        outer = H * np.outer(c, c)
        A, B = (a - 1) * 3, (b - 1) * 3
        D[A:A + 3, A:A + 3] += outer
        D[A:A + 3, B:B + 3] -= outer
        D[B:B + 3, B:B + 3] += outer
        D[B:B + 3, A:A + 3] -= outer

    # -- 6. eig(D) (:323-325) ---------------------------------------------
    w, v = np.linalg.eig(D)
    freq = np.abs(np.real(w))
    eigvector = np.real(v)
    return freq, eigvector


def _rotation_matrix(axis, theta):
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    t = 1 - c
    return np.array([
        [c + t * x * x, t * x * y - s * z, t * x * z + s * y],
        [t * y * x + s * z, c + t * y * y, t * y * z - s * x],
        [t * z * x - s * y, t * z * y + s * x, c + t * z * z],
    ])


def move_along_soft_mode(molecules, eigvector, mtype_list, stdmol, new_coord2zmatrix):
    """move_along_SoftMode_molMutation.m.

    ``eigvector`` is one (scaled) eigenvector column, length 3*N_atoms, in the
    species-blocked COORDINATES order.  Returns ``(new_molecules, deviation)``.
    """
    disp = np.asarray(eigvector, dtype=float).ravel()
    n_mols = len(mtype_list)
    translations = np.zeros((n_mols, 3))
    rotations = np.zeros((n_mols, 3))
    normrot = np.zeros(n_mols)
    centered = []
    centers = []
    count = 0
    for i in range(n_mols):
        numatom = len(stdmol[mtype_list[i] - 1].types)
        coords = np.asarray(molecules[i].MOLCOORS, dtype=float).copy()
        center = np.asarray(molecules[i].MOLCENTER, dtype=float)
        rel = coords - center
        centered.append(rel)
        centers.append(center)
        total_inertia = float(np.sum(rel ** 2))
        for j in range(numatom):
            d = disp[count * 3:count * 3 + 3]
            direction = rel[j]
            n = np.linalg.norm(direction)
            if n < 0.05:
                translation, rotation = d, np.zeros(3)
            else:
                translation = np.dot(d, direction) * direction / (n * n)
                rotation = np.cross(d, direction)
            translations[i] += translation
            rotations[i] += rotation
            count += 1
        translations[i] /= numatom
        if numatom > 1:
            rotations[i] /= total_inertia
        normrot[i] = np.linalg.norm(rotations[i])

    new_molecules = []
    deviation = []
    for i in range(n_mols):
        std = stdmol[mtype_list[i] - 1]
        numatom = len(std.types)
        rel = centered[i].copy()
        if numatom > 1 and normrot[i] > 0:
            rel = rel @ _rotation_matrix(rotations[i] / normrot[i], normrot[i])
        new_center = centers[i] + translations[i]
        new_coords = rel + new_center
        mol = _clone_molecule(molecules[i])
        mol.MOLCOORS = new_coords
        mol.MOLCENTER = new_center
        mol.ZMATRIX = np.real(new_coord2zmatrix(new_coords, std.format))
        deviation.append(new_coords - np.asarray(molecules[i].MOLCOORS, dtype=float))
        new_molecules.append(mol)
    # Molecules may have different atom counts, so keep this as a ragged list
    # (MATLAB stores it as a 3-D array only when all molecules are identical).
    return new_molecules, deviation


def _clone_molecule(mol):
    # Shallow copy: the fields we overwrite (MOLCOORS/MOLCENTER/ZMATRIX) are
    # replaced with fresh arrays, the rest are read-only here.
    import copy
    return copy.copy(mol)


def lattice_mutation(old_lattice, mutation_rate):
    """lattice_Mutation.m -- Gaussian strain + volume restoration.

    Returns ``(new_lattice, strain_matrix)``.
    """
    from gpevo.matrix import det, latConverter

    old = np.asarray(old_lattice, dtype=float).reshape(3, 3)
    eps = np.random.randn(6) * float(mutation_rate)
    strain = np.eye(3)
    strain[0, 0] += eps[0]
    strain[1, 1] += eps[1]
    strain[2, 2] += eps[2]
    for p, q, e in ((0, 1, eps[3]), (0, 2, eps[4]), (1, 2, eps[5])):
        strain[p, q] = strain[q, p] = e / 2
    new_lat = old @ strain
    vol_new, vol_old = det(new_lat), det(old)
    if not (np.isfinite(vol_new) and np.isfinite(vol_old)
            and vol_new > 1e-6 and vol_old > 1e-6):
        return old, np.eye(3)
    params = np.asarray(latConverter(new_lat), dtype=float).reshape(-1)[:6].copy()
    params[:3] *= (vol_old / vol_new) ** (1.0 / 3.0)
    return np.asarray(latConverter(params), dtype=float).reshape(3, 3), strain


def move_all_mol_mutation(molecules, numMols, lattice, order, how_many_mut):
    """move_all_mol_Mutation.m -- order-weighted random centre displacement.

    Returns the new (wrapped) fractional coordinates of the molecule centres.
    """
    from gpevo.matrix import latConverter

    lat = np.asarray(lattice, dtype=float).reshape(3, 3)
    n = int(sum(numMols))
    cart = np.vstack([np.asarray(mol.ZMATRIX[0], dtype=float) for mol in molecules[:n]])
    frac = cart @ np.linalg.inv(lat)
    lengths = np.asarray(latConverter(lat), dtype=float).reshape(-1)[:3]
    ord_arr = np.asarray(order, dtype=float).ravel()[:n]
    ranking = np.argsort(ord_arr, kind="stable")
    r1, rN = ord_arr[ranking[0]], ord_arr[ranking[-1]]
    if rN > r1:
        koef = (rN - ord_arr[ranking]) / (rN - r1)
        deviat = np.random.randn(n, 3) * (float(how_many_mut) * koef[:, None])
        frac[ranking] += deviat / lengths
        frac[ranking] -= np.floor(frac[ranking])
    return frac
