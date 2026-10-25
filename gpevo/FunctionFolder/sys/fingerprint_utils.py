"""Pure-Python USPEX 310 fingerprint, quasi-entropy and order metrics.

This module is a direct port of the relevant pieces of
``FunctionFolder/sys/makeMatrices.m``, ``fingerprint_calc.m``,
``structureQuasiEntropy.m`` and ``StructureOrder.m``.  It keeps the same
matrix conventions as the Octave implementation, while using the flattened
neighbour-pair representation already used by the Cython ``uspex_fast_core``
bridge, so it can be called directly from the Python package without shelling
out or importing platform-specific C extensions.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

try:
    from gpevo.FunctionFolder.sys import uspex_fast_core as _ufc
except Exception:
    _ufc = None


def _as_int_array(values):
    return np.asarray([int(v) for v in values], dtype=np.int32)


def _cy_fingerprint(coords, lattice, rmax, num_ions, sigma, delta, dimension, intra_map=None, atom_type=None):
    '''Use the bundled Cython uspex_fast_core extension when available.'''
    if _ufc is None:
        return None
    try:
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        lattice = np.ascontiguousarray(lattice, dtype=np.float64).reshape(3, 3)
        num_ions = _as_int_array(num_ions)
        if atom_type is None:
            atom_type = num_ions.copy()
        atom_type = _as_int_array(atom_type)

        dist, cc, bc, ti, tj, shift, N_out, V, _ = _ufc.build_distance_matrix(
            coords, lattice, float(rmax), num_ions, atom_type
        )
        if intra_map is not None and len(dist):
            intra_map = np.ascontiguousarray(intra_map, dtype=np.int8)
            keep = (shift != 0) | (intra_map[cc, bc] == 1)
            if np.any(keep):
                dist = dist[keep]
                cc = cc[keep]
                bc = bc[keep]
                ti = ti[keep]
                tj = tj[keep]
            else:
                dist = np.asarray([], dtype=np.float64)
                cc = bc = ti = tj = np.asarray([], dtype=np.int32)
        order, fing, atom_fing, _ = _ufc.fingerprint_calc(
            dist, cc, bc, ti, tj, N_out, V, num_ions,
            float(rmax), float(sigma), float(delta), int(dimension),
        )
        return (order, fing, atom_fing), V
    except Exception:
        return None


def _labels_and_counts(num_ions):
    """Return per-atom (1-based) type labels and per-atom species counts."""
    num_ions = _as_int_array(num_ions)
    species = int(num_ions.size)
    natom = int(num_ions.sum())
    if natom == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float64),
            species,
            natom,
        )
    labels = np.empty(natom, dtype=np.int32)
    counts = np.empty(natom, dtype=np.float64)
    pos = 0
    for sp in range(species):
        n = int(num_ions[sp])
        labels[pos:pos + n] = sp + 1
        counts[pos:pos + n] = float(n)
        pos += n
    return labels, counts, species, natom


def build_distance_pairs(coords, lattice, rmax, num_ions, dimension=3):
    """Build the flattened periodic neighbour list used by ``fingerprint_calc``.

    The returned tuple is ``(dist, cc, bc, ti, tj, shift, N_out, V, N)``:

    * ``cc`` -- current-cell atom index (0-based), the neighbour/row side in
      Octave's ``makeMatrices``.
    * ``bc`` -- basic-cell atom index (0-based), the column side.
    * ``ti`` -- 1-based type of ``cc`` (Octave ``typ_j``).
    * ``tj`` -- 1-based type of ``bc`` (Octave ``typ_i``).
    * ``shift`` -- 1 whenever a periodic image was used, 0 for zero-shift
      pairs.  This is the flag used for intra-molecular filtering.
    """
    coords = np.asarray(coords, dtype=np.float64)
    lattice = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    labels, counts, species, natom = _labels_and_counts(num_ions)
    V = float(abs(np.linalg.det(lattice)))
    if natom == 0 or not np.all(np.isfinite(lattice)):
        empty = np.zeros((0,), dtype=np.float64)
        empty_i = np.zeros((0,), dtype=np.int32)
        return empty, empty_i.copy(), empty_i.copy(), empty_i.copy(), empty_i.copy(), empty_i.copy(), counts, V, natom

    signum = np.array(
        [
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, -1],
        ],
        dtype=np.float64,
    )
    condition = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.int32,
    )

    vect = np.zeros((13, 3), dtype=np.float64)
    a = lattice[0]
    b = lattice[1]
    c = lattice[2]
    vect[0] = a
    vect[1] = b
    vect[2] = c
    vect[3] = a + b
    vect[4] = a - b
    vect[5] = a + c
    vect[6] = a - c
    vect[7] = c + b
    vect[8] = c - b
    vect[9] = a + b + c
    vect[10] = a + b - c
    vect[11] = a - b + c
    vect[12] = -a + b + c
    abs_vect = np.sqrt(np.einsum("ij,ij->i", vect, vect))
    length = int(np.ceil((rmax + abs_vect.max()) / abs_vect.min())) + 1

    rmax_sq = rmax * rmax
    lat2 = lattice @ lattice.T

    dist_chunks = []
    cc_chunks = []
    bc_chunks = []
    ti_chunks = []
    tj_chunks = []
    shift_chunks = []

    for qi in range(length + 1):
        for qj in range(length + 1):
            for qk in range(length + 1):
                if dimension == 2 and qk > 0:
                    continue
                if dimension == 0 and qi + qj + qk > 0:
                    continue
                for q in range(8):
                    cond = condition[q]
                    if (
                        cond[0] * (qi == 0)
                        + cond[1] * (qj == 0)
                        + cond[2] * (qk == 0)
                    ) != 0:
                        continue
                    shift = signum[q] * np.array([qi, qj, qk], dtype=np.float64)
                    shifted = coords + shift
                    dfrac = shifted[:, None, :] - coords[None, :, :]
                    dcart = dfrac.reshape(-1, 3) @ lattice
                    d2 = np.einsum("ij,ij->i", dcart, dcart).reshape(natom, natom)
                    valid = d2 < rmax_sq
                    if not np.any(valid):
                        continue
                    rows, cols = np.nonzero(valid)
                    dist = np.sqrt(d2[rows, cols])
                    zero_shift = 0 if (qi == 0 and qj == 0 and qk == 0 and q == 0) else 1
                    dist_chunks.append(dist)
                    cc_chunks.append(rows.astype(np.int32))
                    bc_chunks.append(cols.astype(np.int32))
                    ti_chunks.append(labels[rows])
                    tj_chunks.append(labels[cols])
                    shift_chunks.append(np.full(len(rows), zero_shift, dtype=np.int32))

    if dist_chunks:
        return (
            np.concatenate(dist_chunks),
            np.concatenate(cc_chunks),
            np.concatenate(bc_chunks),
            np.concatenate(ti_chunks),
            np.concatenate(tj_chunks),
            np.concatenate(shift_chunks),
            counts,
            V,
            natom,
        )
    empty = np.zeros((0,), dtype=np.float64)
    empty_i = np.zeros((0,), dtype=np.int32)
    return empty, empty_i.copy(), empty_i.copy(), empty_i.copy(), empty_i.copy(), empty_i.copy(), counts, V, natom


def fingerprint_calc(
    dist,
    cc,
    bc,
    ti,
    tj,
    N_out,
    num_ions,
    V,
    rmax=10.0,
    sigma=0.03,
    delta=0.08,
    dimension=3,
):
    """Return ``(order, fing, atom_fing)`` in Octave-compatible shapes."""
    num_ions = _as_int_array(num_ions)
    species = int(num_ions.size)
    total_ions = float(num_ions.sum())
    if species == 0 or len(dist) == 0:
        order = np.zeros(total_ions and int(total_ions) or 0)
        fing = np.zeros((species * species, 0), dtype=np.float64)
        atom_fing = np.zeros((0, species, 0), dtype=np.float64)
        return order, fing, atom_fing

    natom = int(num_ions.sum())
    num_bins = int(round(rmax / delta))
    if num_bins < 2:
        num_bins = 2

    normaliser = 1.0 if dimension != 0 else 0.0
    fing = np.zeros((species * species, num_bins), dtype=np.float64)
    order = np.zeros(natom, dtype=np.float64)
    atom_fing = np.zeros((natom, species, num_bins), dtype=np.float64)
    if normaliser:
        fing[:, 0] = -1.0

    dist = np.asarray(dist, dtype=np.float64)
    cc = np.asarray(cc, dtype=np.intp)
    bc = np.asarray(bc, dtype=np.intp)
    ti = np.asarray(ti, dtype=np.intp)
    tj = np.asarray(tj, dtype=np.intp)
    n_counts = np.asarray(N_out, dtype=np.float64)

    sigm = sigma / np.sqrt(2.0 * np.log(2.0))
    sqrt2_sigma = np.sqrt(2.0) * sigm
    window = 4.0 * sigm
    inv_4pi_delta = 1.0 / (4.0 * np.pi * delta)
    vc_third = (V / total_ions) ** (1.0 / 3.0) if total_ions > 0 else 1.0
    weights = num_ions.astype(np.float64) / total_ions if total_ions > 0 else np.zeros(species)

    for bins in range(2, num_bins + 1):
        ib = bins - 1
        center = delta * (bins - 0.5)
        mask = (dist > 0.0) & (np.abs(dist - center) < window)
        d = dist[mask]
        row = cc[mask]
        col = bc[mask]
        type_row = ti[mask]
        type_col = tj[mask]

        upper = (delta * bins - d) / sqrt2_sigma
        lower = (delta * (bins - 1) - d) / sqrt2_sigma
        np.clip(upper, -5.0, 5.0, out=upper)
        np.clip(lower, -5.0, 5.0, out=lower)
        delt = 0.5 * (erf(upper) - erf(lower))

        inv_r2 = 1.0 / (d * d)
        atom_w = delt * inv_r2 / n_counts[row]
        global_w = delt * inv_r2

        np.add.at(atom_fing[:, :, ib], (col, type_row - 1), atom_w)
        row_idx = (type_col - 1) * species + (type_row - 1)
        np.add.at(fing[:, ib], row_idx, global_w)

        atom_fing[:, :, ib] = V * atom_fing[:, :, ib] * inv_4pi_delta - normaliser
        order += (delta / vc_third) * np.sum(
            weights[None, :] * atom_fing[:, :, ib] ** 2, axis=1
        )

        for si in range(species):
            for sj in range(species):
                denom = 4.0 * np.pi * num_ions[si] * num_ions[sj] * delta
                if denom > 0:
                    fing[si * species + sj, ib] = (
                        V * fing[si * species + sj, ib] / denom - normaliser
                    )

    np.maximum(order, 0.0, out=order)
    np.sqrt(order, out=order)
    return order, fing, atom_fing


def compute_molecular_fingerprint(
    lattice,
    coords,
    num_mols,
    rmax=10.0,
    sigma=0.03,
    delta=0.08,
    dimension=3,
):
    """Fingerprint for molecular center-of-mass sites."""
    num_mols_list = _as_int_array(num_mols)
    fast = _cy_fingerprint(coords, lattice, rmax, num_mols_list, sigma, delta, dimension)
    if fast is not None:
        return fast[0], fast[1]

    pairs = build_distance_pairs(coords, lattice, rmax, num_mols_list, dimension=dimension)
    return fingerprint_calc(
        pairs[0],
        pairs[1],
        pairs[2],
        pairs[3],
        pairs[4],
        pairs[6],
        num_mols_list,
        pairs[7],
        rmax=rmax,
        sigma=sigma,
        delta=delta,
        dimension=dimension,
    ), pairs[7]


def compute_full_atom_fingerprint(
    lattice,
    coords,
    num_ions,
    rmax=10.0,
    sigma=0.03,
    delta=0.08,
    dimension=3,
    intra_map=None,
):
    """Fingerprint for all atoms, optionally filtering intra-molecular pairs."""
    fast = _cy_fingerprint(
        coords, lattice, rmax, _as_int_array(num_ions), sigma, delta, dimension, intra_map=intra_map
    )
    if fast is not None:
        return fast[0], fast[1]

    pairs = build_distance_pairs(coords, lattice, rmax, _as_int_array(num_ions), dimension=dimension)
    dist, cc, bc, ti, tj, shift, N_out, V, N = pairs
    if intra_map is not None and len(dist):
        intra_map = np.asarray(intra_map, dtype=np.int8)
        keep = (shift != 0) | (intra_map[cc, bc] == 1)
        dist = dist[keep]
        cc = cc[keep]
        bc = bc[keep]
        ti = ti[keep]
        tj = tj[keep]

    result = fingerprint_calc(
        dist,
        cc,
        bc,
        ti,
        tj,
        N_out,
        _as_int_array(num_ions),
        V,
        rmax=rmax,
        sigma=sigma,
        delta=delta,
        dimension=dimension,
    )
    return result, V


def intra_mol_dist(mtype_list, num_ions, stdmol):
    """Build the NxN intra-molecular mask: 1 = inter-molecular, 0 = intra."""
    mtype_list = [int(v) for v in mtype_list]
    num_ions = [int(v) for v in num_ions]
    n_types = len(num_ions)
    n_atom = sum(num_ions)
    intra = np.ones((n_atom, n_atom), dtype=np.int8)
    mol_id = np.zeros(n_atom, dtype=np.int32)
    count = [0] * n_types
    running = 0
    for j in range(1, n_types):
        running += num_ions[j - 1]
        count[j] = running

    for mol_idx, mtype in enumerate(mtype_list, start=1):
        mol_types = list(getattr(stdmol[mtype - 1], "types", []))
        for type_idx in range(1, n_types + 1):
            s = sum(1 for t in mol_types if int(t) == type_idx)
            if s > 0:
                start = count[type_idx - 1]
                mol_id[start:start + s] = mol_idx
                count[type_idx - 1] += s

    iu = np.triu_indices(n_atom, k=1)
    same = mol_id[iu[0]] == mol_id[iu[1]]
    intra[iu[0][same], iu[1][same]] = 0
    intra[iu[1][same], iu[0][same]] = 0
    return intra


def _cosine_distance(mat_a, mat_b, weight):
    mat_a = np.asarray(mat_a, dtype=np.float64)
    mat_b = np.asarray(mat_b, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    weight = np.diag(weight)
    prod_ab = mat_a * mat_b
    prod_aa = mat_a * mat_a
    prod_bb = mat_b * mat_b
    coef1 = np.sum(weight @ prod_ab) if weight.shape[0] == prod_ab.shape[0] else np.sum(weight * prod_ab)
    coef2 = np.sum(weight @ prod_aa) if weight.shape[0] == prod_aa.shape[0] else np.sum(weight * prod_aa)
    coef3 = np.sum(weight @ prod_bb) if weight.shape[0] == prod_bb.shape[0] else np.sum(weight * prod_bb)
    denom = np.sqrt(coef2 * coef3)
    if denom <= 0:
        return 1.0
    dist = (1.0 - coef1 / denom) / 2.0
    if abs(dist - 1.0) < 1e-6:
        return 0.99999
    return dist


def structure_quasi_entropy(atom_fing, num_ions, varcomp=0, dimension=3):
    """Direct Python port of ``structureQuasiEntropy.m``."""
    atom_fing = np.asarray(atom_fing, dtype=np.float64)
    num_ions = [int(v) for v in num_ions]
    if atom_fing.ndim != 3 or atom_fing.shape[0] == 0:
        return 0.0
    _, species, _ = atom_fing.shape
    total = float(sum(num_ions))
    if total <= 0:
        return 0.0
    weight = (
        np.ones(species, dtype=np.float64)
        if (varcomp == 1 or dimension == 2)
        else np.asarray(num_ions, dtype=np.float64) / total
    )

    s_q_e = 0.0
    for i, n_i in enumerate(num_ions):
        if n_i <= 1:
            continue
        start = sum(num_ions[:i])
        tmp = 0.0
        pair_count = 0
        for j in range(start, start + n_i - 1):
            for j1 in range(j + 1, start + n_i):
                dist = _cosine_distance(atom_fing[j], atom_fing[j1], weight)
                tmp += (1.0 - dist) * np.log(max(1e-12, 1.0 - dist))
                pair_count += 1
        if pair_count:
            s_q_e += (n_i / total) * (tmp / pair_count)
    return float(-s_q_e)


def structure_order(fing, volume, num_ions, delta, weight=None):
    """Direct Python port of ``StructureOrder.m``."""
    fing = np.asarray(fing, dtype=np.float64)
    if fing.ndim not in (1, 2) or fing.size == 0:
        return 0.0
    if fing.ndim == 1:
        fing = fing.reshape(1, -1)
    rows, cols = fing.shape
    total = float(sum(map(int, num_ions)))
    if total <= 0 or volume <= 0:
        return 0.0
    if weight is None:
        weight = np.ones(rows, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    if weight.size != rows:
        weight = np.ones(rows, dtype=np.float64)
    order_ab = np.sum(fing * fing, axis=1) * delta / (volume / total) ** (1.0 / 3.0)
    order_full = float(np.sum(order_ab * weight))
    return float(np.sqrt(max(0.0, order_full)))


def compute_structure_metrics(
    individual,
    org,
    num_mols=None,
):
    """Compute molecular/full-atom fingerprints and attach fields to ``individual``."""
    if num_mols is None:
        num_mols = [int(v) for v in getattr(individual, "numMols", org.numMols)]
    total_mols = int(sum(num_mols))
    lattice = np.asarray(getattr(individual, "LATTICE"), dtype=np.float64).reshape(3, 3)
    rmax = float(getattr(org, "RmaxFing", 10.0))
    sigma = float(getattr(org, "sigmaFing", 0.03))
    delta = float(getattr(org, "deltaFing", 0.08))
    dimension = int(getattr(org, "dimension", 3))

    molecules = list(getattr(individual, "MOLECULES", []))
    if total_mols and molecules[:total_mols]:
        centers = np.vstack(
            [np.asarray(mol.MOLCENTER, dtype=np.float64) for mol in molecules[:total_mols]]
        )
        center_frac = centers @ np.linalg.inv(lattice)
        (order_mol, _, _), _ = compute_molecular_fingerprint(
            lattice,
            center_frac,
            num_mols,
            rmax=rmax,
            sigma=sigma,
            delta=delta,
            dimension=dimension,
        )
        for mol, value in zip(molecules[:total_mols], order_mol.tolist()):
            mol.order = float(value)

    num_ions = [int(v) for v in getattr(individual, "numIons", [])]
    coords = np.asarray(getattr(individual, "COORDINATES"), dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] != sum(num_ions):
        coords = coords.reshape(-1, 3)
    intra_map = None
    if getattr(org, "molecule", 0) == 1 and len(num_ions):
        mtype = getattr(individual, "MtypeLIST", list(range(1, total_mols + 1)))
        intra_map = intra_mol_dist(mtype, num_ions, org.STDMOL)

    (order, fing, atom_fing), volume = compute_full_atom_fingerprint(
        lattice,
        coords,
        num_ions,
        rmax=rmax,
        sigma=sigma,
        delta=delta,
        dimension=dimension,
        intra_map=intra_map,
    )
    individual.order = np.asarray(order, dtype=np.float64)
    individual.FINGERPRINT = fing
    individual.struc_entr = structure_quasi_entropy(
        atom_fing,
        num_ions,
        varcomp=int(getattr(org, "varcomp", 0)),
        dimension=dimension,
    )
    individual.S_order = structure_order(fing, volume, num_ions, delta, getattr(org, "weight", None))
    return individual
