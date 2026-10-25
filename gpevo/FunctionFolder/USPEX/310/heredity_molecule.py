"""Python port of ``FunctionFolder/USPEX/310/heredity_molecule.m``.

Slice heredity on the molecule centres: a random lattice direction and a
random cut fraction split the molecules of the two parents, the composition is
corrected back to ``ORG.numMols`` and the two lattices are mixed.  The returned
``mol_table`` tells which parent and which parent molecule each offspring
molecule comes from.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G
from gpevo.matrix import latConverter


def _sort_order(order, indices, descending=False):
    """Sort candidate molecule ids (1-based) by their ``order`` value."""
    order = np.asarray(order, dtype=float).reshape(-1)

    def value(mol_id):
        idx = int(mol_id) - 1
        return float(order[idx]) if 0 <= idx < order.size else 0.0

    pairs = sorted(((int(i), value(i)) for i in indices), key=lambda p: p[1], reverse=bool(descending))
    return [p[0] for p in pairs]


def _characteristic_length(lat, dimension, num_mols):
    """Length of the cell along ``dimension`` and characteristic slab size."""
    lat = np.asarray(lat, dtype=float).reshape(3, 3)
    det = abs(float(np.linalg.det(lat)))
    if dimension == 1:
        cross = np.cross(lat[1], lat[2])
    elif dimension == 2:
        cross = np.cross(lat[0], lat[2])
    else:
        cross = np.cross(lat[0], lat[1])
    length = det / max(float(np.linalg.norm(cross)), 1e-12)
    total = max(int(sum(num_mols)), 1)
    return length, 0.5 * (det / total) ** (1.0 / 3.0)


def heredity_molecule(par_one, par_two):
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    rng = np.random.random

    ordering_on = int(getattr(ORG, "ordering", 0))
    frac_frac = 0.25 + rng() / 2.0

    first = POP.POPULATION[par_one]
    second = POP.POPULATION[par_two]
    num_mols1 = [int(v) for v in first.numMols]
    num_mols2 = [int(v) for v in second.numMols]

    lat1 = np.asarray(first.LATTICE, dtype=float).reshape(3, 3)
    lat2 = np.asarray(second.LATTICE, dtype=float).reshape(3, 3)
    parent1 = np.array([np.asarray(m.MOLCENTER, dtype=float) for m in first.MOLECULES]) @ np.linalg.inv(lat1)
    parent2 = np.array([np.asarray(m.MOLCENTER, dtype=float) for m in second.MOLECULES]) @ np.linalg.inv(lat2)
    order1 = np.array([float(getattr(m, "order", 0.0)) for m in first.MOLECULES])
    order2 = np.array([float(getattr(m, "order", 0.0)) for m in second.MOLECULES])

    dimension = int(np.random.randint(1, 4))
    dim = dimension - 1

    perc = float(getattr(ORG, "percSliceShift", 1.0))
    if rng() > perc:
        offset = rng(2)
        parent1[:, dim] = np.mod(parent1[:, dim] + offset[0], 1.0)
        parent2[:, dim] = np.mod(parent2[:, dim] + offset[1], 1.0)
    else:
        offset = rng(6)
        parent1 = np.mod(parent1 + offset[0:3], 1.0)
        parent2 = np.mod(parent2 + offset[3:6], 1.0)

    which1 = np.nonzero(parent1[:, dim] < frac_frac)[0] + 1  # 1-based molecule ids
    which2 = np.nonzero(parent2[:, dim] > frac_frac)[0] + 1

    if ordering_on:
        cor_dir = int(getattr(ORG, "cor_dir", 1))
        corr = float(getattr(ORG, "correlation_coefficient", 1))
        length1, char1 = _characteristic_length(lat1, dimension, num_mols1)
        length2, char2 = _characteristic_length(lat2, dimension, num_mols2)

        def n_slabs(length, char):
            denom = char + (length - char) * (np.cos(corr * np.pi / 2)) ** 2
            return int(max(0, round(length / max(denom, 1e-12))))

        def mean_or_fallback(order, ids):
            if len(ids):
                return float(np.mean(order[np.asarray(ids, dtype=int) - 1]))
            return 0.0 if cor_dir <= 0 else 1.0

        ord1 = mean_or_fallback(order1, which1)
        ord2 = mean_or_fallback(order2, which2)
        for _ in range(n_slabs(length1, char1)):
            extra = rng()
            trial = np.mod(parent1[:, dim] + extra, 1.0)
            ids = np.nonzero(trial < frac_frac)[0] + 1
            value = mean_or_fallback(order1, ids)
            if (cor_dir <= 0 and ord1 > value) or (cor_dir == 1 and ord1 < value):
                continue
            parent1[:, dim] = trial
            which1 = ids
            ord1 = value
        for _ in range(n_slabs(length2, char2)):
            extra = rng()
            trial = np.mod(parent2[:, dim] + extra, 1.0)
            ids = np.nonzero(trial > frac_frac)[0] + 1
            value = mean_or_fallback(order2, ids)
            if (cor_dir <= 0 and ord2 > value) or (cor_dir == 1 and ord2 < value):
                continue
            parent2[:, dim] = trial
            which2 = ids
            ord2 = value

    n_types = len(ORG.numMols)
    ion_ch1 = [0] + [int(v) for v in np.cumsum(num_mols1)]
    ion_ch2 = [0] + [int(v) for v in np.cumsum(num_mols2)]
    desired = [int(v) for v in ORG.numMols]

    offspring_rows = []
    mol_table = []
    for ind in range(1, n_types + 1):
        sel1 = [i for i, w in enumerate(which1) if ion_ch1[ind - 1] < w <= ion_ch1[ind]]
        sel2 = [i for i, w in enumerate(which2) if ion_ch2[ind - 1] < w <= ion_ch2[ind]]
        ions1 = list(sel1)
        ions2 = list(sel2)
        count = len(ions1) + len(ions2)
        candidates = []
        supplement = []

        if count < desired[ind - 1]:
            howmany = desired[ind - 1] - count
            from1 = int(sum(1 for _ in range(howmany) if rng() < frac_frac))
            from2 = howmany - from1
            # ``from1`` molecules are taken from parent 2 (below the cut) ...
            pool1 = [i + ion_ch2[ind - 1] + 1 for i in range(ion_ch2[ind] - ion_ch2[ind - 1])
                     if parent2[i + ion_ch2[ind - 1], dim] < frac_frac]
            pool2 = [i + ion_ch1[ind - 1] + 1 for i in range(ion_ch1[ind] - ion_ch1[ind - 1])
                     if parent1[i + ion_ch1[ind - 1], dim] > frac_frac]
            if ordering_on:
                pool1 = _sort_order(order2, pool1, descending=(cor_dir <= 0))
                pool2 = _sort_order(order1, pool2, descending=(cor_dir <= 0))
            else:
                np.random.shuffle(pool1)
                np.random.shuffle(pool2)
            take1 = pool1[:from1]
            take2 = pool2[:from2]
            for mol_id in take1:
                candidates.append(parent2[mol_id - 1])
                supplement.append([2, mol_id, ind])
            for mol_id in take2:
                candidates.append(parent1[mol_id - 1])
                supplement.append([1, mol_id, ind])
        elif count > desired[ind - 1]:
            howmany = count - desired[ind - 1]
            del1 = int(sum(1 for _ in range(howmany) if rng() < frac_frac))
            del2 = howmany - del1
            if len(ions1) < del1:
                del2 += del1 - len(ions1)
                del1 = len(ions1)
            elif len(ions2) < del2:
                del1 += del2 - len(ions2)
                del2 = len(ions2)
            if ordering_on:
                drop1 = _sort_order(order1, [which1[i] for i in ions1], descending=not (cor_dir <= 0))[:del1]
                drop2 = _sort_order(order2, [which2[i] for i in ions2], descending=not (cor_dir <= 0))[:del2]
                ions1 = [i for i in ions1 if which1[i] not in drop1]
                ions2 = [i for i in ions2 if which2[i] not in drop2]
            else:
                for _ in range(min(del1, len(ions1))):
                    ions1.pop(int(np.random.randint(0, len(ions1))))
                for _ in range(min(del2, len(ions2))):
                    ions2.pop(int(np.random.randint(0, len(ions2))))

        for i in ions1:
            offspring_rows.append(parent1[which1[i] - 1])
            mol_table.append([1, int(which1[i]), ind])
        for i in ions2:
            offspring_rows.append(parent2[which2[i] - 1])
            mol_table.append([2, int(which2[i]), ind])
        for row in candidates:
            offspring_rows.append(np.asarray(row, dtype=float))
        mol_table.extend(supplement)

    offspring = np.array(offspring_rows, dtype=float).reshape(-1, 3)

    if int(getattr(ORG, "constLattice", 0)):
        potential_lattice = np.asarray(getattr(ORG, "lattice", lat1), dtype=float).reshape(3, 3)
    else:
        frac_lattice = rng()
        temp = frac_lattice * lat1 + (1.0 - frac_lattice) * lat2
        if float(np.linalg.det(temp)) < 0:
            temp = -temp
        lat_vol = float(np.linalg.det(lat1)) * frac_frac + float(np.linalg.det(lat2)) * (1.0 - frac_frac)
        ratio = lat_vol / float(np.linalg.det(temp))
        if ratio <= 0:
            ratio = 1.0
        params = np.asarray(latConverter(temp), dtype=float).reshape(-1)[:6].copy()
        params[:3] = params[:3] * (ratio ** (1.0 / 3.0))
        potential_lattice = np.asarray(latConverter(params), dtype=float).reshape(3, 3)

    return list(ORG.numMols), offspring, potential_lattice, frac_frac, dimension, offset, mol_table
