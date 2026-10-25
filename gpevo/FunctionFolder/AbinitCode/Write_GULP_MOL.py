"""Python equivalent of Write_GULP_MOL.m."""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G
from gpevo.matrix import latConverter


def element_symbol(z):
    names = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S",
             "Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn"]
    return names[z - 1] if 1 <= z <= len(names) else f"Z{z}"


def Write_GULP_MOL(ind_no, write_conn=0):
    ORG = G["ORG_STRUC"]
    s = G["POP_STRUC"].POPULATION[ind_no]
    lattice = np.asarray(s.LATTICE, dtype=float).reshape(3, 3)
    num_mols = [int(v) for v in s.numMols]
    types_alist = [int(v) for v in s.typesAList]
    mtype_list = [int(v) for v in s.MtypeLIST]
    molecules = s.MOLECULES
    stdmol = ORG.STDMOL

    new_coords = []
    for i in range(sum(num_mols)):
        new_coords.append(np.asarray(molecules[i].MOLCOORS, dtype=float).reshape(-1, 3))
    if new_coords:
        new_coords = np.concatenate(new_coords, axis=0) @ np.linalg.inv(lattice)
    else:
        new_coords = np.zeros((0, 3))

    lines = ["cell"]
    lat6 = latConverter(lattice).reshape(-1)
    lat6[3:6] = np.degrees(lat6[3:6])
    lines.append(" ".join(f"{x:8.4f}" for x in lat6[:6]))
    lines.append("fractional")

    count_atoms = {}
    for atom_idx in range(len(types_alist)):
        itype = types_alist[atom_idx]
        # Determine molecule/atom-in-molecule
        i, j = _local_id(atom_idx, mtype_list, stdmol)
        mol_std = stdmol[mtype_list[i - 1] - 1]
        charge = float(mol_std.charge[j - 1, 0]) if hasattr(mol_std, "charge") and len(mol_std.charge) else 0.0
        label = element_symbol(itype)
        c = new_coords[atom_idx]
        lines.append(f"{label:>4s} {c[0]:8.4f} {c[1]:8.4f} {c[2]:8.4f} core {charge:8.6f}")
    if write_conn:
        count = 0
        for mi, mtype in enumerate(mtype_list):
            std = stdmol[mtype - 1]
            n = len(std.types)
            for j in range(1, n + 1):
                cn = std.CN[j - 1] if hasattr(std, "CN") and len(std.CN) >= j else []
                for k in cn:
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    if 0 < kk and kk > j:
                        lines.append(f"connect {count + j:4d} {count + kk:4d}")
            count += n
    with open("input", "a+") as fp:
        fp.write("\n".join(lines) + "\n")


def _local_id(atom_idx, mtype_list, stdmol):
    count = 0
    for mi in range(len(mtype_list)):
        mtype = mtype_list[mi]
        n = len(stdmol[mtype - 1].types)
        if atom_idx < count + n:
            return mi + 1, atom_idx - count + 1
        count += n
    return len(mtype_list), atom_idx - count + 1
