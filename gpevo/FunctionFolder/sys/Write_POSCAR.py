"""Python equivalents of ``sys/Write_POSCAR.m`` and ``sys/Write_POSCAR_order.m``."""
from __future__ import annotations

import numpy as np

from gpevo.matrix import latConverter


_ELEMENTS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S",
    "Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru",
    "Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce",
    "Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf",
    "Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
]


def _symbol(z):
    z = int(round(z))
    return _ELEMENTS[z - 1] if 1 <= z <= len(_ELEMENTS) else f"Z{z}"


def _count(symg):
    if symg is None or str(symg).strip().upper() == "N/A":
        return 0
    try:
        return int(round(float(symg)))
    except Exception:
        return 0


def _header(count, symg, lattice):
    lat6 = np.asarray(latConverter(np.asarray(lattice, dtype=float))).reshape(-1)[:6]
    lat6 = lat6.copy()
    lat6[3:6] = np.degrees(lat6[3:6])
    return (
        f"EA{int(count):-4d} {lat6[0]:6.3f} {lat6[1]:6.3f} {lat6[2]:6.3f} "
        f"{lat6[3]:5.2f} {lat6[4]:5.2f} {lat6[5]:5.2f} Sym.group: {_count(symg):4d}\n"
    )


def _species_line(atom_type, num_ions):
    if atom_type is None or sum(int(x) for x in atom_type) == 0:
        return ""
    line = ""
    for i, n in enumerate(num_ions):
        if int(n) > 0:
            if i < len(atom_type):
                line += f"{_symbol(atom_type[i]):4s}"
            else:
                line += f"{'X':4s}"
    return line + "\n"


def _num_ions_line(num_ions):
    return "".join(f"{int(n):4d} " for n in num_ions if int(n) > 0) + "\n"


def _coord_lines(coor, order=None):
    coords = np.asarray(coor, dtype=float).reshape(-1, 3)
    lines = []
    if order is None:
        for xyz in coords:
            lines.append(f"{xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}\n")
    else:
        orders = np.asarray(order, dtype=float).reshape(-1)
        if len(orders) != len(coords):
            orders = np.zeros(len(coords), dtype=float)
        for xyz, o in zip(coords, orders):
            lines.append(f"{xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f} {o:12.6f}\n")
    return lines


def Write_POSCAR(atomType, Count, symg, numIons, lattice, coor, filename="POSCAR"):
    lattice = np.asarray(lattice, dtype=float).reshape(3, 3)
    text = [_header(Count, symg, lattice), "1.0000\n"]
    for row in lattice:
        text.append(f"{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f}\n")
    text.append(_species_line(atomType, numIons))
    text.append(_num_ions_line(numIons))
    text.append("Direct\n")
    text.extend(_coord_lines(coor))
    with open(filename, "w") as fp:
        fp.write("".join(text))


def Write_POSCAR_order(atomType, Count, symg, numIons, lattice, coor, order,
                       filename="POSCAR_order"):
    lattice = np.asarray(lattice, dtype=float).reshape(3, 3)
    text = [_header(Count, symg, lattice), "1.0000\n"]
    for row in lattice:
        text.append(f"{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f}\n")
    text.append(_species_line(atomType, numIons))
    text.append(_num_ions_line(numIons))
    text.append("Direct\n")
    text.extend(_coord_lines(coor, order))
    with open(filename, "w") as fp:
        fp.write("".join(text))
