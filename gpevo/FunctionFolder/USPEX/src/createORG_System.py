"""Python equivalent of createORG_System.m (310-relevant subset)."""
from __future__ import annotations

import os
import math

import numpy as np

from gpevo.runtime import G, struct
from gpevo.inputparser import InputParser, scannum
from gpevo.matrix import det


_ELEMENTS = [
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn",
    "Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
    "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
    "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn"
]


def elementID(symbol):
    s = str(symbol).strip().capitalize()
    if s in _ELEMENTS:
        return _ELEMENTS.index(s) + 1
    return 0


def valence_el(symbol):
    """valence.m -- formal valence by element symbol."""
    from gpevo.FunctionFolder.USPEX.src.softmode import valence
    return valence(elementID(symbol))


def valenceElectronsNumber(atomic_number):
    """valenceElectronsNumber.m -- valence electron count by atomic number."""
    from gpevo.FunctionFolder.USPEX.src.softmode import valence_electrons
    return valence_electrons(atomic_number)


def createORG_System(inputFile):
    ORG = G["ORG_STRUC"]
    p = InputParser(inputFile)

    raw = p.scalar("optType")
    if raw is not None:
        s = str(raw)
        if s.startswith("-"):
            ORG.opt_sign = -1
            s = s[1:]
        if s.lower() == "enthalpy":
            s = "1"
        elif s.lower() == "volume":
            s = "2"
        elif s.lower() == "hardness":
            s = "3"
        elif s.lower() == "struc_order":
            s = "4"
        elif s.lower() == "aver_dist":
            s = "5"
        elif s.lower() == "diel_sus":
            s = "6"
        elif s.lower() == "gap":
            s = "7"
        elif s.lower() == "diel_gap":
            s = "8"
        elif s.lower() == "mag_moment":
            s = "9"
        elif s.lower() == "quasientropy":
            s = "10"
        elif s.lower() == "density":
            s = "11"
        elif s.lower() == "den&ent":
            s = "12"
        elif s.lower() == "distance":
            s = "13"
        elif s.lower() == "powerfactor":
            s = "14"
        try:
            ORG.optType = abs(int(s))
        except Exception:
            pass

    raw = p.scalar("calculationType")
    if raw is not None:
        s = str(raw).strip()
        neg = s.startswith("-")
        if neg:
            s = s[1:]
        ORG.dimension = (-1 if neg else 1) * int(s[0])
        if len(s) >= 2:
            ORG.molecule = int(s[1])
        if len(s) >= 3:
            ORG.varcomp = int(s[2])

    blocker = p.block("numSpeci", "EndNumSpeci")
    if blocker:
        vals = scannum(" ".join(blocker))
        # scannum collapses a single number to a scalar; numIons is always a
        # per-species list, and a one-molecule-type run (e.g. "8") must not
        # become the bare int that breaks every `for x in ORG.numIons`.
        ORG.numIons = list(vals) if isinstance(vals, (list, tuple)) else [vals]
    else:
        ORG.numIons = [1]

    if ORG.molecule == 1:
        ORG.numMols = list(map(int, ORG.numIons))

    atom_block = p.block("atomType", "EndAtomType")
    if atom_block:
        ORG.atomType = [elementID(tok) for tok in " ".join(atom_block).split() if tok]
    else:
        ORG.atomType = [elementID(tok) for tok in str(ORG.numSpeciesSymbols or "")]

    # Ionic distances (atom-atom hard core)
    hard_block = p.block("IonDistances", "EndDistances")
    if hard_block:
        hard_core = []
        for line in hard_block:
            vals = scannum(line)
            if vals is not None:
                hard_core.append(vals if isinstance(vals, list) else [vals])
        if hard_core:
            n = len(ORG.atomType)
            M = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    try:
                        M[i, j] = hard_core[i][j]
                    except Exception:
                        M[i, j] = 1.2
            ORG.minDistMatrice = M
    else:
        ORG.minDistMatrice = None

    good_bond_block = p.block("goodBonds", "EndGoodBonds")
    if good_bond_block:
        rows = [scannum(x) for x in good_bond_block if scannum(x) is not None]
        n = len(ORG.atomType)
        M = np.zeros((n, n))
        if len(rows) == 1:
            M.fill(rows[0][0] if isinstance(rows[0], list) else float(rows[0]))
        else:
            for i in range(min(n, len(rows))):
                for j in range(min(n, len(rows[i] if isinstance(rows[i], list) else []))):
                    M[i, j] = rows[i][j]
            # MATLAB symmetrises the upper triangle (createORG_System.m:242-245).
            M = np.triu(M) + np.triu(M, 1).T
        ORG.goodBonds = M
    else:
        # calcDefaultGoodBonds.m: sqrt(gB_i * gB_j).
        from gpevo.FunctionFolder.USPEX.src.softmode import default_good_bonds
        ORG.goodBonds = default_good_bonds(ORG.atomType)

    # valences / valenceElectr blocks (createORG_System.m:112-143).  The soft
    # mode force constants need per-species formal valences and valence electron
    # counts; defaults come from the MATLAB lookup tables when no block is given.
    from gpevo.FunctionFolder.USPEX.src.softmode import valence
    n = len(ORG.atomType)
    val_block = p.block("valences", "EndValences")
    if val_block:
        vals = scannum(" ".join(val_block))
        ORG.valences = np.asarray(vals if isinstance(vals, (list, tuple)) else [vals], dtype=float)
    else:
        ORG.valences = np.array([valence(z) for z in ORG.atomType], dtype=float)
    nval_block = p.block("valenceElectr", "EndValenceElectr")
    if nval_block:
        vals = scannum(" ".join(nval_block))
        ORG.NvalElectrons = np.asarray(vals if isinstance(vals, (list, tuple)) else [vals], dtype=float)
    else:
        ORG.NvalElectrons = np.array([valenceElectronsNumber(z) for z in ORG.atomType], dtype=float)
    if ORG.valences.size != n:
        ORG.valences = np.array([valence(z) for z in ORG.atomType], dtype=float)
    if ORG.NvalElectrons.size != n:
        ORG.NvalElectrons = np.array([valenceElectronsNumber(z) for z in ORG.atomType], dtype=float)

    G["ORG_STRUC"] = ORG
