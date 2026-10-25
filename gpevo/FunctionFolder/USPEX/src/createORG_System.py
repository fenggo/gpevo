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
    return 0


def valenceElectronsNumber(atomic_number):
    return 0


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
        ORG.numIons = scannum(" ".join(blocker))
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
        ORG.goodBonds = M
    else:
        ORG.goodBonds = None

    G["ORG_STRUC"] = ORG
