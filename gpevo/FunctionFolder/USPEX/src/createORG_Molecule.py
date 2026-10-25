"""Python equivalent of createORG_Molecule.m."""
from __future__ import annotations

from gpevo.runtime import G
from gpevo.inputparser import InputParser
from gpevo.FunctionFolder.AbinitCode.read_molecules import read_molecules


def createORG_Molecule(inputFile=""):
    ORG = G["ORG_STRUC"]
    if inputFile:
        raw = InputParser(inputFile).scalar("checkMolecules")
        if raw is not None:
            ORG.checkMolecules = int(float(raw))
    G["ORG_STRUC"] = ORG
    read_molecules()
