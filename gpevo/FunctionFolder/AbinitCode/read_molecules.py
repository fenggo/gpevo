"""Python equivalent of read_molecules.m for USPEX MOL_* files."""
from __future__ import annotations

import glob
import math
import os

import numpy as np

from gpevo.runtime import G, AttrDict, struct, unix, isempty
from gpevo.FunctionFolder.sys.covalentRadius import covalentRadius
from gpevo.log_utils import console_print


def parse_mol_atom_line(line, n_col):
    toks = line.replace(",", " ").split()
    if not toks:
        return None
    symbol = toks[0]
    numbers = []
    for tok in toks[1:]:
        try:
            numbers.append(float(tok))
        except ValueError:
            continue
    if len(numbers) < 7:
        return None
    return symbol, numbers


def read_molecules():
    ORG = G["ORG_STRUC"]
    home = ORG.homePath

    mol_files = []
    for pattern in ("MOL_[1-9]", "MOL_[0-9]"):
        mol_files.extend(sorted(glob.glob(os.path.join(home, pattern))))
    # robust: find all MOL_<int>
    candidates = []
    for name in os.listdir(home):
        if name.startswith("MOL_") and name[4:].isdigit():
            candidates.append(name)
    files = []
    for name in sorted(candidates, key=lambda x: int(x[4:])):
        files.append(os.path.join(home, name))

    if not files:
        console_print("For Molecules, you did not give MOL_x files")
        console_print("The program is STOPS here........")
        raise RuntimeError("Missing MOL_* files required for calculationType 310")

    n_files = len(files)
    if not ORG.numMols:
        ORG.numMols = [1] * n_files
    elif len(ORG.numMols) > n_files:
        console_print(f"  Warning: numSpecies has {len(ORG.numMols)} entries "
                      f"but only {n_files} MOL_x files; using the first "
                      f"{n_files} molecule types")
        ORG.numMols = ORG.numMols[:n_files]
    # numSpecies gives one entry per molecule type, so only MOL_1..MOL_N are
    # read -- extra MOL_x files in the run directory are simply not part of the
    # system (read_molecules.m loops over numMols the same way).  A single
    # entry therefore means a single-component cell, not "plus the others".
    files = files[:len(ORG.numMols)]

    stdmol = []
    for idx, path in enumerate(files):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        if not lines:
            raise RuntimeError(f"MOL file {path} is empty")

        name = lines[0]
        if "charge" in name.lower():
            n_col = 9
        else:
            n_col = 8

        raw_counts = lines[1].split()
        num_atoms = None
        for tok in raw_counts:
            try:
                num_atoms = int(float(tok))
                break
            except ValueError:
                pass
        if num_atoms is None:
            raise RuntimeError(f"Cannot parse atom count from {path}")

        mol = AttrDict()
        mol.name = name
        mol.format = np.zeros((num_atoms, 3), dtype=int)
        mol.types = []
        mol.molecule = np.zeros((num_atoms, 3))
        mol.coords = np.zeros((num_atoms, 3))
        mol.charge = np.zeros((num_atoms, 1))
        mol.optFlags = np.ones((num_atoms, 3))
        mol.index = list(range(1, num_atoms + 1))
        atoms = []

        for row in lines[2:2 + num_atoms]:
            parsed = parse_mol_atom_line(row, n_col)
            if parsed is None:
                continue
            symbol, nums = parsed
            atoms.append((symbol, nums))

        if len(atoms) < num_atoms:
            raise RuntimeError(f"Not enough atom lines in {path}: {len(atoms)}/{num_atoms}")

        # Determine element ID from symbol
        from gpevo.FunctionFolder.USPEX.src.createORG_System import elementID
        for i, (symbol, nums) in enumerate(atoms):
            z = elementID(symbol)
            try:
                type_idx = list(ORG.atomType).index(z) + 1
            except ValueError:
                type_idx = 0
            mol.types.append(type_idx)
            mol.molecule[i, 0:3] = nums[0:3]
            # fields 5:7 are internal coordinate references (1-based) when present
            if len(nums) >= 6:
                mol.format[i, 0] = max(0, int(round(nums[3])))
                mol.format[i, 1] = max(0, int(round(nums[4])))
                mol.format[i, 2] = max(0, int(round(nums[5])))
            if len(nums) >= 7:
                mol.optFlags[i, 2] = nums[6]
            if n_col == 9 and len(nums) >= 8:
                mol.charge[i, 0] = nums[7]

        mol.molecule = np.asarray(mol.molecule, dtype=float)
        mol.molcenter = np.mean(mol.molecule, axis=0)
        mol.types = mol.types or [1] * num_atoms

        # Covalent radius/radius around molcenter
        distances = np.linalg.norm(mol.molecule - mol.molcenter, axis=1)
        if num_atoms == 1:
            z = ORG.atomType[mol.types[0] - 1 if mol.types[0] > 0 else 0]
            mol.radi = 0.8 * covalentRadius(z) if isinstance(covalentRadius(z), (int, float)) else 0.8
        else:
            mol.radi = float(np.max(distances) + 0.4)

        # Assign internal-format with default connection topologies
        if mol.format.shape[0] != num_atoms or mol.format.shape[1] < 3:
            fmt = np.zeros((num_atoms, 3), dtype=int)
            fmt[0] = [1, 1, 1]
            for i in range(1, num_atoms):
                fmt[i] = [i, i, 1]
            mol.format = fmt
        mol.format = mol.format.astype(int)

        # ZMATRIX must describe the template through the same connectivity the
        # molecule checks use (read_molecules.m: NEW_coord2Zmatrix(coords, format)).
        from gpevo.FunctionFolder.USPEX.src.NEW_coord2Zmatrix import NEW_coord2Zmatrix

        mol.ZMATRIX = NEW_coord2Zmatrix(mol.molecule, mol.format)

        mol.num_optFlags = max(0, int(np.sum(mol.optFlags)) - 6)
        stdmol.append(mol)

    ORG.STDMOL = stdmol
    # Reference intramolecular bond tables (built once at startup; the molecule
    # check compares every relaxed individual against these).
    from gpevo.FunctionFolder.sys.mol_bond_table import build_reference_bond_tables
    build_reference_bond_tables()
    # rad_ij, center minimum distance matrix
    n = len(stdmol)
    cm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cm[i, j] = stdmol[i].radi + stdmol[j].radi
    ORG.CenterminDistMatrice = cm

    # Optional MolCenters override
    from gpevo.inputparser import InputParser, scannum
    p = InputParser(os.path.join(home, "INPUT.txt"))
    block = p.block("MolCenters", "EndMol")
    if block:
        # Each line is one row of the n x n centre-distance matrix.  scannum
        # collapses a single number to a scalar, so normalise to a row first:
        # a one-molecule-type run writes MolCenters as a single value.
        rows = []
        for line in block:
            value = scannum(line)
            if value is None:
                continue
            rows.append(list(value) if isinstance(value, (list, tuple)) else [value])
        if len(rows) == n and all(len(row) >= n for row in rows):
            for i in range(n):
                row = rows[i]
                for j in range(i, n):
                    cm[i, j] = cm[j, i] = float(row[j])
            ORG.CenterminDistMatrice = cm

    G["ORG_STRUC"] = ORG
