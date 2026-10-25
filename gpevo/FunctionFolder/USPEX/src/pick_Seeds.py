"""Python port of ``FunctionFolder/USPEX/src/pick_Seeds.m``.

Reads the structures stored in ``Seeds/POSCARS`` (or ``Seeds/POSCARS_<gen>``)
and inserts them into the next population, exactly where MATLAB puts them:
in generation 1 they land at ``ORG.initialPopSize`` (which counts up as seeds
are consumed) and afterwards they are appended.  The consumed file is moved to
``POSCARS_<gen>`` so the same seeds are not re-read every generation.
"""
from __future__ import annotations

import os

import numpy as np

from gpevo.runtime import G, AttrDict
from gpevo.log_utils import console_print


def element_id(symbol):
    from gpevo.FunctionFolder.USPEX.src.createORG_System import elementID

    return elementID(symbol)


def _floats(line):
    """Return the numeric tokens of a line, or None when it is not numeric."""
    tokens = line.split()
    if not tokens:
        return None
    values = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            return None
    return values


def _ints(line):
    """Return the integer tokens of a line, or None when it is not numeric."""
    values = _floats(line)
    if values is None:
        return None
    return [int(round(v)) for v in values]


def _read_poscar_blocks(path):
    """Parse a POSCARS file with several concatenated VASP-style blocks."""
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        lines = [line.rstrip("\n") for line in fp]

    blocks = []
    i = 0
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break

        comment = lines[i]
        i += 1
        if i >= len(lines):
            break
        scale_tokens = _floats(lines[i])
        if scale_tokens is None:
            break
        scale = float(scale_tokens[0])
        i += 1
        if i + 2 >= len(lines):
            break

        try:
            lattice = np.array([_floats(lines[i + k])[:3] for k in range(3)], dtype=float)
        except (TypeError, IndexError):
            break
        i += 3

        # Optional species line followed by the counts line.
        counts = _ints(lines[i]) if i < len(lines) else None
        species = None
        if counts is None:
            species = lines[i].split()
            i += 1
            if i >= len(lines):
                break
            counts = _ints(lines[i])
        i += 1
        if counts is None:
            break

        if i < len(lines) and lines[i].strip().lower().startswith("s"):
            i += 1
        cartesian = False
        if i < len(lines) and lines[i].strip():
            mode = lines[i].strip().lower()
            if mode[0] in ("d", "c", "k"):
                cartesian = mode[0] in ("c", "k")
                i += 1

        n_atoms = int(sum(counts))
        coords = []
        for _ in range(n_atoms):
            if i >= len(lines):
                break
            tokens = lines[i].split()
            i += 1
            if len(tokens) < 3:
                break
            try:
                coords.append([float(v) for v in tokens[:3]])
            except ValueError:
                break
        if len(coords) != n_atoms:
            break

        blocks.append(
            {
                "comment": comment,
                "lattice": np.asarray(lattice, dtype=float) * scale,
                "species": species,
                "counts": counts,
                "coords": np.asarray(coords, dtype=float),
                "cartesian": cartesian,
            }
        )
    return blocks


def _molecule_counts_from_comment(comment):
    """``numMolReadSeeds`` reads the numbers inside the first [ ]."""
    text = comment
    if "[" not in text or "]" not in text:
        return None
    inside = text.split("[", 1)[1].split("]", 1)[0]
    return _ints(inside)


def pick_Seeds():
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    home = getattr(ORG, "homePath", None) or os.getcwd()
    seeds_dir = os.path.join(home, "Seeds")
    generation = int(getattr(POP, "generation", 1))
    poscars = os.path.join(seeds_dir, "POSCARS")
    generation_file = os.path.join(seeds_dir, f"POSCARS_{generation}")

    if not os.path.isdir(seeds_dir):
        return 0

    if os.path.isfile(generation_file):
        if os.path.isfile(poscars):
            with open(generation_file, "a", encoding="utf-8") as out:
                with open(poscars, "r", encoding="utf-8", errors="replace") as src:
                    out.write(src.read())
        with open(generation_file, "r", encoding="utf-8", errors="replace") as src:
            merged = src.read()
        with open(poscars, "w", encoding="utf-8") as out:
            out.write(merged)

    if not os.path.isfile(poscars):
        return 0

    console_print(" ")
    console_print("Read Seeds ... ")
    console_print(" ")

    atom_type = [int(v) for v in ORG.atomType]
    molecule = int(getattr(ORG, "molecule", 0))
    loops_seed = 0

    for block in _read_poscar_blocks(poscars):
        try:
            species = block["species"]
            if species is None:
                console_print("Seeds: block without element symbols, stop reading seeds")
                break
            symbol_ids = [element_id(sym) for sym in species]
            if any(z == 0 for z in symbol_ids) or any(z not in atom_type for z in symbol_ids):
                console_print(f"Seeds : Elemental types are inconsistent when reading Seeds-{loops_seed + 1}")
                break

            counts = block["counts"]
            num_ions = [0] * len(atom_type)
            for i, z_system in enumerate(atom_type):
                for j, z_read in enumerate(symbol_ids):
                    if z_read == z_system:
                        num_ions[i] = counts[j]

            if molecule == 1:
                std_mol = ORG.STDMOL
                comp = np.zeros((len(std_mol), len(atom_type)))
                for i, std in enumerate(std_mol):
                    for j in std.types:
                        idx = int(j) - 1
                        if 0 <= idx < len(atom_type):
                            comp[i, idx] += 1

                read_counts = _molecule_counts_from_comment(block["comment"])
                num_mols = None
                if (
                    read_counts is not None
                    and len(read_counts) == comp.shape[0]
                    and np.linalg.norm(np.asarray(num_ions, float) - np.asarray(read_counts, float) @ comp) < 1e-3
                ):
                    num_mols = [int(round(v)) for v in read_counts]
                else:
                    solved = np.linalg.lstsq(comp.T, np.asarray(num_ions, float), rcond=None)[0]
                    num_mols = [int(round(v)) for v in solved]
                    if np.linalg.norm(np.asarray(num_ions, float) - np.asarray(num_mols, float) @ comp) > 1e-3:
                        console_print(
                            f"Seeds : Impossible to make combination {num_mols} when reading "
                            f"Seeds-{loops_seed + 1}, so we skip this Seed ..."
                        )
                        continue
                if any(v < 0 for v in num_mols):
                    console_print(f"Seeds : negative molecule count in Seeds-{loops_seed + 1}, skip this Seed ...")
                    continue
            else:
                num_mols = [1]

            coords = np.asarray(block["coords"], dtype=float)
            if block["cartesian"]:
                coords = coords @ np.linalg.inv(block["lattice"])

            # Reorder atoms so they follow ORG.atomType, like MATLAB does.
            new_coords = None
            start = 0
            for z_system in atom_type:
                block_rows = []
                for j, z_read in enumerate(symbol_ids):
                    if z_read == z_system:
                        offset = sum(counts[:j])
                        block_rows.append(coords[offset:offset + counts[j]])
                if block_rows:
                    rows = np.concatenate(block_rows, axis=0)
                    new_coords = rows if new_coords is None else np.concatenate([new_coords, rows], axis=0)
                start += len(block_rows)
            if new_coords is None:
                console_print(f"Seeds : no usable coordinates in Seeds-{loops_seed + 1}, skip this Seed ...")
                continue

            add = len(POP.POPULATION) + 1
            if generation == 1:
                ORG.initialPopSize = int(getattr(ORG, "initialPopSize", 0)) + 1
                add = int(ORG.initialPopSize)

            individual = AttrDict()
            individual.Parents = [AttrDict(parent="", enthalpy=0.0)]
            individual.LATTICE = np.asarray(block["lattice"], dtype=float)
            individual.COORDINATES = new_coords
            individual.numIons = list(num_ions)
            individual.howCome = "  Seeds"
            individual.numMols = list(num_mols)

            index = add - 1
            if index < len(POP.POPULATION):
                POP.POPULATION[index] = individual
            else:
                POP.POPULATION.append(individual)

            if molecule == 1:
                from gpevo.FunctionFolder.USPEX.src.GetPOP_MOL import GetPOP_MOL
                from gpevo.FunctionFolder.AbinitCode.readMOL import readMOL

                types_a, mtype_list, _ = GetPOP_MOL(num_mols)
                individual.typesAList = list(types_a)
                individual.MtypeLIST = list(mtype_list)
                readMOL(index, 0)

            loops_seed += 1
            console_print(f"seed number {loops_seed} has been successfully added")
        except Exception as exc:  # keep the run alive, like MATLAB's try/catch
            console_print(f"Seeds : Meet a problem when reading Seeds-{loops_seed + 1}: {exc}")
            break

    G["POP_STRUC"] = POP
    G["ORG_STRUC"] = ORG

    res_folder = getattr(ORG, "resFolder", None)
    if res_folder and os.path.isdir(res_folder):
        history = os.path.join(res_folder, "Seeds_history")
        with open(history, "a", encoding="utf-8") as out:
            out.write(f"Generation:{generation}\n")
            with open(poscars, "r", encoding="utf-8", errors="replace") as src:
                out.write(src.read())
        os.replace(poscars, generation_file)

    console_print(" ")
    console_print("End of pickup Seeds")
    return loops_seed
