"""Check whether an optType-14 target can be represented by a smaller run cell.

The fingerprint cosine distance is size-intensive, so a target with N times as
many molecules as the run IS comparable -- but only if the target's packing is
a pure translational repeat of the smaller cell.  ``spglib.find_primitive``
tells us the smallest periodic unit; if it needs more molecules than
``numMols``, no amount of evolution can reach distance 0 and optType 14 has a
non-zero floor.

Usage::

    gpevo target-primitive -p data_exp/structures_mlp.traj \
        --nmol-target 8 --nmol-run 4 --out POSCAR_primitive.traj
"""
from __future__ import annotations

import numpy as np


def _spacegroup(dataset):
    if dataset is None:
        return None
    try:
        return int(dataset.number)
    except AttributeError:
        return int(dataset["number"])


def _primitive(atoms, symprec):
    """Return (primitive Atoms, spacegroup number) or (None, number)."""
    import spglib
    from ase import Atoms

    cell = np.asarray(atoms.cell.array, dtype=float)
    scaled = np.asarray(atoms.get_scaled_positions(), dtype=float)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    spg_input = (cell, scaled, numbers)
    sg = _spacegroup(spglib.get_symmetry_dataset(spg_input, symprec=symprec))
    prim = spglib.find_primitive(spg_input, symprec=symprec)
    if prim is None:
        return None, sg
    p_cell, p_scaled, p_numbers = prim
    return (
        Atoms(numbers=np.asarray(p_numbers, dtype=int),
              cell=np.asarray(p_cell, dtype=float),
              scaled_positions=np.asarray(p_scaled, dtype=float),
              pbc=True),
        sg,
    )


def _verdict(nmol_target, nmol_run, prim_ratio):
    """Text explaining whether the run cell can express the target packing."""
    if not nmol_target or not nmol_run:
        return "n/a (pass --nmol-target and --nmol-run to check)"
    nprim = max(1, int(round(nmol_target * prim_ratio)))
    if nmol_run < nprim:
        return (f"NO - smallest periodic unit needs {nprim} molecules, but the "
                f"run cell has {nmol_run}. This packing CANNOT be expressed; "
                "optType 14 will have a non-zero distance floor.")
    if nmol_run % nprim:
        return (f"NO - run cell ({nmol_run}) is not an integer multiple of the "
                f"primitive ({nprim}).")
    return (f"YES - {nmol_run}-molecule cell = {nmol_run // nprim}x primitive; "
            "use --out as a reduced target for an exact distance=0 search.")


def target_primitive(path, symprec=0.05, nmol_target=None, nmol_run=None,
                     out=None, quiet=False):
    """Analyse every frame of *path*; print a report and return one row/frame."""
    from ase.io import read

    images = list(read(path, index=":"))
    rows = []
    primitives = []
    for i, atoms in enumerate(images):
        prim, sg = _primitive(atoms, symprec)
        natoms = len(atoms)
        prim_natoms = len(prim) if prim is not None else natoms
        ratio = prim_natoms / natoms if natoms else 1.0
        row = {
            "frame": i,
            "natoms": natoms,
            "volume": float(atoms.get_volume()),
            "spacegroup": sg,
            "primitive_natoms": prim_natoms,
            "primitive_ratio": ratio,
            "reducible": prim_natoms < natoms,
        }
        row["verdict"] = _verdict(nmol_target, nmol_run, ratio)
        rows.append(row)
        primitives.append(prim)

    if out:
        valid = [a for a in primitives if a is not None]
        if not valid:
            raise RuntimeError("no primitive structure could be determined")
        if out.endswith(".traj"):
            from ase.io.trajectory import TrajectoryWriter
            with TrajectoryWriter(out, mode="w") as writer:
                for a in valid:
                    writer.write(a)
        else:
            from ase.io import write
            write(out, valid[0])

    if not quiet:
        print(f"target-primitive: {path}  (symprec={symprec}, {len(images)} frame(s))")
        for row in rows:
            print(f"  frame {row['frame']}: natoms={row['natoms']:5d}  "
                  f"V={row['volume']:9.1f}  SG={str(row['spacegroup']):>4s}  "
                  f"primitive={row['primitive_natoms']:5d} atoms "
                  f"(x{row['primitive_ratio']:.3f})")
            print(f"    {row['verdict']}")
        if out:
            print(f"  wrote primitive structure(s) to {out}")
        if nmol_target and nmol_run and nmol_target != nmol_run:
            print("  NOTE: target and run have different molecule counts. The "
                  "RDF fingerprint is size-intensive but leaves a systematic "
                  "residual (~0.02 measured); targetTolerance defaults to 0.03.")
    return rows


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Check whether an optType-14 target can be reduced to the "
                    "run's cell size.")
    parser.add_argument("-p", "--path", required=True,
                        help="target POSCAR/CIF/.traj")
    parser.add_argument("--symprec", type=float, default=0.05,
                        help="spglib symmetry tolerance")
    parser.add_argument("--nmol-target", type=int, default=None,
                        help="molecules in the target cell")
    parser.add_argument("--nmol-run", type=int, default=None,
                        help="molecules in the run cell (numMols)")
    parser.add_argument("--out", default=None,
                        help="write the primitive structure(s) here")
    args = parser.parse_args(argv)
    target_primitive(path=args.path, symprec=args.symprec,
                     nmol_target=args.nmol_target, nmol_run=args.nmol_run,
                     out=args.out)


if __name__ == "__main__":
    main()
