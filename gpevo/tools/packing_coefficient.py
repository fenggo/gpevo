"""Molecular packing coefficient (PC) of a crystal from an ASE trajectory.

PC = V_vdW / V_cell, where V_vdW is the volume actually occupied by the
van der Waals spheres of ALL atoms (periodic images included).  Overlapping
spheres -- both within a molecule and between neighbouring molecules -- are
counted ONCE, so the result stays physical (PC < 1) and needs no molecule
grouping / bond detection.

Why the union and not a sum of atomic spheres: summing ``4/3*pi*r**3`` over
atoms ignores overlaps and gives PC > 1 on real molecular crystals (e.g. 1.40
on ct22), which is impossible.  The union of the same spheres gives ~0.74.

pymatgen has no ready-made PC module; its useful pieces are
``Element.van_der_waals_radius`` and ``Structure.volume``.  Radii come from
ASE's ``vdw_radii`` table (bundled with ase, already a gpevo dependency) so the
tool works inside the torch-free offline package without new dependencies.
Passing ``radii="rdkit"`` switches to RDKit's ``GetRvdw`` table, hard-coded
below (zero rdkit dependency): it matches ``AllChem.ComputeMolVolume`` and
makes volumes comparable to RDKit to within grid resolution (~0.5%).

Algorithm
---------
A regular grid of ``grid**3`` points is laid over the unit cell in fractional
coordinates.  For each point, the minimum-image displacement to every atom is
computed (fractional wrapping, valid for arbitrary triclinic cells), and the
point is marked occupied if it lies inside any vdW sphere.  PC is the occupied
fraction.  The grid is chunked to bound memory.

Usage::

    gpevo packing structures.traj
    gpevo packing structures.traj -o pc.csv --grid 60
"""
from __future__ import annotations

import csv
import os

import numpy as np

from ase.data import atomic_numbers, vdw_radii
from ase.io import read as ase_read

# RDKit's vdW radii (Angstrom), indexed by atomic number 1..118, copied from
# rdkit.Chem.GetPeriodicTable().GetRvdw(z) (RDKit 2026.09).  Hard-coded so the
# offline, torch-free package needs no rdkit install to reproduce RDKit's
# molecular volumes.  Index 0 is a safe fallback and never used.
_RDKIT_RVDW = np.array([
    0.0,
    1.2, 1.4, 2.2, 1.9, 1.8, 1.7, 1.6, 1.55,
    1.5, 1.54, 2.4, 2.2, 2.1, 2.1, 1.95, 1.8,
    1.8, 1.88, 2.8, 2.4, 2.3, 2.15, 2.05, 2.05,
    2.05, 2.05, 2, 2, 2, 2.1, 2.1, 2.1,
    2.05, 1.9, 1.9, 2.02, 2.9, 2.55, 2.4, 2.3,
    2.15, 2.1, 2.05, 2.05, 2, 2.05, 2.1, 2.2,
    2.2, 2.25, 2.2, 2.1, 2.1, 2.16, 3, 2.7,
    2.5, 2.48, 2.47, 2.45, 2.43, 2.42, 2.4, 2.38,
    2.37, 2.35, 2.33, 2.32, 2.3, 2.28, 2.27, 2.25,
    2.2, 2.1, 2.05, 2, 2, 2.05, 2.1, 2.05,
    2.2, 2.3, 2.3, 2, 2, 2, 2, 2,
    2, 2.4, 2, 2.3, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2,
], dtype=float)

_RADII_TABLES = {"ase", "rdkit"}


def _vdw_radii(atoms, radii="ase"):
    """vdW radii (Angstrom) for each atom; ``radii`` selects ``ase``/``rdkit``."""
    table = _RDKIT_RVDW if radii == "rdkit" else vdw_radii
    out = []
    for symbol in atoms.get_chemical_symbols():
        z = atomic_numbers[symbol]
        r = table[z] if z < len(table) else 0.0
        out.append(float(r) if np.isfinite(r) and r > 0 else 1.70)
    return np.asarray(out, dtype=float)


def packing_coefficient(atoms, grid=60, chunk=5000, radii="ase"):
    """Return ``(pc, v_vdw, v_cell)`` for one ASE ``Atoms`` object.

    ``pc`` is the packing coefficient, ``v_vdw`` the occupied (vdW union)
    volume in Angstrom^3, ``v_cell`` the cell volume.  Higher ``grid`` gives
    finer sampling; the occupied fraction is converged to ~1e-3 by grid=60
    for typical cells.  ``radii`` is ``"ase"`` (default) or ``"rdkit"``.
    """
    if radii not in _RADII_TABLES:
        raise ValueError(f"unknown radii table: {radii!r} (use 'ase' or 'rdkit')")
    cell = np.asarray(atoms.cell, dtype=float)
    v_cell = float(abs(np.linalg.det(cell)))
    if v_cell <= 0:
        raise ValueError("packing coefficient requires a periodic 3-D cell")
    inv = np.linalg.inv(cell)
    pos = np.asarray(atoms.get_positions(), dtype=float)
    sphere_radii = _vdw_radii(atoms, radii=radii)

    n = int(grid)
    if n < 4:
        raise ValueError("grid must be >= 4")
    frac = (np.indices((n, n, n)).reshape(3, -1).T + 0.5) / n
    cart = frac @ cell

    occupied = 0
    for start in range(0, len(cart), chunk):
        block = cart[start:start + chunk]
        delta = block[:, None, :] - pos[None, :, :]
        dfrac = delta @ inv
        dfrac -= np.round(dfrac)                 # minimum image, any cell shape
        dcart = dfrac @ cell
        d2 = np.einsum("ijk,ijk->ij", dcart, dcart)
        inside = np.any(d2 <= sphere_radii[None, :] ** 2, axis=1)
        occupied += int(np.count_nonzero(inside))

    v_vdw = v_cell * occupied / len(cart)
    return v_vdw / v_cell, v_vdw, v_cell


def packing_coefficients(traj, grid=60, first=0, last=None, radii="ase"):
    """Yield ``(index, pc, v_vdw, v_cell, enthalpy)`` for frames of *traj*.

    ``index`` is the 0-based frame number; ``enthalpy`` is taken from
    ``atoms.info`` / single-point calculator when present, else NaN.
    """
    images = ase_read(traj, index=":")
    if last is None:
        last = len(images)
    for i in range(first, min(last, len(images))):
        atoms = images[i]
        pc, v_vdw, v_cell = packing_coefficient(atoms, grid=grid, radii=radii)
        enthalpy = float(atoms.info.get("enthalpy", float("nan")))
        if atoms.calc is not None:
            try:
                enthalpy = float(atoms.get_potential_energy())
            except Exception:
                pass
        yield i, pc, v_vdw, v_cell, enthalpy


def write_packing_report(traj, output, grid=60, first=0, last=None, radii="ase"):
    """Compute PC for the selected frames and write a CSV report."""
    rows = list(packing_coefficients(traj, grid=grid, first=first, last=last,
                                     radii=radii))
    parent = os.path.dirname(os.path.abspath(output))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["frame", "packing_coefficient", "v_vdw_A3",
                         "v_cell_A3", "enthalpy_eV"])
        for i, pc, v_vdw, v_cell, enthalpy in rows:
            writer.writerow([i, f"{pc:.6f}", f"{v_vdw:.4f}",
                             f"{v_cell:.4f}", f"{enthalpy:.6f}"])
    return rows


def packing(traj, output=None, grid=60, first=0, last=None, quiet=False,
            radii="ase"):
    """CLI entry point: print a PC table and optionally write a CSV report."""
    rows = list(packing_coefficients(traj, grid=grid, first=first, last=last,
                                     radii=radii))
    if not quiet:
        print(f"Packing coefficient ({len(rows)} frame(s), grid={grid}^3, "
              f"radii={radii})")
        print(f"{'frame':>7} {'PC':>8} {'V_vdw(A^3)':>12} {'V_cell(A^3)':>12} {' enthalpy':>12}")
        for i, pc, v_vdw, v_cell, enthalpy in rows:
            enth = f"{enthalpy:12.4f}" if np.isfinite(enthalpy) else "         n/a"
            print(f"{i:7d} {pc:8.4f} {v_vdw:12.2f} {v_cell:12.2f} {enth}")
    if output:
        parent = os.path.dirname(os.path.abspath(output))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(output, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["frame", "packing_coefficient", "v_vdw_A3",
                             "v_cell_A3", "enthalpy_eV"])
            for i, pc, v_vdw, v_cell, enthalpy in rows:
                writer.writerow([i, f"{pc:.6f}", f"{v_vdw:.4f}",
                                 f"{v_cell:.4f}", f"{enthalpy:.6f}"])
        if not quiet:
            print(f"Report written to {output}")
    return rows
