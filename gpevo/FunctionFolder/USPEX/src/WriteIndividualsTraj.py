"""Append every finished structure to ``resFolder/Individuals.traj``.

``gatheredPOSCARS`` (and ``Individuals``) are plain text, so reading a run back
for analysis needs a parser.  ``Individuals.traj`` is the ASE-readable twin of
``gatheredPOSCARS``: one frame per structure, written from memory at the same
moment the POSCAR block is appended (``WriteIndividualOutput_310``), so the two
files can never drift apart.

Frame <-> row correspondence
---------------------------
``uspexkit`` indexes the trajectory by the global structure id
(``images[id - 1]`` in ``calc``/``pred``), and ids are handed out in completion
order (``POP.bodyCount`` in ReadJobs_310) -- exactly the order frames are
appended here, so frame *i* always is the structure whose Individuals ID is
*i + 1*.

The ``Individuals`` table itself lists a generation's structures in
``POP.POPULATION`` order, so the frame sequence equals the file order whenever
structures finish in that order -- always true for ``numParallelCalcs = 1``
(the common single-CalcFold setup) and the order used by ``gatheredPOSCARS``
in any case.

Frame contents
--------------
``atoms.info`` carries the Individuals columns needed to cross-reference a
frame without parsing the text file (Number / generation / Origin / SYMMETRY /
Enthalpy / Density), plus the GP predictions when the structure has them.  The
enthalpy is also attached as a single-point energy, so ``get_potential_energy()``
works directly on a frame.

Like ``gp_plot`` this never raises: a missing ASE or an unwritable file only
warns, the run itself must not depend on the convenience file.
"""
from __future__ import annotations

import math
import os

import numpy as np

from gpevo.log_utils import console_print
from gpevo.FunctionFolder.sys.Write_POSCAR import _symbol

# Keys of the gp()/pred() result dict that are reported per structure; they are
# stored under the same names in atoms.info.
_GP_FIELDS = ("energy_pred", "uncertainty_eng", "density_gp", "uncertainty")


def _num(value):
    """float(value) when it is a finite number, else None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _gp_info(s):
    """GP prediction fields of a structure, from its dict or its attributes."""
    values = {}
    gp = getattr(s, "gp", None)
    if isinstance(gp, dict):
        values.update({k: _num(gp.get(k)) for k in _GP_FIELDS})
    for key, attr in (("energy_pred", "energy_gp"), ("uncertainty_eng", "uq_eng"),
                      ("density_gp", "density_gp"), ("uncertainty", "uq")):
        if values.get(key) is None:
            values[key] = _num(getattr(s, attr, None))
    return {k: v for k, v in values.items() if v is not None}


def _symbols(atom_type, num_ions):
    """Element symbols, in the same order Write_POSCAR writes them."""
    out = []
    for i, n in enumerate(num_ions):
        count = int(n)
        if count <= 0:
            continue
        symbol = _symbol(atom_type[i]) if i < len(atom_type) else "X"
        out.extend([symbol] * count)
    return out


def _atoms(s, atom_type):
    """ASE Atoms of the relaxed structure (fractional coordinates)."""
    from ase import Atoms

    num_ions = [int(n) for n in np.asarray(s.numIons, dtype=float).reshape(-1)]
    lattice = np.asarray(s.LATTICE, dtype=float).reshape(3, 3)
    coor = np.asarray(s.COORDINATES, dtype=float).reshape(-1, 3)
    symbols = _symbols(atom_type, num_ions)
    if not symbols or len(symbols) != len(coor):
        raise ValueError(f"numIons/COORDINATES mismatch: {len(symbols)} symbols "
                         f"vs {len(coor)} coordinates")
    if not np.all(np.isfinite(lattice)) or not np.all(np.isfinite(coor)):
        raise ValueError("non-finite lattice or coordinates")
    return Atoms(symbols=symbols, cell=lattice, scaled_positions=coor, pbc=True)


def append_individuals_traj(res_folder, s, atom_type, generation=0):
    """Append one structure to ``resFolder/Individuals.traj``.

    Returns True when a frame was written.  Any problem (no ASE, bad geometry,
    unwritable file) is reported on the terminal and swallowed -- the caller is
    in the middle of writing the run's bookkeeping files.
    """
    path = os.path.join(res_folder, "Individuals.traj")
    try:
        atoms = _atoms(s, atom_type)
    except Exception as exc:
        console_print(f"  Warning: Individuals.traj skipped structure "
                      f"{getattr(s, 'Number', '?')}: {exc}")
        return False

    number = getattr(s, "Number", None)
    try:
        number = int(number)
    except (TypeError, ValueError):
        number = None

    info = {"generation": int(generation)}
    if number:
        info["Number"] = number
    for key, attr in (("howCome", "howCome"), ("symg", "symg")):
        value = getattr(s, attr, None)
        if value is not None and str(value).strip() != "":
            info[key] = value
    enthalpy = None
    try:
        enthalpy = _num(np.asarray(s.Enthalpies, dtype=float).reshape(-1)[-1])
    except (AttributeError, IndexError, TypeError, ValueError):
        enthalpy = None
    if enthalpy is not None:
        info["enthalpy"] = enthalpy
    try:
        from gpevo.FunctionFolder.sys.elementMass import calc_density

        volume = float(np.linalg.det(np.asarray(s.LATTICE, dtype=float).reshape(3, 3)))
        density = _num(calc_density(list(s.numIons), atom_type, volume))
    except Exception:
        density = None
    if density is not None:
        info["density"] = density
    info.update(_gp_info(s))
    atoms.info.update(info)

    if enthalpy is not None:
        try:
            from ase.calculators.singlepoint import SinglePointCalculator

            atoms.calc = SinglePointCalculator(atoms, energy=enthalpy)
        except Exception:
            pass                            # info alone is still useful

    try:
        from ase.io.trajectory import TrajectoryWriter

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # mode='a' keeps the frame count in the file header correct across the
        # separate open/close cycles of per-structure appends; ase.io.write with
        # append=True does not (it leaves the count at 0 and the file unreadable).
        writer = TrajectoryWriter(path, mode="a")
        try:
            writer.write(atoms)
        finally:
            writer.close()
    except ImportError:
        console_print("  Warning: ASE is not installed, Individuals.traj not written")
        return False
    except Exception as exc:
        console_print(f"  Warning: cannot append to Individuals.traj: {exc}")
        return False
    return True
