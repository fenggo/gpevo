"""Python port of ``FunctionFolder/sys/anasym_stokes.m``.

The MATLAB version shells out to the ``spacegroup/findsym_new`` binary: it
writes a ``sym.in`` card (fractional coordinates + identity primitive lattice
vectors + requested accuracy), reads the space group out of ``sym.out`` and
also leaves a symmetrized ``symmetrized.cif`` behind.  This port calls spglib
on the same data, so no binary is needed, and falls back to space group 1
(``P 1``) exactly like MATLAB does when findsym fails or "bombed".
"""
from __future__ import annotations

import numpy as np

_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
    "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb",
    "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
]


def _symbol(z):
    z = int(round(z))
    return _ELEMENTS[z - 1] if 1 <= z <= len(_ELEMENTS) else f"Z{z}"


def _lattice_matrix(lat):
    """Accept either the 3x3 lattice (rows are vectors) or [a,b,c,al,be,ga]."""
    lat = np.asarray(lat, dtype=float)
    if lat.shape == (3, 3):
        return lat
    flat = lat.reshape(-1)
    if flat.size != 6:
        raise ValueError("lattice must be 3x3 or the 6 lattice parameters")
    a, b, c, alpha, beta, gamma = (float(v) for v in flat)
    alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])
    cos_a, cos_b, cos_g = np.cos([alpha, beta, gamma])
    sin_g = np.sin(gamma)
    volume = a * b * c * np.sqrt(1 - cos_a**2 - cos_b**2 - cos_g**2 + 2 * cos_a * cos_b * cos_g)
    return np.array(
        [
            [a, 0.0, 0.0],
            [b * cos_g, b * sin_g, 0.0],
            [c * cos_b, c * (cos_a - cos_b * cos_g) / sin_g, volume / (a * b * sin_g)],
        ]
    )


def _cell_parameters(lat):
    """Return [a, b, c, alpha, beta, gamma] in Angstrom and degrees."""
    lat = _lattice_matrix(lat)
    lengths = np.linalg.norm(lat, axis=1)
    alpha = np.degrees(np.arccos(np.dot(lat[1], lat[2]) / (lengths[1] * lengths[2])))
    beta = np.degrees(np.arccos(np.dot(lat[0], lat[2]) / (lengths[0] * lengths[2])))
    gamma = np.degrees(np.arccos(np.dot(lat[0], lat[1]) / (lengths[0] * lengths[1])))
    return list(lengths) + [alpha, beta, gamma]


def _prepared_cell(lat, coord, numIons, atomType):
    """Build the ``(lattice, positions, types)`` cell plus atomic numbers.

    Returns ``None`` when there is nothing to analyse.
    """
    if coord is None or np.size(coord) == 0:
        return None

    counts = [int(v) for v in np.asarray(numIons).reshape(-1)]
    atom_type = [int(v) for v in np.asarray(atomType).reshape(-1)]

    # Drop species with zero atoms, as the MATLAB version does.
    keep = [(n, z) for n, z in zip(counts, atom_type) if n != 0]
    if not keep:
        return None
    counts = [n for n, _ in keep]
    atom_type = [z for _, z in keep]

    positions = np.asarray(coord, dtype=float).reshape(-1, 3)
    n_atoms = sum(counts)
    if positions.shape[0] != n_atoms:
        positions = positions[:n_atoms]
    if positions.shape[0] == 0:
        return None

    types = []
    for type_index, count in enumerate(counts):
        types.extend([type_index] * count)

    return _lattice_matrix(lat), positions, types, atom_type


def anasym_stokes(lat, coord, numIons, atomType, tolerance=0.1):
    """Return the space group number of the relaxed structure.

    ``coord`` are fractional coordinates, ``numIons``/``atomType`` the per-type
    atom counts and atomic numbers, ``tolerance`` the symmetry accuracy in
    Angstrom (``ORG.SGtolerance``).  Returns 1 when the symmetry cannot be
    determined, mirroring the MATLAB fallback to ``P 1``.
    """
    prepared = _prepared_cell(lat, coord, numIons, atomType)
    if prepared is None:
        return 1
    try:
        import spglib
    except ImportError:
        return 1

    cell = prepared[:3]
    symprec = float(tolerance) if tolerance else 0.1
    try:
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    except Exception:
        return 1
    if dataset is None:
        return 1

    number = getattr(dataset, "number", None)
    if number is None and isinstance(dataset, dict):
        number = dataset.get("number")
    if number is None or int(number) <= 0:
        return 1
    return int(number)


def symmetrized_cif(lat, coord, numIons, atomType, tolerance=0.1):
    """Return the CIF body findsym writes next to the space group result.

    The symmetry-unique sites of the detected group are listed together with
    the Hermann-Mauguin symbol and the International Tables number.  When the
    symmetry cannot be determined this prints ``P 1`` with every atom listed,
    which is what the MATLAB fallback does.
    """
    prepared = _prepared_cell(lat, coord, numIons, atomType)
    if prepared is None:
        return ""
    _, positions, types, atom_type = prepared

    a, b, c, alpha, beta, gamma = _cell_parameters(lat)
    header = [
        "_cell_length_a %6.3f" % a,
        "_cell_length_b %6.3f" % b,
        "_cell_length_c %6.3f" % c,
        "_cell_angle_alpha %6.3f" % alpha,
        "_cell_angle_beta  %6.3f" % beta,
        "_cell_angle_gamma %6.3f" % gamma,
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]

    site_lines = []
    space_group_line = '_symmetry_space_group_name_H-M "P 1" '
    number_line = "_symmetry_Int_Tables_number 1"

    listed = list(range(len(types)))
    try:
        import spglib

        dataset = spglib.get_symmetry_dataset(
            prepared[:3], symprec=float(tolerance) if tolerance else 0.1
        )
        if dataset is not None:
            number = getattr(dataset, "number", None)
            symbol = getattr(dataset, "international", "P 1").strip()
            if number:
                # Keep only one representative per symmetry-equivalent orbit,
                # the way findsym lists Wyckoff positions.
                equivalent = np.asarray(dataset.equivalent_atoms).reshape(-1)
                seen = set()
                listed = []
                for index in range(len(types)):
                    key = int(equivalent[index])
                    if key in seen:
                        continue
                    seen.add(key)
                    listed.append(index)
                space_group_line = '_symmetry_space_group_name_H-M "%s" ' % symbol
                number_line = "_symmetry_Int_Tables_number %d" % int(number)
    except Exception:
        listed = list(range(len(types)))

    for index in listed:
        symbol = _symbol(atom_type[types[index]])
        x, y, z = positions[index]
        site_lines.append(
            f"  {symbol:<2s} {symbol:<2s} {x:8.4f} {y:8.4f} {z:8.4f} 1.0000"
        )

    return "\n".join([space_group_line, number_line, ""] + header + site_lines) + "\n"
