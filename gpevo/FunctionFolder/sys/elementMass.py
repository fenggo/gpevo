"""Atomic masses and density helper, ported from ``sys/elementMass.m``."""
from __future__ import annotations


_ATOMIC_MASSES = (
    1.00794,
    4.002602,
    6.941,
    9.012182,
    10.811,
    12.0107,
    14.0067,
    15.9994,
    18.9984,
    20.1797,
    22.98977,
    24.305,
    26.98154,
    28.0855,
    30.97376,
    32.065,
    35.453,
    39.948,
    39.0983,
    40.078,
    44.95591,
    47.867,
    50.9415,
    51.9961,
    54.93805,
    55.845,
    58.9332,
    58.6932,
    63.546,
    65.409,
    69.723,
    72.64,
    74.9216,
    78.96,
    79.904,
    83.798,
    86.4678,
    87.62,
    88.90585,
    91.224,
    92.90638,
    95.94,
    98.0,
    101.07,
    102.9055,
    106.42,
    107.8682,
    112.411,
    114.818,
    118.71,
    121.76,
    127.6,
    126.9045,
    131.293,
    132.9055,
    137.327,
    138.9055,
    140.116,
    140.9077,
    144.24,
    145.0,
    150.36,
    151.964,
    157.25,
    158.9253,
    162.5,
    164.9303,
    167.259,
    168.9342,
    173.04,
    174.967,
    178.48,
    180.9479,
    183.84,
    186.207,
    190.23,
    192.217,
    195.078,
    196.9666,
    200.59,
    204.3833,
    207.2,
    208.9804,
    209.0,
    210.0,
    222.0,
    223.0,
    226.0,
    227.0,
    232.0381,
    231.0359,
    238.0289,
    237.0,
    244.0,
    243.0,
    247.0,
    247.0,
    251.0,
    252.0,
    257.0,
    258.0,
    259.0,
    262.0,
    261.0,
    262.0,
    266.0,
    264.0,
    277.0,
    268.0,
    281.0,
    272.0,
)


def element_mass(atomic_number):
    """Return the atomic mass used by USPEX for 1-based ``atomic_number``."""
    idx = int(atomic_number)
    if 1 <= idx <= len(_ATOMIC_MASSES):
        return _ATOMIC_MASSES[idx - 1]
    return None


def calc_density(num_ions, atom_type, volume):
    """Mass density in g/cm^3, matching ``calcDensity.m``.

    ``volume`` is the cell volume in A^3 and may be ``None``/``N/A``.
    """
    if volume is None:
        return None
    try:
        volume = float(volume)
    except (TypeError, ValueError):
        return None
    if volume <= 0:
        return None
    mass = 0.0
    for z, n in zip(atom_type, num_ions):
        m = element_mass(int(z))
        if m is None:
            return None
        mass += m * float(n)
    # calcDensity.m: density = mass / volume / 0.602214129
    return mass / (volume * 1e-24 * 6.02214076e23)
