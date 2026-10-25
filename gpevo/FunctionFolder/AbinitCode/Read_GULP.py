"""Python equivalent of Read_GULP.m."""
from __future__ import annotations

import os

from gpevo.FunctionFolder.AbinitCode.Read_GULP_Structure import Read_GULP_Structure


def Read_GULP(flag, ID=None):
    if flag == -1:
        try:
            with open("output", errors="replace") as fp:
                text = fp.read()
        except OSError:
            text = ""
        return 1 if "Finished" in text else 0
    if flag == 0:
        if _get_stuff_energy("output") is None:
            return 0
        if not os.path.exists("optimized.structure"):
            return 0
        try:
            with open("optimized.structure", errors="replace") as fp:
                structure_text = fp.read()
        except OSError:
            return 0
        return 1 if "dump" in structure_text else 0
    if flag == 1:
        energy = _get_stuff_energy("output")
        if energy is None:
            return 100000.0
        return energy
    if flag == 2:
        # Pressure not used in the 310 minimal path.
        return 0.0
    raise NotImplementedError(f"Read_GULP flag {flag} is not ported")


def _get_stuff_energy(filename):
    if not os.path.exists(filename):
        return None

    last_energy_hit = None
    last_total_lattice_ev = None
    last_final_energy_ev = None

    with open(filename, errors="replace") as fp:
        for line in fp:
            lower_line = line.lower()
            if "energy" not in lower_line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            token = parts[4]
            try:
                value = float(token)
            except ValueError:
                continue

            # Keeps the tail-1 ``getStuff output Energy 5`` fallback for GULP
            # outputs that do not print the eV/kJ totals in the current order.
            last_energy_hit = value

            if "total lattice energy" in lower_line and "ev" in lower_line:
                last_total_lattice_ev = value
            elif "final energy" in lower_line and "ev" in lower_line:
                last_final_energy_ev = value

    return last_total_lattice_ev if last_total_lattice_ev is not None else (
        last_final_energy_ev if last_final_energy_ev is not None else last_energy_hit
    )
