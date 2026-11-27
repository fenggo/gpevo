"""Tests for the molecular packing coefficient tool."""
from __future__ import annotations

import csv

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from gpevo.tools.packing_coefficient import (
    packing, packing_coefficient, packing_coefficients, write_packing_report,
)


def _cell(vectors, symbols, scaled, grid=20):
    return Atoms(symbols=symbols, cell=vectors, scaled_positions=scaled, pbc=True)


def test_single_atom_cell_pc_equals_sphere_volume_over_cell():
    # A 10 A cubic cell, one H atom (vdW radius 1.2 A).  No overlap with its
    # own periodic images (2*1.2 < 10), so PC = 4/3*pi*r^3 / V_cell.
    atoms = _cell([10, 10, 10], "H", [[0.5, 0.5, 0.5]])
    pc, v_vdw, v_cell = packing_coefficient(atoms, grid=24)
    expected = (4 / 3 * np.pi * 1.2 ** 3) / 1000
    assert v_cell == pytest.approx(1000.0)
    assert pc == pytest.approx(expected, abs=0.002)
    assert v_vdw == pytest.approx(4 / 3 * np.pi * 1.2 ** 3, abs=2.0)


def test_overlapping_atoms_counted_once():
    # Two atoms at the same site: the occupied region is ONE sphere, not two.
    single = _cell([10, 10, 10], "H", [[0.5, 0.5, 0.5]])
    double = _cell([10, 10, 10], "HH", [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    pc1, _, _ = packing_coefficient(single, grid=24)
    pc2, _, _ = packing_coefficient(double, grid=24)
    assert pc2 == pytest.approx(pc1, abs=0.003)


def test_pc_is_periodic_and_below_one():
    # Dense cell of overlapping spheres across the boundary: PC must stay < 1
    # and be independent of where the cell origin falls.  Translation shifts
    # the grid phase, so allow the ~1% sampling difference it causes.
    atoms = _cell([4, 4, 4], "CCCC",
                  [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
    shifted = atoms.copy()
    shifted.translate([2.0, 1.0, 3.0])
    pc, _, _ = packing_coefficient(atoms, grid=40)
    pc_shift, _, _ = packing_coefficient(shifted, grid=40)
    assert 0 < pc < 1.0
    assert pc_shift == pytest.approx(pc, abs=0.01)


def test_grid_convergence(tmp_path):
    atoms = _cell([8, 9, 10], "CHCH",
                  [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.6, 0.6, 0.6], [0.7, 0.7, 0.7]])
    pc_lo, _, _ = packing_coefficient(atoms, grid=20)
    pc_hi, _, _ = packing_coefficient(atoms, grid=40)
    assert pc_lo == pytest.approx(pc_hi, abs=0.01)


def test_rdkit_radius_table_matches_hardcoded_values():
    # RDKit GetRvdw: N 1.6, O 1.55, Cl 1.8 vs ASE 1.55/1.52/1.75.
    from gpevo.tools.packing_coefficient import _RDKIT_RVDW
    assert _RDKIT_RVDW[7] == pytest.approx(1.6)   # N
    assert _RDKIT_RVDW[8] == pytest.approx(1.55)  # O
    assert _RDKIT_RVDW[17] == pytest.approx(1.8)  # Cl
    assert len(_RDKIT_RVDW) == 119               # index 0 fallback + Z=1..118


def test_radii_table_changes_volume_for_oxygen():
    # Oxygen: RDKit radius 1.55 > ASE 1.52, so RDKit table gives a larger vdW volume.
    atoms = _cell([10, 10, 10], "O", [[0.5, 0.5, 0.5]])
    _, v_ase, _ = packing_coefficient(atoms, grid=60, radii="ase")
    _, v_rd, _ = packing_coefficient(atoms, grid=60, radii="rdkit")
    assert v_rd > v_ase
    assert v_rd == pytest.approx(4 / 3 * np.pi * 1.55 ** 3, abs=1.5)


def test_unknown_radius_table_raises():
    atoms = _cell([10, 10, 10], "H", [[0.5, 0.5, 0.5]])
    with pytest.raises(ValueError, match="unknown radii table"):
        packing_coefficient(atoms, grid=10, radii="bad")


def test_non_periodic_cell_raises():
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[0, 0, 0], pbc=False)
    with pytest.raises(ValueError, match="periodic 3-D cell"):
        packing_coefficient(atoms, grid=10)


def test_traj_report_and_csv(tmp_path):
    traj = tmp_path / "frames.traj"
    frames = [
        _cell([10, 10, 10], "H", [[0.5, 0.5, 0.5]]),
        _cell([8, 8, 8], "C", [[0.5, 0.5, 0.5]]),
    ]
    write(traj, frames)

    rows = list(packing_coefficients(str(traj), grid=20))
    assert [r[0] for r in rows] == [0, 1]
    assert all(0 < r[1] < 1 for r in rows)

    out = tmp_path / "pc.csv"
    write_packing_report(str(traj), str(out), grid=20)
    with open(out, newline="", encoding="utf-8") as fp:
        reader = list(csv.reader(fp))
    assert reader[0] == ["frame", "packing_coefficient", "v_vdw_A3",
                         "v_cell_A3", "enthalpy_eV"]
    assert len(reader) == 3


def test_packing_cli_and_parser():
    from gpevo.cli import _build_parser

    sub_action = next(a for a in _build_parser()._actions
                      if isinstance(getattr(a, "choices", None), dict))
    assert "packing" in sub_action.choices
    parsed = vars(_build_parser().parse_args(["packing", "x.traj", "--grid", "30"]))
    assert parsed["grid"] == 30
    assert parsed["traj"] == "x.traj"
    assert parsed["radii"] == "ase"
    parsed_rd = vars(_build_parser().parse_args(
        ["packing", "x.traj", "--radii", "rdkit"]))
    assert parsed_rd["radii"] == "rdkit"


def test_packing_cli_dispatch(monkeypatch, tmp_path):
    from gpevo.cli import main
    import gpevo.tools.packing_coefficient as pc_mod

    calls = []
    monkeypatch.setattr(pc_mod, "packing", lambda **kw: calls.append(kw))
    main(["packing", "x.traj", "-o", "pc.csv", "--grid", "30", "--quiet"])
    assert calls == [{"traj": "x.traj", "output": "pc.csv", "grid": 30,
                      "first": 0, "last": None, "quiet": True, "radii": "ase"}]
