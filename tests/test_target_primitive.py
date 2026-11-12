"""``gpevo target-primitive``: can an optType-14 target be reduced to the run cell?"""
from __future__ import annotations

import numpy as np
import pytest

from gpevo.tools.target_primitive import _verdict, target_primitive


def _write(path, atoms):
    from ase.io import write
    write(str(path), atoms, format="vasp", direct=True)


def _reducible():
    """A 2x1x1 supercell: a (0.5, 0, 0) translation maps it onto itself."""
    from ase import Atoms
    base = Atoms(symbols=["C", "C"], cell=np.eye(3) * 4.0,
                 scaled_positions=[[0.0, 0.0, 0.0], [0.3, 0.4, 0.5]],
                 pbc=True)
    return base * (2, 1, 1)


def _irreducible():
    """Four different species at generic sites: no translation survives."""
    from ase import Atoms
    return Atoms(symbols=["C", "N", "O", "H"],
                 cell=np.diag([8.0, 4.0, 4.0]),
                 scaled_positions=[[0.0, 0.0, 0.0],
                                   [0.15, 0.4, 0.5],
                                   [0.5, 0.0, 0.0],
                                   [0.75, 0.26, 0.25]],
                 pbc=True)


def test_verdict_pure_function():
    assert _verdict(None, 4, 0.5).startswith("n/a")
    assert _verdict(4, 2, 0.5).startswith("YES")
    assert _verdict(4, 4, 1.0).startswith("YES")
    assert _verdict(4, 2, 1.0).startswith("NO")     # needs 4, run has 2
    assert _verdict(4, 3, 0.5).startswith("NO")     # 3 is not 2x primitive


def test_reducible_target_can_be_expressed_by_smaller_cell(tmp_path):
    path = tmp_path / "POSCAR"
    _write(path, _reducible())
    rows = target_primitive(str(path), symprec=0.05,
                            nmol_target=4, nmol_run=2, quiet=True)
    assert len(rows) == 1
    assert rows[0]["natoms"] == 4
    assert rows[0]["primitive_natoms"] == 2
    assert rows[0]["primitive_ratio"] == pytest.approx(0.5)
    assert rows[0]["reducible"] is True
    assert rows[0]["verdict"].startswith("YES")


def test_irreducible_target_cannot_be_expressed_by_smaller_cell(tmp_path):
    path = tmp_path / "POSCAR"
    _write(path, _irreducible())
    rows = target_primitive(str(path), symprec=0.05,
                            nmol_target=4, nmol_run=2, quiet=True)
    assert rows[0]["primitive_natoms"] == 4
    assert rows[0]["reducible"] is False
    assert rows[0]["verdict"].startswith("NO")
    assert "distance floor" in rows[0]["verdict"]


def test_out_writes_the_primitive_structure(tmp_path):
    from ase.io import read
    path = tmp_path / "POSCAR"
    out = tmp_path / "prim.traj"
    _write(path, _reducible())
    target_primitive(str(path), symprec=0.05, out=str(out), quiet=True)
    images = read(str(out), index=":")
    assert len(images) == 1 and len(images[0]) == 2


def test_output_mentions_residual_when_molecule_counts_differ(tmp_path, capsys):
    path = tmp_path / "POSCAR"
    _write(path, _reducible())
    target_primitive(str(path), symprec=0.05, nmol_target=4, nmol_run=2)
    out = capsys.readouterr().out
    assert "target-primitive:" in out
    assert "YES" in out
    assert "targetTolerance defaults to 0.03" in out


def test_cli_subcommand_is_registered(tmp_path, capsys):
    from gpevo.cli import main
    path = tmp_path / "POSCAR"
    _write(path, _reducible())
    main(["target-primitive", "-p", str(path),
          "--nmol-target", "4", "--nmol-run", "2"])
    assert "YES" in capsys.readouterr().out
