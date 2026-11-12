"""``gpevo dedup``: duplicate removal from an ASE trajectory."""
from __future__ import annotations

import os

import pytest

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write


def _atoms(cell, scaled, energy=None, symbols="NaCl"):
    atoms = Atoms(symbols=symbols, cell=cell, scaled_positions=scaled, pbc=True)
    if energy is not None:
        atoms.calc = SinglePointCalculator(atoms, energy=energy)
    return atoms


def _fixture():
    return [
        _atoms([4, 4, 4], [[0, 0, 0], [0.5, 0.5, 0.5]], -1.0),
        _atoms([4, 4, 4], [[0, 0, 0], [0.5, 0.5, 0.5]], -1.0),      # identical
        _atoms([5, 5, 5], [[0, 0, 0], [0.25, 0.3, 0.35]], -1.0),    # different packing
        _atoms([4, 4, 4], [[0, 0, 0], [0.5, 0.5, 0.5]], -2.0),      # same packing, dE=1 eV
        _atoms([4, 4, 4], [[0.001, 0, 0], [0.501, 0.5, 0.5]], -1.0),  # near-identical
    ]


def _write(tmp_path, images):
    path = tmp_path / "structures.traj"
    write(str(path), images)
    return str(path)


def test_packing_and_energy_both_required(tmp_path):
    from gpevo.tools.dedup import dedup
    path = _write(tmp_path, _fixture())
    out = str(tmp_path / "out.traj")
    result = dedup(path, output=out, quiet=True)

    # {0,1,4} are the same packing at the same energy; 2 is a different
    # packing; 3 is the same packing but 1 eV away, so it is NOT degenerate.
    assert result["n_total"] == 5
    assert result["groups"] == [([0, 1, 4], 0)]
    assert result["redundant"] == [1, 4]
    assert result["keep"] == [0, 2, 3]
    assert len(read(out, index=":")) == 3


def test_keep_id_wins_over_lowest_energy(tmp_path):
    from gpevo.tools.dedup import dedup
    path = _write(tmp_path, _fixture())
    result = dedup(path, output=str(tmp_path / "out.traj"),
                   keep_id=[1], quiet=True)
    assert result["groups"] == [([0, 1, 4], 1)]
    assert result["keep"] == [1, 2, 3]


def test_dry_run_writes_no_trajectory_but_still_reports(tmp_path):
    from gpevo.tools.dedup import dedup
    path = _write(tmp_path, _fixture())
    out = str(tmp_path / "out.traj")
    report = str(tmp_path / "report.csv")
    result = dedup(path, output=out, report=report, dry_run=True, quiet=True)

    assert result["output"] is None
    assert not os.path.exists(out)
    assert os.path.exists(report)
    lines = open(report, encoding="utf-8").read().splitlines()
    assert lines[0] == "duplicate_id,representative_id,cluster_size,cluster_ids"
    assert len(lines) == 3            # header + two redundant ids


def test_default_output_name(tmp_path):
    from gpevo.tools.dedup import dedup
    path = _write(tmp_path, _fixture())
    result = dedup(path, quiet=True)
    assert result["output"] == os.path.splitext(path)[0] + "_dedup.traj"
    assert os.path.exists(result["output"])


def test_missing_energies_fall_back_to_packing_only(tmp_path, capsys):
    from gpevo.tools.dedup import dedup
    images = [
        _atoms([4, 4, 4], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        _atoms([4, 4, 4], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        _atoms([5, 5, 5], [[0, 0, 0], [0.25, 0.3, 0.35]]),
    ]
    path = _write(tmp_path, images)
    result = dedup(path, output=str(tmp_path / "out.traj"))
    assert "skipping the energy-degeneracy criterion" in capsys.readouterr().out
    assert result["keep"] == [0, 2]


def test_different_atom_counts_are_never_compared(tmp_path):
    from gpevo.tools.dedup import dedup
    four = _atoms([4, 4, 8], [[0, 0, 0], [0.5, 0.5, 0.25],
                              [0, 0, 0.5], [0.5, 0.5, 0.75]],
                  -1.0, symbols="NaClNaCl")
    path = _write(tmp_path, _fixture() + [four])
    result = dedup(path, output=str(tmp_path / "out.traj"), quiet=True)
    assert len(result["keep"]) == 4
    assert all(len(cluster) <= 3 for cluster, _ in result["groups"])


def test_empty_trajectory_raises(tmp_path):
    from gpevo.tools.dedup import dedup
    path = str(tmp_path / "empty.traj")
    write(path, [])
    with pytest.raises(RuntimeError, match="no structures"):
        dedup(path, quiet=True)


def test_cli_subcommand_registered(tmp_path, capsys):
    from gpevo.cli import main
    path = _write(tmp_path, _fixture())
    out = str(tmp_path / "out.traj")
    main(["dedup", path, "-o", out, "--quiet"])
    assert os.path.exists(out)
    assert len(read(out, index=":")) == 3
