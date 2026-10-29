"""``gpevo ei-report`` on a synthetic run directory.

No GULP, no SIESTA, no real run: the parsers and the convergence baseline are
exercised on files written by the fixture, and the plots are only checked for
being written (matplotlib is required for those two assertions).
"""
from __future__ import annotations

import math
import os

import pytest

from gpevo.tools import ei_report as report_module

HEADER = ("Gen   ID    Origin   Composition    Enthalpy   Volume  Density   "
          "Fitness   KPOINTS  SYMM  Q_entr A_order S_order\n"
          "                                      (eV)     (A^3)  (g/cm^3)\n")


def _row(gen, cid, density):
    return (f"{gen:3d} {cid:5d}   Random    [ 26 36 30 22]    -1.630  1307.324   "
            f"{density:.3f}     -{density:.3f} [ 1  1  1]   1  0.315  3.465  1.525\n")


@pytest.fixture
def run(tmp_path):
    """Four generations: the density climbs, EI picks the high ones."""
    res = tmp_path / "results1"
    res.mkdir()
    densities = {}
    cid = 0
    for gen in range(1, 5):
        for i in range(8):
            cid += 1
            # later generations hold denser structures, with id 1 of every
            # generation the densest of that generation
            densities[cid] = 1.70 + 0.01 * gen + (0.02 if i == 0 else 0.0)
    with open(res / "Individuals", "w", encoding="utf-8") as fd:
        fd.write(HEADER)
        for gen in range(1, 5):
            for i in range(8):
                cid = (gen - 1) * 8 + i + 1
                fd.write(_row(gen, cid, densities[cid]))

    with open(res / "density_predict.log", "w", encoding="utf-8") as fd:
        fd.write("# Crystal_id Residual Density_mlp Density_rf Density_gp std_den "
                 "Energy std_eng\n")
        for cid, value in densities.items():
            fd.write(f" {cid} 0.004 1.90 1.81 {value:.4f} 0.0170 -28080.9 0.11\n")

    # generations 2..4 each got one label: the densest structure of that generation
    picks = {2: 9, 3: 17, 4: 25}
    with open(tmp_path / "ei.log", "w", encoding="utf-8") as fd:
        for gen, cid in picks.items():
            fd.write(f"gen {gen}:{cid}\n")
    with open(res / "density.log", "w", encoding="utf-8") as fd:
        fd.write("# Crystal_id Density Energy\n")
        for cid in picks.values():
            fd.write(f"  {cid}   {densities[cid]:.6f} -28081.4\n")
    with open(tmp_path / "log", "w", encoding="utf-8") as fd:
        rows = 216
        for gen, cid in picks.items():
            rows += 1
            fd.write("  GP learning curve (gen %d): rows %d -> %d (+1)  "
                     "LOO RMSE density=0.0312 g/cm^3  energy=0.4123 eV\n"
                     % (gen, rows - 1, rows))
    return str(tmp_path), res, densities, picks


def test_parsers_read_the_run(run):
    path, res, densities, picks = run
    rows, gen_ids = report_module.parse_individuals(str(res / "Individuals"))
    assert sorted(gen_ids) == [1, 2, 3, 4]
    assert len(rows) == 32

    pred = report_module.parse_density_predict(str(res / "density_predict.log"))
    assert pred[17]["density_gp"] == pytest.approx(densities[17], abs=1e-4)

    dft = report_module.parse_density_log(str(res / "density.log"))
    assert sorted(dft) == sorted(picks.values())

    ei_log = report_module.parse_ei_log(os.path.join(path, "ei.log"))
    assert ei_log == {gen: [cid] for gen, cid in picks.items()}

    learning = report_module.parse_learning_curve(os.path.join(path, "log"))
    assert sorted(learning) == [2, 3, 4]
    assert learning[3]["added"] == 1
    assert learning[3]["rmse_density"] == pytest.approx(0.0312)


def test_ei_log_keeps_the_last_attempt_of_a_generation(tmp_path):
    path = tmp_path / "ei.log"
    path.write_text("gen 2:11 12\ngen 3:21 22\ngen 2:31 32\n", encoding="utf-8")
    assert report_module.parse_ei_log(str(path)) == {2: [31, 32], 3: [21, 22]}


def test_ei_selection_beats_the_random_baseline(run):
    """EI picked the densest crystal of every generation: it must win."""
    path, res, densities, picks = run
    rows, gen_ids = report_module.parse_individuals(str(res / "Individuals"))
    pred = report_module.parse_density_predict(str(res / "density_predict.log"))
    dft = report_module.parse_density_log(str(res / "density.log"))
    ei_log = report_module.parse_ei_log(os.path.join(path, "ei.log"))

    summary = report_module.build_summary(rows, gen_ids, pred, dft, ei_log)
    ei = report_module.ei_curve(summary, gen_ids, ei_log)
    steps, mean, low, high = report_module.random_baseline(
        gen_ids, {row[1]: row for row in rows}, ei_log, trials=400)

    assert len(ei) == len(picks)
    assert ei[-1] == pytest.approx(max(densities[c] for c in picks.values()))
    assert steps[-1] == len(picks)
    # compared at the same budget: EI is never worse than the average random
    # draw, and ends at or above its 90th percentile
    assert all(e >= m for e, m in zip(ei, mean))
    assert ei[-1] >= high[-1]


def test_evolution_legend_names_the_markers(run):
    """Green rings and red diamonds are only explained if the legend has them.

    EI starts labelling at generation 2, so labelling the first *generation*
    instead of the first *drawn* marker dropped both entries.
    """
    path, res, densities, picks = run
    rows, gen_ids = report_module.parse_individuals(str(res / "Individuals"))
    pred = report_module.parse_density_predict(str(res / "density_predict.log"))
    dft = report_module.parse_density_log(str(res / "density.log"))
    ei_log = report_module.parse_ei_log(os.path.join(path, "ei.log"))
    summary = report_module.build_summary(rows, gen_ids, pred, dft, ei_log)

    fig, ax = report_module._evolution_figure(summary, rows, dft)
    texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any("EI-selected" in t for t in texts), texts
    assert any("DFT density" in t for t in texts), texts
    assert any("best so far" in t for t in texts), texts
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_ei_report_writes_the_three_figures(run, tmp_path):
    path, res, densities, picks = run
    out = str(tmp_path / "fig")
    text = report_module.ei_report(path=path, out=out, quiet=True)

    assert "32 structures" in text
    for suffix in ("_evolution.png", "_convergence.png", "_learning.png"):
        assert os.path.isfile(out + suffix), suffix
        assert os.path.getsize(out + suffix) > 0


def test_convergence_plots_generation_on_the_y_axis(run):
    """The convergence panel was flipped: y = Generation, x = Density."""
    path, res, densities, picks = run
    rows, gen_ids = report_module.parse_individuals(str(res / "Individuals"))
    pred = report_module.parse_density_predict(str(res / "density_predict.log"))
    dft = report_module.parse_density_log(str(res / "density.log"))
    ei_log = report_module.parse_ei_log(os.path.join(path, "ei.log"))
    summary = report_module.build_summary(rows, gen_ids, pred, dft, ei_log)

    series = report_module._ei_series(summary)
    assert [entry["gen"] for entry in series] == [2, 3, 4]
    assert len(series) == 3
    # each pick carries its DFT reference (the densest crystal of its generation)
    assert all(math.isfinite(entry["dft"]) for entry in series)

    fig, ax = report_module._convergence_figure(series)
    import matplotlib.pyplot as plt
    from matplotlib.container import ErrorbarContainer
    assert ax.get_xlabel() == "Generation"
    assert "Density" in ax.get_ylabel()
    # every pick carries the GP uncertainty, and it is drawn as an error bar
    assert all(math.isfinite(entry["sigma"]) for entry in series)
    assert any(isinstance(c, ErrorbarContainer) for c in ax.containers)
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("1.96" in t for t in texts), texts
    plt.close(fig)


def test_convergence_without_sigma_still_plots(run):
    """A run whose log predates the append-only fix: no σ, but no crash."""
    path, res, densities, picks = run
    rows, gen_ids = report_module.parse_individuals(str(res / "Individuals"))
    # only the last generation was kept, as a rewritten density_predict.log had
    (res / "density_predict.log").write_text(
        "# Crystal_id Residual Density_mlp Density_rf Density_gp std_den Energy std_eng\n"
        f" 25 0.004 1.90 1.81 {densities[25]:.4f} 0.0170 -28080.9 0.11\n",
        encoding="utf-8")
    pred = report_module.parse_density_predict(str(res / "density_predict.log"))
    dft = report_module.parse_density_log(str(res / "density.log"))
    ei_log = report_module.parse_ei_log(os.path.join(path, "ei.log"))
    summary = report_module.build_summary(rows, gen_ids, pred, dft, ei_log)

    series = report_module._ei_series(summary)
    assert len(series) == 3
    assert sum(1 for e in series if math.isfinite(e["sigma"])) == 1

    import matplotlib.pyplot as plt
    fig, ax = report_module._convergence_figure(series)
    plt.close(fig)


def test_ei_report_on_a_run_without_any_label(run, tmp_path):
    """A run whose gates blocked everything: the plot says so instead of breaking."""
    path, res, _densities, _picks = run
    os.remove(os.path.join(path, "ei.log"))
    text = report_module.ei_report(path=path, out=str(tmp_path / "fig"), quiet=True)
    assert "0 EI-selected" in text


def test_cli_subcommand_is_wired(tmp_path, capsys):
    from gpevo.cli import main

    # -n prints the report without needing matplotlib; an empty directory must
    # say so instead of raising
    main(["ei-report", "-p", str(tmp_path), "-n"])
    assert "No Individuals table" in capsys.readouterr().out
