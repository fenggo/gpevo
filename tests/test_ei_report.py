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

    with open(res / "OUTPUT.txt", "w", encoding="utf-8") as fd:
        header = ("ID  Origin     Composition Enthalpy(eV) Volume(A^3) "
                  "KPOINTS SYMMETRY Enthalpy(GP)        UC Density(GP)        "
                  "UC  Residual\n")
        for gen in range(1, 5):
            fd.write(header)
            for i in range(8):
                cid = (gen - 1) * 8 + i + 1
                fd.write(f" {cid:4d}   Random      [ 26 36 30 22] -1129.659 "
                         f"1372.287  [1 1 1]    1     -28080.821    0.1951  "
                         f"     {densities[cid]:.3f}    0.0333    0.0031\n")
            fd.write("Local optimization finished\n\n")

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


def test_gp_predictions_come_from_output_txt(run, tmp_path):
    path, res, densities, _picks = run
    pred = report_module.parse_output_gp(str(res / "OUTPUT.txt"))
    assert len(pred) == 32
    assert pred[25]["density_gp"] == pytest.approx(densities[25], abs=1e-3)
    assert pred[25]["std"] == pytest.approx(0.0333 / 1.96)
    assert pred[25]["residual"] == pytest.approx(0.0031)

    # the old log is no longer the source: deleting it changes nothing
    os.remove(str(res / "density_predict.log"))
    text = report_module.ei_report(path=path, out=str(tmp_path / "fig"), quiet=True)
    assert "32 GP predictions" in text


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
    assert "Density of all validate structures (MLP)" in texts, texts
    assert "EI-selected" in texts, texts
    assert "DFT density of EI-selected" in texts, texts
    assert "best of the generation" in texts, texts   # kept
    assert "best so far" not in texts, texts          # line removed
    assert not any("(" in t and "MLP)" not in t for t in texts), texts
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_build_summary_top10_sigma_uses_highest_gp_densities():
    """TOP-10 is ranked by density_gp, not by σ or by crystal id."""
    rows = [(1, cid, -1.0, 100.0, 1.8, -1.8) for cid in range(1, 13)]
    gen_ids = {1: list(range(1, 13))}
    pred = {
        cid: {"residual": 0.1, "density_gp": 1.70 + 0.01 * cid,
              # 1.96σ = 0.005 * (13 - cid): the densest are the most certain
              "std": 0.005 * (13 - cid) / report_module.UNCERT_SCALE}
        for cid in range(1, 13)
    }
    summary = report_module.build_summary(rows, gen_ids, pred, {}, {})

    expected = sum(0.005 * (13 - cid) for cid in range(3, 13)) / 10
    assert summary[1]["sigma_top10"] == pytest.approx(expected)
    assert summary[1]["sigma_mean"] == pytest.approx(
        sum(0.005 * (13 - cid) for cid in range(1, 13)) / 12)
    assert summary[1]["sigma_top10"] < summary[1]["sigma_mean"]


def test_build_summary_ei_uc_uses_only_the_ei_picks():
    """The purple curve is the EI picks' mean UC, not the whole population's."""
    rows = [(1, cid, -1.0, 100.0, 1.8, -1.8) for cid in range(1, 6)]
    gen_ids = {1: [1, 2, 3, 4, 5]}
    pred = {
        cid: {"residual": 0.1, "density_gp": 1.8,
              "std": 0.01 * cid / report_module.UNCERT_SCALE}
        for cid in range(1, 6)
    }
    summary = report_module.build_summary(rows, gen_ids, pred, {}, {1: [5, 3]})

    assert summary[1]["sigma_ei"] == pytest.approx((0.05 + 0.03) / 2)
    assert summary[1]["sigma_mean"] == pytest.approx(0.03)
    assert summary[1]["sigma_ei"] != pytest.approx(summary[1]["sigma_mean"])


def test_learning_plot_draws_the_top10_curve(run, tmp_path, monkeypatch):
    import matplotlib.figure
    path, res, _densities, _picks = run
    captured = {}

    def fake_savefig(self, *args, **kwargs):
        captured["fig"] = self
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fake_savefig)

    rows, gen_ids = report_module.parse_individuals(str(res / "Individuals"))
    pred = report_module.parse_output_gp(str(res / "OUTPUT.txt"))
    dft = report_module.parse_density_log(str(res / "density.log"))
    ei_log = report_module.parse_ei_log(os.path.join(path, "ei.log"))
    summary = report_module.build_summary(rows, gen_ids, pred, dft, ei_log)
    learning = report_module.parse_learning_curve(os.path.join(path, "log"))

    assert "Saved" in report_module.plot_learning(
        summary, learning, str(tmp_path / "learning.png"))
    ax = captured["fig"].axes[0]
    labels = [line.get_label() for line in ax.get_lines()]
    assert "mean UC of EI selected" in labels, labels
    assert "mean UC of TOP-10 cocrystal" in labels, labels
    assert "(" not in "".join(labels), labels      # no parenthetical suffixes
    assert ax.get_title() == "", ax.get_title()
    assert len(captured["fig"].axes) == 2          # twin kept when data exists


def test_learning_plot_omits_the_right_axis_without_data(run, tmp_path, monkeypatch):
    """No GP learning curve lines -> no meaningless 0-1 axis on the right."""
    import matplotlib.figure
    path, res, _densities, _picks = run
    captured = {}

    def fake_savefig(self, *args, **kwargs):
        captured["fig"] = self
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fake_savefig)

    rows, gen_ids = report_module.parse_individuals(str(res / "Individuals"))
    pred = report_module.parse_output_gp(str(res / "OUTPUT.txt"))
    dft = report_module.parse_density_log(str(res / "density.log"))
    ei_log = report_module.parse_ei_log(os.path.join(path, "ei.log"))
    summary = report_module.build_summary(rows, gen_ids, pred, dft, ei_log)

    assert "Saved" in report_module.plot_learning(
        summary, {}, str(tmp_path / "learning.png"))
    assert len(captured["fig"].axes) == 1, captured["fig"].axes


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
    assert ax.get_title() == "", ax.get_title()
    assert "best so far" in texts, texts
    assert not any("best so far (" in t for t in texts), texts
    plt.close(fig)


def test_convergence_puts_the_uncertainty_on_the_gp_prediction():
    """σ is the spread of the GP mean, so the bar is drawn around μ, not ρ_ML."""
    series = [
        {"gen": 2, "density": 1.700, "density_gp": 1.820, "sigma": 0.030,
         "dft": 1.760},
        {"gen": 3, "density": 1.740, "density_gp": 1.860, "sigma": 0.020,
         "dft": 1.850},
    ]
    import matplotlib.pyplot as plt
    from matplotlib.container import ErrorbarContainer

    fig, ax = report_module._convergence_figure(series)
    bars = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert bars, ax.containers
    # centres on the GP predictions, not on the densities of the fourth argument
    assert list(bars[0][0].get_ydata()) == pytest.approx([1.820, 1.860])
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("GP prediction" in t and "1.96" in t for t in texts), texts
    assert any("DFT density" in t for t in texts), texts
    # the search's own density is only the ceiling line, not a marker series
    assert not any("ML density of selected" in t for t in texts), texts
    step = [l for l in ax.get_lines() if l.get_marker() == "o"]
    assert step, [l.get_marker() for l in ax.get_lines()]
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


# ──────────────────────────────────────────────────────────────────────
#  results<N> resolution: ei.log now lives with the run that wrote it
# ──────────────────────────────────────────────────────────────────────

def test_newest_results_folder_wins(tmp_path):
    (tmp_path / "results1").mkdir()
    (tmp_path / "results2").mkdir()
    assert report_module.resolve_results_dir(str(tmp_path)) == str(tmp_path / "results2")


def test_results_folder_can_be_selected(tmp_path):
    (tmp_path / "results1").mkdir()
    (tmp_path / "results2").mkdir()
    assert report_module.resolve_results_dir(str(tmp_path), 1) == str(tmp_path / "results1")
    assert report_module.resolve_results_dir(str(tmp_path), "results2") == str(tmp_path / "results2")


def test_unknown_results_folder_is_reported(tmp_path):
    with pytest.raises(SystemExit):
        report_module.resolve_results_dir(str(tmp_path), 9)


def test_run_without_a_results_folder_reads_the_run_dir(tmp_path):
    assert report_module.resolve_results_dir(str(tmp_path)) == str(tmp_path)


def test_cli_selects_the_results_folder(tmp_path, capsys):
    """cli.py carries its own copy of the parser: --results has to land there."""
    for name, gens in (("results1", 2), ("results2", 3)):
        res = tmp_path / name
        res.mkdir()
        with open(res / "Individuals", "w", encoding="utf-8") as fd:
            fd.write(HEADER)
            for gen in range(1, gens + 1):
                for i in range(4):
                    fd.write(_row(gen, (gen - 1) * 4 + i + 1, 1.70 + 0.01 * gen))
    from gpevo.cli import main

    main(["ei-report", "-p", str(tmp_path), "--results", "1", "-n"])
    assert "Generations 1–2" in capsys.readouterr().out
    main(["ei-report", "-p", str(tmp_path), "-n"])      # newest by default
    assert "Generations 1–3" in capsys.readouterr().out


def test_ei_log_is_read_from_the_results_folder(run):
    """It sits next to Individuals now, not in the home directory."""
    path, res, _densities, picks = run
    (res / "ei.log").write_text(
        "".join(f"gen {gen}:{cid}\n" for gen, cid in picks.items()), encoding="utf-8")
    os.remove(os.path.join(path, "ei.log"))   # nothing left to fall back on
    text = report_module.ei_report(path=path, quiet=True, trials=20)
    assert "3 EI-selected crystals" in text
