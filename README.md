![GPEvo](docs/assets/lockup-wave-light.svg)
**GPEvo** is a pure-Python implementation of the USPEX-GP (based on USPEX-9.4.4) molecular crystal structure prediction workflow, with Gaussian-process (GP) guided active learning built in.

| Item | Description |
|---|---|
| Upstream project | [USPEX](https://uspex-team.org) (the GP-enabled `uspex-gp` variant) |
| This repository | <https://github.com/fenggo/gpevo> |
| License | MIT, see [LICENSE](LICENSE) |

> Chinese version: [README.cn.md](README.cn.md).

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Running and output](#running-and-output)
  - [Output files](#output-files)
  - [`Individuals.traj`](#individualstraj)
  - [`origin`](#origin)
- [INPUT.txt extensions](#inputtxt-extensions)
  - [GP active learning](#gp-active-learning)
  - [swarm operator (optional)](#swarm-operator-optional)
  - [Reinforcement-learning operators / parameter tuning (optional)](#reinforcement-learning-operators--parameter-tuning-optional)
- [Analysis and plotting tools](#analysis-and-plotting-tools)
- [Directory layout](#directory-layout)
- [Design documents](#design-documents)
- [Relationship to the MATLAB version](#relationship-to-the-matlab-version)

---

## Overview

- **Molecular-crystal evolutionary search (310)**: all the variation operators — heredity / random / softmutation / permutation / latmutation / rotmutation — plus elitism, following the same pipeline as the MATLAB version; also an optional **swarm operator** (molecule-aware rigid-body moves, see [swarm operator](#swarm-operator-optional)).
- **GP active learning**: every structure gets density/energy predictions and uncertainties from `uspexkit.gp()`; the training set **grows automatically** — at the end of generation 1 the best few structures are labelled by the reference method to build the database, and from then on each generation picks top-K by EI (or uncertainty / random), labels them and appends them to the training set, invalidates the model cache, retrains, re-predicts, and writes the new fitness back.
- **Symmetry and random structure generation**: Wyckoff expansions for all 230 space groups, with a Cython port of `random_cell_mol` (compiled automatically by `pip install .`).
- **Molecule validity checks**: `newMolCheck`'s atom-pair distance scan also has a Cython kernel (`molcheck_cy.pyx`), about 85× faster on an 8-molecule system; it falls back to a pure-numpy implementation when the extension is not built.
- **Cluster submission**: two paths, local subprocesses and remote schedulers, with MPI environment variables handled automatically.
- **GP training-set bootstrap**: no need to prepare `feature_mlp.csv` / `feature.csv` / `structures.traj` up front — an empty (or even non-existent) directory is created automatically at the end of generation 1.
- **Automatic plotting**: after each generation is evaluated, a four-panel GP status figure (`gp_gen*.png`) is written under `results*/`.
- **Result comparison tools**: `compare_runs.py` compares density/energy distributions across run directories, and `plot_dft.py` plots DFT results.

## Installation

Requirements: Python ≥ 3.9 plus numpy / scipy / networkx / spglib; building the Cython extensions needs a C compiler (GCC is fine).

```bash
git clone https://github.com/fenggo/gpevo.git
cd gpevo
pip install .                          # with network access (build isolation pulls setuptools/cython)
pip install . --no-build-isolation     # offline, uses the already-installed setuptools + Cython
```

After installing:

| Item | Name |
|---|---|
| Distribution | `GPEvo` |
| Import package | `gpevo` |
| Command | `gpevo` |

```bash
python -c "import gpevo; print(gpevo.__file__)"   # verify where it was installed
```

**Running without installing**: put `INPUT.txt` in the repository root and run `python gpevo.py` directly (it adds the repo root to `sys.path`).

## Quick start

`example/` contains a sample 310 input set (`INPUT.txt`, `MOL_1`, `MOL_2`, `POSCAR`, etc.); copy it out as a template for a working directory:

```bash
mkdir -p run && cp example/INPUT.txt example/MOL_* run/
cd run
gpevo                       # equivalent to python -m gpevo
```

## Running and output

1. Prepare a working directory with `INPUT.txt` and the molecule templates `MOL_1`, `MOL_2`, ... (all inputs of the 310 molecular-crystal path).
2. In that directory, run:

```bash
gpevo                # or python -m gpevo
```

### Output files

| File / directory | Contents |
|---|---|
| `OUTPUT.txt` | Run log, including a per-generation `ID Origin Composition Enthalpy Volume KPOINTS SYMMETRY` table; with GP enabled, four extra columns `Enthalpy(GP) UC Density(GP) UC` are appended |
| `results1/Parameters.txt` | A byte-for-byte copy of `INPUT.txt` made at startup (the equivalent of the `cp` in MATLAB `Start.m:66`); written only once when a new `results<N>` is created, never overwritten on restart, so you can reconstruct the settings the run originally started with |
| `results1/` (`results2/`, ...) | `gatheredPOSCARS`, `Individuals`, `Individuals.traj`, `origin`, `gp_gen*.png`, `density_predict.log` |
| `CalcFold1/`, ... | Per-structure relaxation / GP working directories; `gp.csv` records every GP prediction |
| `results*/<ID>/` | Working directory where a single structure is labelled by the reference method (SIESTA); empty directories left behind by an interrupted run are cleaned up automatically at the next selection |
| `ei.log` | IDs of the crystals EI selected in each generation |

### `Individuals.traj`

An ASE-readable version of `gatheredPOSCARS`: one frame is appended as soon as each structure finishes, in the same order as the `Individuals` table (strictly identical when `numParallelCalcs=1`), so frame *i* is the structure with ID *i+1* in `Individuals`. Each frame's `atoms.info` carries `Number / generation / howCome / symg / enthalpy / density` (plus `energy_pred / uncertainty_eng / density_gp / uncertainty` when a GP prediction exists), and the enthalpy is also attached as a single-point energy, so it can be analysed directly:

```python
from ase.io import read
frames = read("results1/Individuals.traj", index=":")
print(frames[0].info)                          # metadata matching the Individuals row
print(frames[0].get_potential_energy())        # enthalpy (eV)
```

### `origin`

Records which operator produced each structure and who its parents were (`resFolder/origin` of `WriteOUTPUT.m`); it is the data source `makeFigures.m` uses to colour structures by operator:

```
 ID    Origin    Enthalpy   Parent-E   Parent-ID
   1   Random      -0.829    -0.829  [         0]
   2   Seeds       -1.956    [          ]
 223 keptBest      -1.988    -1.988  [       151]
```

`Enthalpy` / `Parent-E` are both **per-atom** enthalpies (`Enthalpies(end)/sum(numIons)`); structures with no parent (random) have `Parent-ID` = `0` and repeat their own enthalpy, while seed structures have empty parent info, so `Parent-E` is left blank. The format is byte-for-byte identical to the MATLAB version (including the blank columns when a parent is empty).

## INPUT.txt extensions

The option groups below are all **optional extensions** — omit them and the default behaviour applies. Prefer the canonical USPEX form `value : keyword` (`keyword : value` is also accepted); comments may start with `%`, `!` or `#`.

### GP active learning

```
1         : gp            % switch: 1 = enabled, adds a GP evaluation step after relaxation
24        : ncpu          % parallel cores for GP training; if omitted, takes the last % numProcessors entry, else 8
data11_22 : data          % training-data directory (used directly by gp() if present, otherwise built by this run)
1.8       : den           % density gate: gp()'s --den, also the lower density bound of the CalcFitness EI gate
0.04      : u             % uncertainty threshold (±1.96σ); a structure is sent to DFT only above it
2         : k             % number of EI top-K picks per generation
ei        : mode          % ei | random | uncertainty
2         : nboot         % structures labelled in the first batch when the training set is empty (default 2, must be ≥2)
1         : cache         % 1 (default) = pred reuses the 10-D feature already computed in the GP step; 0 = recompute per crystal
1         : explore       % 1 (default) = full EI; 0 = drop the σ·φ(Z) term and rank by improvement alone
sklearn   : gpbackend     % GP training backend: sklearn (default) | gpytorch (opt-in)
200       : gpiter        % max L-BFGS iterations per restart (gpytorch only)
0.1       : gplr          % L-BFGS learning rate (gpytorch only)
```

`gpbackend` / `gpiter` / `gplr` select the opt-in GPyTorch training backend (`pip install gpevo[gpytorch]`). With the default `sklearn` backend nothing changes. GPyTorch is a **different model** (LinearKernel / RBFKernel parameterisation, L-BFGS on the exact marginal log likelihood, GaussianLikelihood noise), so its OUTPUT.txt values are not byte-for-byte comparable with the sklearn/MATLAB parity run, and the learning curve reports `LOO RMSE: n/a`. The scheduler command is self-contained (`python -m gpevo.uspexkit.cli gp ...`) and no longer needs an external uspexkit install.

#### `explore : 0` — turning off EI's exploration term

EI = (μ − f_best)·Φ(Z) + **σ·φ(Z)**. The second term is the only one that **rewards the unknown**: a crystal the GP has never seen carries a wide posterior, so σ·φ(Z) can outweigh the improvement term and a **low-quality crystal whose prediction is actually worse than the incumbent** ends up ranked first, wasting an expensive reference calculation. With it disabled only the pure exploitation term remains — improvement magnitude × probability of improvement:

```
1.9 is the observed incumbent (largest DFT density in feature.csv)
  crystal A: μ=1.82, σ=0.50  →  EI = −0.035 + 0.197 = 0.162   ← surprisingly wins
  crystal B: μ=1.95, σ=0.05  →  EI =  0.042 + 0.012 = 0.054
with explore : 0
  crystal A: −0.035   ← back where it belongs
  crystal B:  0.042   ← wins
```

The price is losing exploration: when the training database has had no new region for a long time, high-σ crystals are no longer sampled preferentially and the model may converge to a local region. To keep exploration, leave it at `1`, or switch to `mode : uncertainty` (which is pure σ ranking — a different thing). The `mode` label is printed as `EI (no exploration)` to mark the difference.

#### INPUT.txt is re-read on restart

The restart branch of `Start()` recovers ORG from `Current_ORG.pkl`; it used to re-parse only the fingerprint block, so **any INPUT.txt edits made while stopped had no effect** (the GP `den` / `u` / `k` / `explore` settings were especially easy to overlook). A restart now re-runs the whole option parser (`createORG_System` / `_AbinitCode` / `_Symmetry` / `_EA` / `_Molecule` / `_Fingerprint` — all pure functions that only change ORG attributes), and the changed entries are printed:

```
  INPUT.txt (re-read on restart) gpDen: 1.8 -> 1.75
  INPUT.txt (re-read on restart) gpU: 0.03 -> 0.06
```

Two kinds of attributes are **never** overwritten by INPUT.txt and still come from the checkpoint:

- **Structural quantities fixed at the very start**: `dimension` / `molecule` / `varcomp` / `optType` / `populationSize` / `numGenerations` / `numIons` / `resFolder` — changing them mid-run would contradict the population already restored from the checkpoint;
- **Evolving quantities the loop writes back itself**: `latVolume` (re-averaged from population volumes each generation), `firstGeneSplit*`, and the `tournament` / `fracTrans` / `howMany*` values that `update_STUFF` rebuilds every generation.

A broken INPUT.txt does not crash a running calculation either: if parsing throws, it prints `Warning: INPUT.txt could not be re-read on restart` and puts the checkpoint back unchanged before continuing.

What `cache : 1` saves is one ReaxFF re-optimisation + four GULP runs (feature calculation) per crystal per generation. The precondition is that the `gulp.cif` read by the GP step and the `Individuals.traj` frame are **the same relaxed structure** (`WriteIndividualsTraj` writes `s.COORDINATES`), so reusing it does not swap structures — it merely skips a repeated minimisation. The log prints how many were actually reused:

```
  pred: reusing the GP-step feature for 132/134 crystal(s), skipping their ReaxFF re-optimisation
```

> If the `gp` line is not read because of a syntax issue (for example the keyword was polluted by a comment), the switch **silently** stays at 0 and the entire GP chain never happens. Startup prints a warning listing the offending lines.

Which quantity EI uses as its acquisition function is decided by `optType`:

| optType | Acquisition target μ | σ |
|---|---|---|
| `1` (enthalpy minimisation) | GP predicted energy `energy_gp`, negated (minimisation) | `uncertainty_eng` |
| `-1` (enthalpy maximisation) | the same, taken positive (maximisation) | `uncertainty_eng` |
| Others (default 11, etc.) | GP predicted density `density_gp` (maximisation) | `uncertainty` |

`den` (density gate) is unaffected and always applies to the predicted density; `u` (±1.96σ) applies to the σ of the corresponding acquisition target, so with optType 1 its unit is energy (eV) and should be tuned to the scale of the energy uncertainty. When no structure in a generation has a usable energy prediction (an old `gp.csv`, or `gp()` failed), a note is printed and it falls back to density.

Pipeline:

1. **Bootstrap (end of generation 1)**: when the training database has fewer than 2 rows, take the top `nboot` structures by fitness, let `uspexkit.calc()` compute their energies/densities with the reference method (SIESTA), and build `<homePath>/<data>/{feature_mlp.csv, feature.csv, structures.traj}` automatically. The GP step is skipped entirely for this generation (no structure fails).
2. **Per-generation growth (from generation 2)**: inside `CalcFitness`, take top-`k` by the `mode` rule (EI / uncertainty / random), pass the `u` (±1.96σ) and `den` gates, and send the selected structures to the reference method for labelling → appended to the training database.
3. **Retraining**: the model cache is cleared only when a new row really entered the database — these pickles (`gpr_density.pkl` / `gpr_energy.pkl` / `rfr.pkl`, scattered across `CalcFold*`, `results*` and `data/`) are stamped only by kernel version, so without clearing they would not retrain as the data grows. Then `uspexkit.pred()` re-predicts and writes the new values back into the fitness.

Is the data really growing? Watch these two log lines:

```
  GP training set has only 0 labelled row(s) in /home/feng/tnt/data_exp: seeding it with the best structures of generation 1
  GP training set: 0 -> 2 rows; dropped 3 cached model(s)
```

Each generation also prints a learning-curve line — **this line is the evidence for whether learning is actually happening** (it reads `+0` when a gate blocked everything and no label was added):

```
  GP learning curve (gen 3): rows 216 -> 220 (+4)  LOO RMSE density=0.0312 g/cm^3  energy=0.4123 eV
```

`LOO RMSE` is computed directly from the Cholesky factor of the already-fitted model via the hat-matrix identity (Rasmussen & Williams 5.12, no refitting needed); the hyperparameters are the ones fitted on all the data so far, so it is slightly optimistic. It shows `n/a` when the database exceeds 2000 rows, the cached uspexkit model cannot be found, or the backend is `gpytorch` (the hat-matrix identity needs sklearn's Cholesky factor / `alpha_`).

When nothing grows, it also says which gate blocked it (`No crystals above…` / `density threshold gate` / `calc failed` / `training set is still N row(s)`). The usual cause is that the reference method cannot run: labels depend on `siesta` (`mpirun`) + `<homePath>/Specific/*.psf`, and a missing one triggers an early warning at startup. Every generation also draws `results*/gp_gen*.png` (four panels: population distribution / exploration map / predicted vs adopted values / extrapolation check).

> The GP kernel comes from [uspexkit](https://gitee.com/fenggo/uspexkit) (sources under `~/uspexkit`), vendored into `gpevo/uspexkit/` and called as an in-process API; changing the kernel means editing the vendored `core.py`. The default `sklearn` backend is the one used for MATLAB character-for-character parity. With `gpbackend : gpytorch` an opt-in GPyTorch model is trained instead — different kernel parameterisation and optimiser, so its numbers are not directly comparable and the learning curve reports `LOO RMSE: n/a`. The scheduler command is self-contained (`python -m gpevo.uspexkit.cli gp ...`) and needs no external uspexkit install.

### swarm operator (optional)

Molecule-aware rigid-body motion, off by default (leaving it out does not change the existing operator mix at all):

```
0.1   : fracSwarm    % extra swarm structures per generation = round(populationSize × this value)
0.5   : swarmStep    % upper bound (0~1) on the fraction of displacement toward the corresponding molecule of the best structure
30    : swarmAngle   % upper bound (degrees) on the rotation interpolating orientation toward the best structure via slerp (0 = no rotation)
0     : swarmInertia % PSO inertia weight w (0 = no momentum)
0     : swarmCognitive % PSO cognitive coefficient c1 (0 = no personal best)
```

It treats each **molecule as a whole rigid body**, moving it by a random fraction of the displacement toward the corresponding molecule in the current best structure, while interpolating its orientation toward that molecule with Kabsch + quaternion slerp (a single rotation never exceeds `swarmAngle`, so it is no longer a random rotation) — `MOLCOORS` undergoes only rigid transforms, so intramolecular geometry is **bit-for-bit unchanged**, unlike atom-level soft-mode displacement which can tear molecules apart. The displacement first aligns the best structure's centre of mass to the child's reference frame using its cell, then applies minimum-image folding; the cell itself is unchanged (left to the subsequent relaxation). Each structure also carries a picklable velocity / personal-best (pbest) state that survives restarts via `Current_POP.pkl`, with momentum and the cognitive term controlled by `swarmInertia` / `swarmCognitive` (both default 0 = only the social term toward the current best remains). For the algorithm details, how it differs from PSO, and an assessment of its suitability for multi-component cocrystals see [`docs/swarm_operator.md`](docs/swarm_operator.md); recommended values for the two coefficients and their stability derivation are in [`docs/swarm_pso_coefficients.md`](docs/swarm_pso_coefficients.md).

Two notes:

- This is the PSO **idea** (move toward the best individual + random step size) implemented for molecular crystals, **not** a port of uspex-gp's PSO module: the latter places atoms randomly one by one and applies atomic soft-mode displacements, and does not support molecules.
- swarm structures are added **on top of** `populationSize` (like the structures appended by elitism), so they do not take quota away from existing operators; with `fracSwarm` unset the count is 0 and behaviour is exactly as before.

### AntiSeeds (optional)

Treat regions of structure-fingerprint space that have already been **over-sampled** as repulsion sources: the more a structure resembles an anti-seed fingerprint, the more its fitness is raised (the less likely it is to be chosen as a parent), pushing the search surface outward. It corresponds to `pickAntiSeeds.m` + `AntiSeedsCorrection.m` in `uspex-gp` and takes effect after `CalcFitness_310`, before `FitnessRankingCorrection`.

**Usage**: put the structures you no longer want sampled (several POSCARs concatenated in sequence) into `<homePath>/AntiSeeds/POSCARS`; they are read at startup (both for fresh runs and restarts) and their fingerprints computed as repulsion sources.

```
0.5   : antiSeedsMax         % penalty strength coefficient; 0 (default) disables AntiSeeds entirely
0.02  : antiSeedsSigma       % penalty width coefficient
3     : antiSeedsActivation  % |value|: >0 adds the whole population as anti-seeds from that generation on, <0 adds only the best structure
```

Penalty formula (`AntiSeedsCorrection.m:57`): each population member's fitness gets

```
Δ = Max · exp( −cos_dist² / (2·Sigma²) )
```

where `cos_dist` is the cosine distance between that structure and the anti-seed fingerprint: an exact match (distance 0) receives the full `Max`, and the penalty decays quickly as the difference grows.

- `Max` adapts each generation: **current value = `antiSeedsMax` × (mean fitness of the best `bestFrac` portion − its minimum)**, i.e. it grows as the fitness spread widens; after `generation > antiSeedsActivation + 1` a half-decay floor keeps it from dropping faster than `0.5× the previous generation`.
- `Sigma` adapts each generation as well: **current value = `antiSeedsSigma` × the mean pairwise fingerprint cosine distance among structures in the best region**; generation 1 uses it to calibrate all preloaded anti-seeds uniformly.
- Once `antiSeedsActivation` is reached it starts self-learning: `>0` appends all population fingerprints, `<0` only the single best of that generation — so the search automatically steers away from regions that are already crowded.
- State lives in `<homePath>/Current_ANTISEEDS.pkl` (same family as `Current_ORG/POP.pkl`); on restart it is loaded in preference to re-reading POSCARS, and `gpevo clean` removes it.

**Tuning tip**: `antiSeedsSigma` directly sets the repulsion radius, `Sigma ≈ antiSeedsSigma × mean fingerprint distance in the best region`. Measured on one set of TNT-like structures (cosine distances to the anti-seed 0.00 / 0.33 / 0.46) with `antiSeedsMax=2`, the fitness boosts are:

| `antiSeedsSigma` | Identical | Fairly similar (0.33) | Very different (0.46) |
|---|---|---|---|
| 0.001 (default) | 1.000 | 0.000 | 0.000 |
| 0.02 | 1.000 | 0.000 | 0.000 |
| **0.5** | 1.011 | 0.143 | 0.036 |
| 1.0 | 1.327 | 0.903 | 0.729 |

The default `0.001` (and small values such as 0.02) really means **only near-duplicate structures are pushed away**; to give repulsion a gradient and push away fairly similar structures too, use **0.3~1.0** (around 0.5 gives the best discrimination); larger values penalise everything globally. `antiSeedsMax` controls how much is added, and `0` (default) disables the whole feature.

> One known difference from MATLAB: the reference implementation computes anti-seed fingerprints **without intramolecular filtering**, whereas population fingerprints for molecular systems in gpevo (`molecule=1`, i.e. the 310 case) filter out intramolecular atom pairs. To stay consistent, `pickAntiSeeds` currently follows the reference (no filtering); if distances turn out systematically inflated and the penalty too weak, `intra_map` can be added on this path.

### Reinforcement-learning operators / parameter tuning (optional)

`RL_strategy ≥ 1` enables a hierarchical contextual bandit: the outer layer learns the quotas of the 6 operators, while the inner layer performs Thompson sampling on the **continuous variation magnitudes** of three operators — Rotation / LatMutate / softmutate. State is stored in `rl_agent_state.pkl`, and priors exportable across tasks live in `RLPriors/`.

```
2      : RL_strategy    % 0 = off; >=1 enables the bandit
0.9    : RL_forget      % forgetting factor λ (decay of historical statistics)
1.0    : RL_w1          % reward weight: being selected into the elite set
0.5    : RL_w2          % reward weight: relative enthalpy gain
0.6    : RL_armCap      % upper bound on a single operator's share
1.0    : RL_kappa       % inner exploration coefficient; larger = more aggressive amplitude exploration
1      : RL_autoExport  % export cross-task priors
angle  : RL_inner       % inner layer learns only the rotation angle, the other three keep their INPUT values (default all)
```

The four continuous parameters learned by the inner layer also have INPUT entries (with `RL_strategy=0` they are **fixed outright to the values you set**; with RL enabled they serve as the **learning starting point**):

```
45     : rotationAngleMax   % upper bound on rotation mutation angle, **in degrees** (converted to radians internally)
0.3    : translationMax     % upper bound on rigid-body translation after rotation
0.15   : mutationRate       % LatMutation strain σ (normal distribution, aligned with lattice_Mutation)
1.5    : softStepScale      % softmutate step-size scaling
```

With RL on, the search ranges are fixed: angle 15°–90°, translation 0.1–0.8, strain 0.05–0.30, step scaling 0.5–2.0 (settings outside the range are clamped to its endpoints).

### Letting RL learn a single parameter (`RL_inner`)

By default (with `RL_inner` omitted or set to `all`) the inner layer Thompson-samples **all four** magnitudes together every generation; to ablate one parameter at a time, name them individually with `RL_inner` — anything not named keeps its fixed INPUT value:

```
angle        : RL_inner   % learn only rotationAngleMax (translation/strain/step fixed)
translation  : RL_inner   % learn only translationMax
mutation     : RL_inner   % learn only mutationRate
softstep     : RL_inner   % learn only softStepScale
rotate       : RL_inner   % learn both parameters of arm 4 together (angle + translation)
angle,softstep : RL_inner % separate several with commas/spaces (do not write "angle, softstep")
none         : RL_inner   % inner layer learns nothing; all four use the fixed INPUT values
```

Recognition is case-insensitive (also accepts `all` / `on` / `1`, `none` / `off` / `0`, and `latmut` / `softmut`); misspelled names warn and fall back to `all`.

So "test only the rotation angle" means: `1 : RL_strategy` + `angle : RL_inner` + fixed values for the other three. Two caveats:

- The outer 6-arm bandit is still learning operator quotas (to freeze that too, you have to use `0 : RL_strategy` and sweep fixed values);
- For a controlled comparison, use a separate directory or delete `rl_agent_state.pkl` first (while it exists, the INPUT starting values no longer apply), and move `RLPriors/operator_prior.pkl` and `~/.config/uspex-gp/RLPriors/operator_prior.pkl` aside, otherwise the priors will override the starting point.

The angle actually used is `±rotationAngleMax/3` (`Rotation_310.py:40`), so setting 90° means ±30°; the `u` sampled each generation (angle = `15° + u×75°`) is recorded in the `history` of `rl_agent_state.pkl`.

## Analysis and plotting tools

```bash
# Compare density/energy distributions across runs (writes compare_runs.png)
python compare_runs.py --dirs tnt/results2 uspex-gp/results1 --labels gpevo octave

# DFT result plotting / space-group coverage statistics / EI log
python plot_dft.py
python sg_coverage.py
python ei_select.py
```

## Command line

```bash
gpevo               # run the search (equivalent to gpevo run, same as the old behaviour)
gpevo clean         # clean up the run directory
gpevo clean -n      # only list what would be deleted, touch nothing
gpevo clean -a      # also delete results*/
gpevo clean -p DIR  # clean a different directory (not the current one)
gpevo clean -f      # clean even without INPUT.txt / CalcFold in the directory
gpevo ei-report     # per-generation density evolution + EI convergence (see below)
gpevo ei-report -p DIR --out PREFIX --trials 300
```

### `gpevo ei-report`

Reads what has **already been written to disk** in a run directory (no GP/ReaxFF recomputation) — `results1/Individuals` (density of every structure in every generation), `results1/OUTPUT.txt` (the `Enthalpy(GP) UC Density(GP) UC Residual` block at the end of each generation's SUMMARY table, i.e. the GP mean μ and its uncertainty), `results1/density.log` (DFT reference), `ei.log` (ids EI sent to DFT each generation), `log` (`GP learning curve` lines) — and draws three figures plus a per-generation report:

- **evolution**: scatter of all structures + per-generation best curve + best-so-far step line; EI-selected crystals get green rings and DFT-verified values red diamonds;
- **convergence**: EI selection trend with **generation** on the x-axis and **density** on the y-axis — every EI-selected crystal is a purple square at its generation, holding the **GP predicted mean μ** that EI actually ranked on, with the `±1.96σ` error bar drawn on **that GP prediction** (σ is the posterior standard deviation of the GP, not of the MLP relaxation density); a grey dashed line connects the GP prediction and the DFT truth of the same crystal for comparison, the blue **dots + line** are the best density so far at that generation (density from `Individuals`), and the red diamond is that crystal's DFT truth;
- **learning**: mean GP σ per generation, number of labels added, and LOO RMSE (drawn only when logged).

The σ of the error bars comes from the `UC` column of each generation's SUMMARY table in `OUTPUT.txt` (it is already `1.96σ`; `ei-report` divides it back to the raw σ and scales it up again). SUMMARY tables are written **every generation**, so every EI-selected crystal of a fresh run has both μ and an error bar; a `σ for n/m EI-selected crystals` note is printed only when a generation genuinely has no GP row. Early versions read `density_predict.log`, which was rewritten every generation and held the post-label re-prediction, so its coverage was incomplete; the source is now uniformly `OUTPUT.txt`. Pointing it at an old run whose `OUTPUT.txt` has no GP table (e.g. a pure MATLAB output) yields a report with no GP predictions.

On a real run it answers "is EI actually useful?" directly: on one TNT·CL-20 run (14 generations, 690 structures, 26 labels), EI reached 1.918 on its first label and 1.94 by the fourth, while the random baseline only caught up around the ~8th label, finishing at 1.9470 versus the random 90th percentile of 1.9460.

What `gpevo clean` deletes is what a new round should not inherit: `AntiSeeds/`, `CalcFold*` (all of them, not just `CalcFold1`), `results*/<ID>/` (single-structure directories left by labelling), `*.pkl` / `*.mat` / `*.backup`, and status files such as `still_running`, `USPEX_IS_DONE`, `octave-workspace`, `Warnings`, `ei.log` (`ei.log` is appended to, so keeping it would mix crystals selected by two runs into one file, and `gpevo ei-report` would plot them together).

Never deleted: `Specific/`, `Seeds/`, `INPUT.txt`, `MOL_*`, `results*/` (unless `-a`), and the GP training-data directory `<data>/`. If a directory has neither `INPUT.txt` nor `CalcFold*`, the command refuses to run; add `-f` once you are sure.

## Directory layout

```
gpevo/                        # Python package
  FunctionFolder/
    USPEX/                    # core evolutionary algorithm (310 path)
    AbinitCode/               # GULP / SIESTA interfaces and I/O
    Submission/               # job submission (local subprocess / cluster scheduler)
    spacegroup/               # symmetry analysis + random_cell_mol (Cython port)
    sys/  Tool/               # system utility functions
  cli.py                      # console entry point (gpevo run / gpevo clean)
  tools/                      # subcommand implementations (clean.py)
  runtime.py  inputparser.py  log_utils.py  matrix.py  importers.py
gpevo.py                      # no-install entry point (run from the repository root)
compare_runs.py  plot_dft.py  # result plotting tools
sg_coverage.py                # space-group coverage statistics
ei_select.py                  # EI log
example/  Specific/  train/  ebind/  docs/   # examples, force fields and docs
```

## Feature-engineering findings (backed by a controlled experiment)

Whether adding structure fingerprints to the GP's 10-D features improves accuracy: the answer is no. The full procedure and scripts are in
[`docs/feature_ablation.md`](docs/feature_ablation.md):

- 216 real DFT samples, 5-fold CV, measured with the production kernel (ARD): baseline R² 0.650 (density) / 0.602 (energy); **adding fingerprints gave no improvement, and even a slight decrease**.
- **A control with shuffled fingerprints gave the same (or better) scores** → there is no incremental signal: the density proxy among the 10 dimensions already correlates with DFT density at r=0.707, which is redundant with fingerprint information.
- A trap recorded along the way: a multi-dimensional GP **must use ARD length scales**; an isotropic kernel collapses to "predict the mean" (R²≈0), which once led to the opposite, wrong conclusion.

Reproduce with: `python docs/feature_ablation.py --data ~/tnt22/data22`

## Design documents

`docs/` holds several design and rationale notes:

| File | Contents |
|---|---|
| `ea_flowchart.md` / `ea_mechanism.png` / `draw_ea.py` | Evolutionary-algorithm flowchart and the script that draws it |
| `uq_active_learning.md` / `uq_active_learning_flowchart.md` | GP active-learning (UQ) pipeline |
| `fingerprint_method.md` | Fingerprint method |
| `rl_bandit_design.md` | Reinforcement-learning bandit design (derivation / roadmap) |
| `rl_bandit_tutorial.md` | Reinforcement-learning module tutorial (principles + INPUT.txt configuration / single-parameter ablation / interpretation) |
| `molecule_damage_check.md` | The three molecule-damage checks, why the 1.780 Å intermolecular N–N bond was missed, and suggested improvements |
| `strain_matrix.md` | Where `Parents.strainMatrix` comes from (lattice strain), why it has zero readers, and the impact of gpevo not storing it |
| `rotation_operator.md` | The two kinds of rotation (rigid-body vs group torsion), default angles (90° / ±90°·±60°·±30°), `fracRotMut` default 0.1, breathing lattice 10–20% |
| `swarm_pso_coefficients.md` | Derivation of `swarmInertia` / `swarmCognitive`: stability triangle, constriction factor, order-of-magnitude check and recommended values |
| `swarm_operator.md` | swarm operator algorithm (per-molecule interpolation toward the current best + orientation quaternion slerp + pbest/inertia), comparison with textbook PSO / MATLAB PSO, suitability for molecular crystals, and the fixes and remaining improvements around cell alignment / MtypeLIST assertions / orientation slerp / pbest inertia state |

## Relationship to the MATLAB version

The MATLAB / OCTAVE source tree is not part of this repository, and the Python path does not need it either: no `USPEXPATH` environment variable, no MATLAB path configuration; the only entry points are `gpevo` / `python -m gpevo` / `python gpevo.py`.

## License

MIT, see [LICENSE](LICENSE).

## Links

- Repository: <https://github.com/fenggo/gpevo>
- Author: FengGo
