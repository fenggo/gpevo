# Supplementary Information — Nearest-Neighbour Residual Table (`nn_residual_table`)

## S1. Purpose

This supplementary note documents `data11/nn_residual_table.py`, a small
analysis utility that tabulates, for a reproducible random subset of the
training-set crystals, the crystal in the same set whose ten-dimensional
(10-D) feature vector is closest to it, together with the corresponding
residual. The table provides a direct, per-crystal view of how tightly the
ReaxFF-derived feature vector clusters, and of the numerical scale of the
`0.005` residual criterion used elsewhere in the code base (`gp()` and
`gpevo dedup`).

All structures and features referred to below belong to the `data11` data set:
719 crystals of the same composition, with a ReaxFF single-point feature row
per crystal in `feature_mlp.csv` and the matching relaxed geometry in
`structures_mlp.traj`.

## S2. Feature descriptor

Each crystal is represented by the 10-D vector stored in `feature_mlp.csv`:

| Index | Component | Physical meaning |
|------:|-----------|------------------|
| 1 | `etot` | total ReaxFF energy |
| 2 | `ebond` | bond-stretch energy |
| 3 | `eang` | angle-bending energy |
| 4 | `etor` | torsion energy |
| 5 | `evdw` | van der Waals energy |
| 6 | `ehb_cho` | hydrogen-bond energy, C–H···O environment |
| 7 | `ehb_chn` | hydrogen-bond energy, C–H···N environment |
| 8 | `ehb_chc` | hydrogen-bond energy, C–H···C environment |
| 9 | `ecoul` | Coulomb energy |
| 10 | `density` | crystal density of the ReaxFF-relaxed cell |

The vector is used **as stored**, in the raw (unstandardised) feature space.
This is exactly the representation consumed by `gpevo.uspexkit.core.raw_residual`
during Gaussian-process training and prediction.

## S3. Residual definition and nearest-neighbour search

For two crystals with feature vectors $\mathbf{x}_i$ and $\mathbf{x}_j$, the
residual is the squared Euclidean distance

$$
r_{ij} = \sum_{k=1}^{10} (x_{ik} - x_{jk})^2 .
$$

No standardisation, weighting, or normalisation is applied; the sum runs over
all ten components in the order given in Table S1. This definition is
numerically identical to the residual computed by `raw_residual()`.

For each sampled crystal $i$, the nearest neighbour is

$$
\mathrm{NN}(i) = \arg\min_{j \neq i} r_{ij},
$$

with the self-term excluded by setting $r_{ii} = \infty$ before the minimisation.
Both the sampled crystal and its neighbour are drawn from the same 719-crystal
set.

## S4. Sampling

The utility draws $n = 50$ indices without replacement from the 719 available
rows using `numpy.random.default_rng(seed)` with the default `seed = 42`.
The sample is therefore fully reproducible. The number of draws and the seed
are exposed as `--n` and `--seed` command-line options; when `--n` exceeds the
size of the data set it is silently capped at that size.

## S5. Output format

The output is a comma-separated table (`nn_residual_table.csv`) with one row
per sampled crystal and 11 columns:

| Column | Description |
|--------|-------------|
| `id` | index of the sampled crystal in `feature_mlp.csv` |
| `neighbor_id` | index of its nearest neighbour (self excluded) |
| `residual` | $r_{i,\mathrm{NN}(i)}$ as defined in Section S3 |
| `self_ehb_cho`, `self_ehb_chn`, `self_ehb_chc`, `self_density` | four selected descriptor components of the sampled crystal |
| `nn_ehb_cho`, `nn_ehb_chn`, `nn_ehb_chc`, `nn_density` | the same four components of the nearest neighbour |

The four reported components are the three hydrogen-bond terms and the
density; they are copied verbatim from the corresponding rows of
`feature_mlp.csv` and are intended to let a reader judge, by inspection,
whether a small residual reflects a genuinely similar local environment
rather than a coincidental cancellation of other terms. All numeric values
are written with eight decimal places.

## S6. Representative results

Table S2 lists six rows spanning the observed residual range, from the
smallest residual in the sample to the largest.

**Table S2.** Six representative rows of `nn_residual_table.csv`. Energies in
eV, density in g cm⁻³; residual is the squared 10-D distance defined in
Section S3.

| id | neighbor_id | residual | self_ehb_cho | self_ehb_chn | self_ehb_chc | self_density | nn_ehb_cho | nn_ehb_chn | nn_ehb_chc | nn_density |
|---:|------------:|---------:|-------------:|-------------:|-------------:|-------------:|-----------:|-----------:|-----------:|-----------:|
| 310 | 584 | 0.00000112 | -3.62954189 | -0.15621042 | -0.12072361 | 1.94835286 | -3.63000760 | -0.15614424 | -0.12072719 | 1.94842235 |
| 539 | 665 | 0.00000221 | -3.77739732 | -0.20848246 | -0.23223389 | 1.96496307 | -3.77725291 | -0.20861211 | -0.23179957 | 1.96473728 |
| 256 | 254 | 0.00014758 | -3.97141355 | -0.21987836 | -0.11760706 | 1.89725316 | -3.96615825 | -0.22040501 | -0.11769093 | 1.89571312 |
| 44  | 42  | 0.00263586 | -3.72529085 | -0.25730452 | -0.10900676 | 1.90093302 | -3.74070506 | -0.25205875 | -0.10845438 | 1.89906300 |
| 642 | 595 | 0.00391164 | -3.70565750 | -0.20418226 | -0.14450844 | 1.84031658 | -3.68400119 | -0.20211281 | -0.16213643 | 1.83897592 |
| 447 | 576 | 0.28350864 | -3.27159391 | -0.14583132 | -0.10407664 | 1.80149616 | -3.06352911 | -0.17243189 | -0.14573223 | 1.82871281 |

For the 50 sampled crystals the nearest-neighbour residual spans
$1.12\times10^{-6}$ to $0.284$, with a median of $3.27\times10^{-3}$;
26 of the 50 values (52 %) fall below 0.005.

## S7. Relation to the 0.005 criterion and interpretation

The threshold 0.005 is the default `tolerance` of `gp()`, where the residual
is used to flag a candidate whose feature vector lies close to a training-set
entry, and it is also the default `feature_tol` of `gpevo dedup`, where a
small residual may replace the energy-degeneracy test. The table shows that
this is a *tight* criterion: only about half of randomly drawn crystals have
any neighbour within it, even in a data set that contains many structurally
repeated entries.

Two caveats should be kept in mind when reading the table:

1. **Feature-space proximity is not proof of structural identity.** A full
   pair-wise analysis of the same 719 crystals found 1,610 pairs with
   $r_{ij} < 0.005$; pymatgen `StructureMatcher` confirmed 1,585 of them
   (98.4 %) as the same crystal, but 25 pairs (1.6 %) were distinct crystals
   with nearly identical features. This is why `gpevo dedup` still requires a
   positive `StructureMatcher.fit()` in addition to a small residual.
2. **The residual is computed in raw feature units.** Its scale is therefore
   tied to the force field and to the composition of the data set; the value
   0.005 is not transferable to a different descriptor or a differently
   normalised feature table without re-calibration.

## S8. Reproduction

From the `data11` directory:

```bash
python nn_residual_table.py \
    --data feature_mlp.csv \
    --out nn_residual_table.csv \
    --n 50 --seed 42
```

The script prints the number of crystals loaded, the number sampled, the
output path, and the residual range of the generated table. It depends only on
NumPy and the Python standard library.

## S9. Related documents

* `docs/fingerprint_method.md` — the Oganov–Valle structural fingerprint used
  by the USPEX evolutionary engine (a different descriptor from the 10-D
  feature vector discussed here).
* `docs/algorithm_review.md` — role of the residual cutoff in the GP/EI
  acquisition workflow.
* `data11/fingerprint_uniqueness.py` — the full pair-wise analysis summarised
  in Section S7, including the pymatgen structure-matching check.
