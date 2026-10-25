## Fingerprint-Based Structure Characterization for Crystal Structure Prediction

### 1. Fingerprint Descriptor

The fingerprint, originally proposed by Oganov and Valle [1], encodes the local chemical environment of a crystal structure into a fixed-length vector representation. For a structure containing \(N\) atoms of \(S\) species in a periodic cell of volume \(V\), the fingerprint is defined as a discretized, Gaussian-smeared radial distribution function (RDF).

For each atom \(i\) in the unit cell, the set of neighboring atoms \(\{j\}\) within a cutoff radius \(R_{\max}\) is identified across all periodic images. The contribution of each neighbor pair to bin \(b\) (spanning \([r_b, r_{b+1}]\) with bin width \(\delta\)) is computed via the integrated Gaussian overlap:

\[
\Delta_{ij}^{(b)} = \frac{1}{2} \left[ \mathrm{erf}\!\left(\frac{r_{b+1} - d_{ij}}{\sqrt{2}\,\sigma}\right) -
\mathrm{erf}\!\left(\frac{r_{b} - d_{ij}}{\sqrt{2}\,\sigma}\right) \right]
\]

where \(d_{ij}\) is the interatomic distance between atom \(i\) and its neighbor \(j\), and \(\sigma\) is the Gaussian smearing width. The per-atom fingerprint is then accumulated as:

\[
F_i^{\alpha}(b) = \frac{V}{4\pi\delta} \sum_{j \in \alpha} \frac{\Delta_{ij}^{(b)}}{N_\alpha \cdot d_{ij}^2} - 1
\]

where \(\alpha\) denotes the chemical species of neighbor \(j\) and \(N_\alpha\) is the number of atoms of species \(\alpha\) in the unit cell. The global pair fingerprint \(F^{\alpha\beta}(b)\) is obtained by summing over all atom pairs of the corresponding species, normalized by \(V / (4\pi \, N_\alpha N_\beta \, \delta)\).

The structure order parameter \(O_i\) for atom \(i\) is then defined as the species-weighted Euclidean norm across all bins:

\[
O_i = \sqrt{\sum_{b} \sum_{\alpha} w_\alpha \, \delta \cdot \left[F_i^{\alpha}(b)\right]^2 \Big/ (V/N)^{1/3}}
\]

with \(w_\alpha = N_\alpha / \sum_\beta N_\beta\). This order parameter quantifies the deviation from a uniform distribution—higher values indicate stronger local ordering. In the USPEX implementation, two fingerprint variants are computed: (i) a **molecule-centroid fingerprint** using only the centers of mass of each molecule (for molecular crystals), and (ii) a **full-atom fingerprint** where intra-molecular distances are masked out to focus on intermolecular packing.

### 2. Role in the USPEX Evolutionary Algorithm

The fingerprint serves three critical roles in the USPEX inter-generation pipeline [2]:

1. **Structure Ordering**: The per-atom order parameters \(O_i\) are averaged to produce a global structure order \(A_{\text{order}}\) and \(S_{\text{order}}\), which measure the degree of crystalline ordering. These values are used in the fitness function to penalize disordered structures.

2. **Quasi-Entropy**: The pairwise cosine distance between the atomic fingerprints of same-species atoms yields the structure quasi-entropy \(Q_{\text{entr}}\), a measure of local environment diversity. Low quasi-entropy structures (uniform local environments) are favored.

3. **Structure Similarity**: The cosine distance between the global fingerprint vectors of two structures provides a fast, rotationally invariant similarity metric. This is used extensively in the anti-seeding correction, fitness ranking de-duplication, and the `KeepBestStructures` selection to maintain population diversity.

The default parameters used in this work are \(R_{\max} = 12.0\) Å, \(\sigma = 0.05\) (corresponding to an effective Gaussian width of \(\sigma / \sqrt{2\ln 2}\)), and \(\delta = 0.08\) Å, yielding 150 bins. The smearing width is chosen to be comparable to typical thermal displacements, ensuring that the fingerprint is robust against small structural perturbations.

### 3. Computational Challenges and Acceleration

For large systems, the fingerprint computation is a severe bottleneck in the USPEX workflow. The procedure involves two stages:

**Stage 1 — Distance Matrix Construction** (`makeMatrices`): For \(N\) atoms in the unit cell, all periodic images within \(R_{\max}\) are enumerated by scanning a supercell of \((2L+1)^3\) translations with 8 sign combinations, yielding approximately \(N \times N \times (2L+1)^3 \times 8\) distance evaluations. For the TNT₄·CL20₄ system (228 atoms), this produces ~\(10^6\)–\(10^7\) distance pairs. The original Octave implementation uses `vertcat` for dynamic array growth, which triggers O(\(n^2\)) memory reallocation and dominates the runtime.

**Stage 2 — Fingerprint Calculation** (`fingerprint_calc`): For each of the ~150 bins, every valid distance pair is checked against the Gaussian window \(|\Delta| < 4\sigma\) and the erf integral is evaluated via a pre-computed lookup table. The triple loop (bins × cell atoms × neighbors) executes approximately \(150 \times 228 \times 10^4 \approx 3.4 \times 10^8\) iterations.

For the 228-atom TNT₄·CL20₄ energetic cocrystal system, benchmarked on an Intel Xeon processor with Octave 10.3.0, the distance matrix construction required **74 seconds** and the fingerprint calculation required **204 seconds** per structure. With 70 structures per generation and 50 generations, the fingerprint alone would consume approximately **5.4 hours per generation**, or **270 hours** for a full 50-generation run.

### 4. Cython Acceleration

To address this bottleneck, we developed a Cython-based acceleration module (`uspex_fast_core`) that replaces both stages with compiled C code operating without the Python Global Interpreter Lock (GIL). The key optimizations include:

- **Pre-allocated flat arrays** replacing Octave's `vertcat` dynamic growth, reducing memory operations from O(\(n^2\)) to O(1).
- **Inline erf approximation** using the Abramowitz–Stegun rational approximation (Eq. 7.1.26, ~\(10^{-7}\) precision), eliminating the need for a pre-computed lookup table.
- **Direct index tracking**: The distance matrix returns paired atom indices (`cc_idx`, `bc_idx`) alongside distances, enabling the fingerprint kernel to access `atom_fing` directly without Python-layer boolean masking.
- **Flat memory layout**: All arrays are stored as contiguous C-order memoryviews, maximizing cache locality in the innermost loops.

The Cython module is compiled with `-O3 -march=native` and interfaced with Octave through a temporary POSCAR file and `.mat` file exchange (using `scipy.io.savemat`). The Octave bridge functions (`fast_fingerprint.m`, `fast_fingerprint_fullatom.m`) provide a drop-in replacement with transparent fallback to the original code on any failure.

The benchmark results are summarized in Table 1:

**Table 1.** Performance comparison for a single 228-atom TNT₄·CL20₄ structure fingerprint computation.

| Component | Octave (s) | Cython (s) | Speedup |
|-----------|-----------|------------|---------|
| Distance matrix (`makeMatrices`) | 74.05 | 0.08 | **925×** |
| Fingerprint kernel (`fingerprint_calc`) | 204.14 | 0.013 | **~15,700×** |
| **Full pipeline** | **278.19** | **~0.1** | **~2,780×** |
| Per generation (70 structures) | ~5.4 hours | ~7 seconds | **~2,780×** |

The dramatic speedup in the fingerprint kernel (15,700×) arises from the combination of (a) the inline erf approximation replacing table lookups, (b) direct index-based access eliminating the conditional mask, and (c) the compiled C loop with no Python overhead. The overall 2,780× end-to-end acceleration reduces the per-generation fingerprint cost from the dominant bottleneck to a negligible overhead, shifting the computational bottleneck to the variation operators (particularly the rotation mutation with its Z-matrix conversion loop).

### 5. Verification

The Cython implementation was verified against the original Octave code by extracting identical structures (same enthalpy and volume) from the `Individuals` output files across generation boundaries. The maximum absolute difference in the order parameter \(O\) was \(<10^{-6}\), in the global fingerprint \(F^{\alpha\beta}\) was \(<10^{-5}\), and in the per-atom fingerprint \(F_i^{\alpha}\) was \(<10^{-5}\), confirming that the accelerated version produces numerically identical results within floating-point precision.

---

### References

[1] A. R. Oganov and M. Valle, *J. Chem. Phys.* **130**, 104504 (2009).

[2] A. R. Oganov, A. O. Lyakhov, and M. Valle, *Acc. Chem. Res.* **44**, 227–237 (2011).