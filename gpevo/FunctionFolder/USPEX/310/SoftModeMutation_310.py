"""Python equivalent of SoftModeMutation_310.m (soft-mode displacement).

Port of ``uspex0925/FunctionFolder/USPEX/310/SoftModeMutation_310.m``:

1. Pick a parent and compare its fingerprint to ``POP.SOFTMODEParents`` via
   ``cosineDistance`` / ``toleranceFing``.
2. Compute the soft modes (dynamical matrix -> eigenvectors) with
   :func:`softmode.calc_soft_modes`; sort frequencies with ``IX``.
3. Select a mode: the first non-zero frequency for a new parent, or, for a
   parent already in the cache, the first mode whose participation ratio /
   frequency differs enough from the last one used.
4. Sweep mode amplitudes ``(1-i/21)`` / ``(i/21-1)`` along the eigenvector,
   gated by ``newMolCheck(minDistMatrice - 0.1)``.
5. On success emit the offspring and update ``SOFTMODEParents``; otherwise
   fall back to ``Mutation_310``.

The eigenvector is built from ``COORDINATES`` (species-blocked fractional
coordinates, as in ``Start_POP_310``) and applied to ``MOLECULES`` in molecule
order -- the same ordering convention the MATLAB sources use.
"""
from __future__ import annotations

from importlib import import_module as _im

import numpy as np

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.USPEX.src.variation_common import (
    copy_individual, finalize_offspring, pick_parent, valid_molecules,
)
from gpevo.FunctionFolder.USPEX.src.softmode import (
    calc_soft_modes, default_good_bonds, move_along_soft_mode,
    valence, valence_electrons,
)
from gpevo.FunctionFolder.USPEX.src.NEW_coord2Zmatrix import NEW_coord2Zmatrix
from gpevo.FunctionFolder.sys.fingerprint_utils import _cosine_distance
from gpevo.FunctionFolder.sys.covalentRadius import covalentRadius
from gpevo.log_utils import console_print


def _fallback(count):
    """Mutation_310.m fallback (soft modes unavailable / all moves rejected)."""
    _im("gpevo.FunctionFolder.USPEX.310.Mutation_310").Mutation_310(count)


def _fingerprint_distance(ORG, first, second):
    """cosineDistance.m weighted fingerprint distance (same as KeepBest)."""
    if first is None or second is None:
        return None
    rows = np.asarray(first).shape[0]
    weight = np.asarray(getattr(ORG, "weight", 1.0), dtype=float)
    if weight.ndim == 0 or weight.size != rows:
        weight = np.ones(rows, dtype=float)
    try:
        return float(_cosine_distance(first, second, weight))
    except Exception:
        return None


def _softmode_inputs(ORG, individual):
    """Build the per-species arrays fed to calc_soft_modes."""
    atom_type = [int(z) for z in ORG.atomType]
    n_types = len(atom_type)
    R_val = np.array([float(covalentRadius(z)) for z in atom_type])
    # Missing/empty fields are common after a restart that predates these
    # attributes; fall back to the MATLAB lookup tables in that case.
    nval_raw = getattr(ORG, "NvalElectrons", None)
    N_val = np.asarray([] if nval_raw is None else nval_raw, dtype=float).ravel()
    if N_val.size != n_types or not np.isfinite(N_val).all():
        N_val = np.array([valence_electrons(z) for z in atom_type])
    val_raw = getattr(ORG, "valences", None)
    val = np.asarray([] if val_raw is None else val_raw, dtype=float).ravel()
    if val.size != n_types or not np.isfinite(val).all():
        val = np.array([valence(z) for z in atom_type])
    good_raw = getattr(ORG, "goodBonds", None)
    good = np.asarray([] if good_raw is None else good_raw, dtype=float)
    if good.shape != (n_types, n_types) or not np.isfinite(good).all():
        good = default_good_bonds(atom_type)
    num_ions = [int(v) for v in individual.numIons]
    at_types = np.repeat(np.arange(n_types), num_ions)
    return R_val, N_val, val, good, at_types


def SoftModeMutation_310(count):
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    stdmol = ORG.STDMOL
    tolerance = float(getattr(ORG, "toleranceFing", 0.0))
    how_many_mut = float(getattr(ORG, "howManyMut", 3) or 3)
    # gpevo-only RL knob (SoftModeMutation_310.m has no equivalent).  Default
    # 1.0 keeps the amplitude identical to the MATLAB reference.
    soft_step_scale = float(getattr(ORG, "softStepScale", 1.0) or 1.0)

    idx = pick_parent()
    if idx is None or idx >= len(POP.POPULATION):
        _fallback(count)
        return
    parent = POP.POPULATION[idx]
    molecules = parent.MOLECULES
    lattice = np.asarray(parent.LATTICE, dtype=float).reshape(3, 3)
    mtype_list = [int(v) for v in parent.MtypeLIST]
    num_ions = [int(v) for v in parent.numIons]
    num_mols = [int(v) for v in parent.numMols]
    types_a_list = list(parent.typesAList)
    N = int(sum(num_ions))
    fingerprint = getattr(parent, "FINGERPRINT", None)

    # ---- 1. Fingerprint de-duplication against SOFTMODEParents -------------
    # EA_310 replaces POP_STRUC with OFF_STRUC at the end of the generation,
    # so the cache that survives is the one on OFF_STRUC.  Alias POP's entry to
    # the same list to keep restarts / the current generation consistent.
    OFF = G["OFF_STRUC"]
    cache = getattr(OFF, "SOFTMODEParents", None)
    if cache is None:
        cache = list(getattr(POP, "SOFTMODEParents", []) or [])
        OFF.SOFTMODEParents = cache
    POP.SOFTMODEParents = cache
    flag = 0
    cache_id = -1
    for j, entry in enumerate(cache):
        dist = _fingerprint_distance(ORG, fingerprint, getattr(entry, "fingerprint", None))
        if dist is not None and dist < tolerance:
            flag = 1
            cache_id = j
            break

    # ---- 2. Soft modes ------------------------------------------------------
    try:
        R_val, N_val, val, good, at_types = _softmode_inputs(ORG, parent)
        coords = np.asarray(parent.COORDINATES, dtype=float).reshape(-1, 3)
        if coords.shape[0] != N:
            coords = coords[:N]
        freq, eigvector = calc_soft_modes(coords, lattice, at_types, R_val, N_val, val, good)
    except Exception as exc:
        console_print(f"SoftModeMutation: soft-mode calculation failed ({exc})")
        _fallback(count)
        return
    freq = np.abs(np.asarray(freq, dtype=float).ravel())
    IX = np.argsort(freq)
    freq_sorted = freq[IX]

    # ---- 3. Mode selection --------------------------------------------------
    non_zero = 0
    if flag == 1:
        last = cache[cache_id]
        last_num = int(getattr(last, "Softmode_num", 1))
        col_idx = IX[min(max(last_num - 1, 0), len(IX) - 1)]
        prev = eigvector[:, col_idx]
        M2 = float(np.sum(prev ** 2))
        M4 = float(np.sum(prev ** 4))
        participation_ratio = M2 / M4 if M4 > 0 else 1.0
        last_good = float(getattr(last, "Softmode_Fre", 0.0))
        for i in range(len(freq_sorted)):
            col = eigvector[:, IX[i]]
            pr2 = float(np.sum(col ** 2))
            pr4 = float(np.sum(col ** 4))
            pr1 = pr2 / pr4 if pr4 > 0 else 1.0
            ratio = participation_ratio / pr1 if pr1 > 0 else 1.0
            if (freq_sorted[i] > 1.05 * last_good) or (
                    freq_sorted[i] >= last_good and abs(1 - ratio) > 0.05):
                non_zero = i + 1
                break
    else:
        for i in range(len(freq_sorted)):
            if freq_sorted[i] > 1e-7:
                non_zero = i + 1
                break

    # ---- 4. Amplitude sweep -------------------------------------------------
    good_mutant = 0
    chosen_f = non_zero
    deviation = None
    mut_mol = None
    if non_zero > 0:
        upper = non_zero + int(round((3 * N - non_zero) / 2))
        loop = 0
        for f in range(non_zero, min(upper, len(IX)) + 1):
            if loop == 1:
                break
            col = eigvector[:, IX[f - 1]]
            vecnorm = np.linalg.norm(col[:3 * N].reshape(N, 3), axis=1)
            normfac = float(np.max(vecnorm)) / how_many_mut
            if not np.isfinite(normfac) or normfac <= 0:
                continue
            for i in range(0, 11):
                if np.random.rand() > 0.5:
                    disp = soft_step_scale * (1 - i / 21) * col / normfac
                else:
                    disp = soft_step_scale * (i / 21 - 1) * col / normfac
                try:
                    mut_mol, deviation = move_along_soft_mode(
                        molecules, disp, mtype_list, stdmol, NEW_coord2Zmatrix)
                except Exception:
                    continue
                # SoftModeMutation_310.m: newMolCheck(..., minDistMatrice-0.1).
                # valid_molecules applies the threshold adjustment and guards a
                # missing/short minDistMatrice the same way the other operators do.
                probe = AttrDict(MOLECULES=mut_mol, LATTICE=lattice, MtypeLIST=mtype_list)
                good_mutant = int(valid_molecules(probe, -0.1))
                if good_mutant == 1:
                    loop = 1
                    chosen_f = f
                    break

    # ---- 5. Emit offspring / update cache / fall back -----------------------
    if good_mutant == 1 and mut_mol is not None:
        child = copy_individual(parent)
        child.MOLECULES = mut_mol
        child.LATTICE = lattice
        child.numIons = num_ions
        child.numMols = num_mols
        child.MtypeLIST = mtype_list
        child.typesAList = types_a_list
        child.Parents = [AttrDict(
            parent=str(getattr(parent, "Number", "")),
            enthalpy=float(np.asarray(parent.Enthalpies, dtype=float).ravel()[-1]) / max(N, 1),
            mut_degree=deviation,
            mut_mode=chosen_f,
            mut_fre=float(freq_sorted[chosen_f - 1]),
        )]
        finalize_offspring(child, "softmutate")
        G["OFF_STRUC"].POPULATION.append(child)
        console_print(f"Structure {count}  generated by SoftModeMutation")

        if flag == 1:
            entry = cache[cache_id]
            entry.Softmode_Fre = float(freq_sorted[chosen_f - 1])
            entry.Softmode_num = chosen_f
        else:
            cache.append(AttrDict(
                lattice=lattice.copy(),
                molecules=molecules,
                fingerprint=fingerprint,
                Softmode_Fre=float(freq_sorted[chosen_f - 1]),
                Softmode_num=chosen_f,
                numIons=num_ions,
                numMols=num_mols,
            ))
        return

    _fallback(count)
