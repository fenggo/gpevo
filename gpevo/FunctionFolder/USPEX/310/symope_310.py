"""Python equivalent of symope_310.m, delegating random generation to the USPEX binary."""
from __future__ import annotations

import io
import math
import os
import re
import subprocess
import sys

import numpy as np

from gpevo.runtime import G, num2str, resource_path
from gpevo.matrix import latConverter, det
from gpevo.FunctionFolder.sys.spaceGroups import spaceGroups

def _run_cython_mol(rc_text):
    """Run the in-process Cython port of ``random_cell_mol``.

    Returns ``(ier, out)`` or ``None`` when the extension is not built or
    fails.  The port reads ``rc.in`` from stdin and writes the generator
    listing to stdout, exactly like the Fortran binary.
    """
    try:
        from gpevo.FunctionFolder.spacegroup import random_cell_mol_cy as _rcm
    except Exception:
        return None

    old_in, old_out = sys.stdin, sys.stdout
    out_buf = io.StringIO()
    sys.stdin = io.StringIO(rc_text)
    sys.stdout = out_buf
    try:
        ier = _rcm.main()
        out = out_buf.getvalue()
    except Exception:
        return None
    finally:
        sys.stdin, sys.stdout = old_in, old_out
    if not out:
        return None
    return (ier if ier is not None else 0), out


def _run_random_cell_mol(random_gen, rc_text):
    cwd = os.getcwd()
    rc_in = os.path.join(cwd, "rc.in")
    rc_out = os.path.join(cwd, "rc.out")

    with open(rc_in, "w", encoding="utf-8") as fh:
        fh.write(rc_text)

    # Preferred: the compiled Cython port -- no process spawn, no dependency
    # on the Fortran binary being present/executable.
    result = _run_cython_mol(rc_text)
    if result is not None:
        with open(rc_out, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(result[1])
        return result

    if random_gen and os.path.isfile(random_gen) and not random_gen.endswith(".py"):
        # Fall back to the original USPEX invocation:
        # `random_cell_mol < rc.in > rc.out`.
        try:
            with open(rc_in, "rb") as stdin_fh, open(rc_out, "wb") as stdout_fh:
                proc = subprocess.run(
                    [random_gen],
                    stdin=stdin_fh,
                    stdout=stdout_fh,
                    stderr=subprocess.STDOUT,
                    cwd=cwd,
                )
            with open(rc_out, "r", encoding="utf-8", errors="replace") as fh:
                out = fh.read()
            return proc.returncode, out
        except Exception:
            pass

    run_cmd = [sys.executable, random_gen] if str(random_gen).endswith(".py") else [random_gen]
    proc = subprocess.run(
        run_cmd, input=rc_text, text=True, capture_output=True, cwd=cwd,
        errors="replace",
    )
    return proc.returncode, proc.stdout if proc.stdout is not None else ""


def symope_310(nsym, num_ions, lat, min_d):
    ORG = G["ORG_STRUC"]
    spg_bindir = resource_path("FunctionFolder", "spacegroup")
    random_gen = os.path.join(spg_bindir, "random_cell_mol")
    if not os.path.exists(random_gen):
        random_gen = os.path.join(spg_bindir, "random_cell_mol.py")

    error_s = 0
    error_n = 0
    P = [1, 2, 3]
    PB = [1, 2, 3]
    permutation = [1, 2, 3]
    permutation_back = [1, 2, 3]

    # Optional lattice-axis permutation for selected P groups (same as original)
    if 15 < nsym < 75:
        s_group = spaceGroups(nsym)[0]
        if s_group == "P":
            list_perm = [(1, 2, 3), (3, 2, 1), (1, 3, 2), (2, 1, 3), (3, 1, 2), (2, 3, 1)]
            pick = np.random.randint(0, 6)
            permutation = list_perm[pick]
            if pick == 4:
                permutation_back = list_perm[5]
            elif pick == 5:
                permutation_back = list_perm[4]
            else:
                permutation_back = permutation

    lat = np.asarray(lat, dtype=float)

    # Determine lattice parameters
    if lat.size == 1:
        fix_lat = 0
        a = np.random.rand(3) + 0.5
        x = 0
        while x < 0.3:
            alpha_beta_gamma = (np.random.rand(3) * 120 + 30) * math.pi / 180
            ca, cb, cg = [math.cos(v) for v in alpha_beta_gamma]
            x = 1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg
        lat_params = np.concatenate([a, alpha_beta_gamma])
        ratio = lat[0] / det(latConverter(lat_params))
        lat_6 = np.zeros(6)
        lat_6[0:3] = a * ratio**(1/3)
        lat_6[3:6] = alpha_beta_gamma * 180 / math.pi
    elif lat.size == 9:
        fix_lat = 1
        lat_6 = latConverter(lat.reshape(3, 3)).reshape(-1)
        # NOTE: MATLAB leaves latConverter angles in radians here; convert to degrees
        lat6_rad = lat_6.copy()
        lat6_rad[3:6] = lat6_rad[3:6] * 180 / math.pi
        # permutation of lattice axes
        lat_perm = lat6_rad.copy()
        for axis in range(3):
            lat_perm[axis] = lat6_rad[permutation[axis] - 1]
        lat_perm = _apply_gamma_swap(lat_perm, permutation)
        lat_6 = lat_perm.copy()
    else:
        # Treat vector of six lattice parameters
        fix_lat = 1
        lat_perm = np.asarray(lat, dtype=float).copy()
        for axis in range(3):
            lat_perm[axis] = lat[permutation[axis] - 1]
        lat_6 = _apply_gamma_swap(lat_perm, permutation)

    # Remove zero-count types from numIons and minD columns/rows
    ions = np.asarray(num_ions, dtype=int).copy()
    min_d = np.asarray(min_d, dtype=float).copy() if min_d is not None else np.ones((len(ions), len(ions)))
    keep = np.where(ions > 0)[0] if ions.dtype != object else np.array([i for i, v in enumerate(ions) if v > 0])
    if len(keep) < len(ions):
        ions = ions[keep]
        min_d = min_d[np.ix_(keep, keep)]

    # Generate rc.in; matlab lat1(4:6) is degrees
    rc_lines = [f"{int(nsym):4d} ! space group"]
    angles = lat_6[3:6]
    if len(angles) == 0:
        angles = np.zeros(3)
    rc_lines.append(f"{lat_6[0]:6.3f} {lat_6[1]:6.3f} {lat_6[2]:6.3f} "
                    f"{angles[0]:6.3f} {angles[1]:6.3f} {angles[2]:6.3f} ! lattice")
    rc_lines.append(f"{len(ions):4d} ! number of types of atoms")
    rc_lines.append(" ".join(f"{int(v):4d}" for v in ions) + " ! number of atoms of each type")
    for i in range(min_d.shape[0]):
        rc_lines.append(" ".join(f"{min_d[i, j]:5.3f}" for j in range(min_d.shape[1])))
    rc_lines.append("! minimum distance between atoms")
    if ORG.fixRndSeed > 0:
        rc_lines.append(f"{int(round(np.random.rand()*1e6)):10d} {int(round(np.random.rand()*1e6)):10d} ! RandSeeds")

    rc_lines = [line.split("!")[0].rstrip() for line in rc_lines]
    rc_text = "\n".join(rc_lines) + "\n"

    # Mirror the MATLAB side effects: ``rc.in`` is the generator input and
    # ``rc.out`` is the parsed generator output (kept in CalcFoldTemp).
    cwd_now = os.getcwd()

    ier, out = _run_random_cell_mol(random_gen, rc_text)
    if out and not os.path.exists(os.path.join(cwd_now, "rc.out")):
        with open(os.path.join(cwd_now, "rc.out"), "w", encoding="utf-8", errors="replace") as fh:
            fh.write(out)

    # Parse number of site types (originally ``grep NUMBER rc.out | wc -l``)
    sites = len(re.findall(r"(?im)^\s*NUMBER\s*$", out))

    candidate = np.zeros((int(np.sum(num_ions)), 3))
    num_sites = np.zeros((sites, len(ions)))
    operation = None

    if out is not None:
        lines = out.splitlines()
        num_type = len(ions)
        f1 = int(math.ceil(num_type * num_type / 10))
        ptr = min(6 + f1, len(lines))

        # NUMBER blocks: each line containing NUMBER is followed by the
        # site-count line for that site type. Reproduce MATLAB's adjustment
        # of the first column.
        number_idx = [i for i, line in enumerate(lines) if line.strip().upper() == "NUMBER"]
        sumatom = 0
        num_sites = np.zeros((sites, len(ions)))
        for row_i, ni in enumerate(number_idx[:sites]):
            if ni + 1 >= len(lines):
                continue
            tmp = lines[ni + 1]
            vals = [float(v) for v in tmp.split()]
            if len(vals) == len(ions):
                num_sites[row_i, :] = vals
                if row_i > 0:
                    num_sites[row_i, 0] = float(np.sum(num_sites[row_i, :])) - sumatom
                sumatom += num_sites[row_i, 0]

        ptr = min(6 + f1, len(lines))

        # All lines starting with "operation" until cell parameters.
        operation_lines = []
        for line in lines:
            lower = line.strip().lower()
            if lower.startswith("operation"):
                parts = line.split()
                operation_lines.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif "cell parameters" in lower:
                break
        operation = np.asarray(operation_lines, dtype=float).reshape(-1, 3)

        # Cell-parameter line.
        cell_ptr = None
        for i, line in enumerate(lines):
            if "cell parameters" in line.lower():
                cell_ptr = i
                break
        if cell_ptr is None or "error" in lines[cell_ptr].lower():
            error_s = 1
        else:
            parts = lines[cell_ptr].split(":")[1].split()
            if len(parts) < 6 and cell_ptr + 1 < len(lines):
                parts = lines[cell_ptr + 1].split()
            cell = [float(x) for x in parts[:6]]

            # Atoms start after the "atoms" count line.
            aptr = cell_ptr
            header_seen = 0
            while aptr < len(lines) and header_seen < 2:
                if re.search(r"atomic parameters|atoms\s*$", lines[aptr], re.I):
                    header_seen += 1
                aptr += 1
            idx = 0
            while aptr < len(lines) and idx < int(np.sum(num_ions)):
                parts = lines[aptr].split()
                if len(parts) >= 5 and _is_number(parts[0]) and _is_number(parts[1]):
                    candidate[idx, :] = [float(parts[2]), float(parts[3]), float(parts[4])]
                    idx += 1
                aptr += 1
            lat2_deg = np.array(cell[:6])
            lat2 = lat2_deg.copy()
            lat2[3:6] = lat2[3:6] * math.pi / 180
            new_lattice = latConverter(lat2)
            error_s = 0
    if sites == 0 or error_s or "not possible" in out.lower():
        if "not possible" in out.lower():
            error_n = 1
        error_s = 1
        new_lattice = np.eye(3)

    # Reverse lattice permutation
    if not error_s and operation is not None:
        try:
            new_lat_6 = latConverter(new_lattice).reshape(-1)
            new_lat_2 = new_lat_6.copy()
            candidate1 = candidate.copy()
            for axis in range(3):
                new_lat_2[axis] = new_lat_6[permutation_back[axis] - 1]
                candidate[:, axis] = candidate1[:, permutation_back[axis] - 1]
            new_lattice = latConverter(new_lat_2)
            # symope_310.m:104-108 -- publish the axis permutation that was
            # applied.  GetOrientation/Operate_molecule need it to move the
            # molecules into the generator's frame, apply the symmetry
            # operation and bring them back; returning the identity here
            # (the previous behaviour) applies the operations in the wrong
            # frame whenever 15 < nsym < 75 with a P lattice.
            P = list(permutation)
            PB = list(permutation_back)
        except Exception:
            pass

    return candidate, new_lattice, num_sites, operation, error_s, error_n, list(P), list(PB)


def _apply_gamma_swap(lat6, permutation):
    lat6 = np.asarray(lat6, dtype=float).copy()
    if tuple(permutation) == (3, 2, 1):
        tmp = lat6.copy()
        lat6[3] = tmp[5]
        lat6[5] = tmp[3]
    return lat6


def _is_number(s):
    try:
        float(s)
        return True
    except Exception:
        return False
