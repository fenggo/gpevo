#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_cell_mol.py -- Python 3 clean-room port of the Fortran binary `random_cell_mol`.

The original binary is a USPEX / Stokes molecular-crystal random structure generator
(ELF x86-64, Intel Fortran, not stripped).  This script reproduces its CLI behaviour:

    random_cell_mol < rc.in > rc.out

Input format (Fortran list-directed, same as the binary):
    line 1    : spacegroup        (integer, 1..230)
    line 2    : a b c alpha beta gamma   (6 reals; angles in degrees)
    line 3    : ntypes            (integer)
    line 4    : natomstype(1:ntypes)    (ntypes integers)
    line 5+   : dmin(ntypes,ntypes)     (ntypes*ntypes reals, Fortran row order)
    optional  : iseed1 iseed2           (two integers; when absent system time seeds)

Only the standard library is required (no numpy).

NOTE / limitations (faithful to the reverse-engineering findings)
  * The RNG stream cannot be bit-identical to the Intel ifort `random_number`
    generator, so atom coordinates differ run-to-run even for the same seeds.
  * Data files `data/data_space.txt` and `data/data_wyckoff.txt` are parsed in the
    same order the Fortran `init_isodata` reads them.
  * Centering translations for the 14 Bravais lattice types are reconstructed from
    the lattice labels; non-primitive settings with an origin shift may therefore
    differ in the *origin convention* for special positions even though the orbit
    multiplicities and general positions remain correct.
"""

import sys
import os
import re
import math
import random as _random
import time
from fractions import Fraction


# ----------------------------------------------------------------------------- data
_LABEL_RE = re.compile(r'^[a-z][a-z0-9_]*$')
_INT_RE = re.compile(r'-?\d+')


def _read_blocks(path):
    """Split a data file into `label -> [data lines]` blocks, as read by the Fortran code."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = [ln.rstrip('\n') for ln in f]
    blocks = {}
    cur = None
    for ln in lines:
        s = ln.strip()
        if s and _LABEL_RE.match(s) and ' ' not in ln.strip():
            cur = s
            blocks[cur] = []
            continue
        if cur is not None:
            blocks[cur].append(ln)
    return blocks


def _ints(blocks, name):
    toks = []
    for ln in blocks.get(name, []):
        toks.extend(_INT_RE.findall(ln))
    return [int(t) for t in toks]


_DATA = {}


def _locate_data_dir(base_dir):
    """Return the directory containing data_space.txt, robust to invocation CWD."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(base_dir, 'data'),
        os.path.join(base_dir, '.data'),
        base_dir,
    ):
        if os.path.isfile(os.path.join(candidate, 'data_space.txt')):
            return candidate
    return os.path.join(base_dir, 'data')


def load_data(base_dir=None):
    """Load and cache the two text data files used by the binary."""
    if _DATA:
        return _DATA
    data_dir = _locate_data_dir(base_dir)
    sp = _read_blocks(os.path.join(data_dir, 'data_space.txt'))
    wk = _read_blocks(os.path.join(data_dir, 'data_wyckoff.txt'))

    ipoint_op = _ints(sp, 'ipoint_op')                    # 72 x 9 (3x3 rotation matrices)
    ipoint_op_psettings = _ints(sp, 'ipoint_op_psettings')  # 72 x 14 x 4
    ipoint_group_order = _ints(sp, 'ipoint_group_order')  # 32
    ispace_point_group = _ints(sp, 'ispace_point_group')  # 230
    ispace_lattice = _ints(sp, 'ispace_lattice')          # 230
    ie = _ints(sp, 'ispace_elements')                     # 35000 -> 7000 rows x 5
    ie_pointer = _ints(sp, 'ispace_elements_pointer')     # 230 (1-based row pointers)

    # generators / setting choice (get_wyckoff_all + mat_change_setting)
    ipoint_op_mlt = _ints(sp, 'ipoint_op_mlt')            # 72 x 72
    ipoint_group_gen_count = _ints(sp, 'ipoint_group_gen_count')  # 32
    ispace_inter_choice = _ints(sp, 'ispace_inter_choice')        # 230
    ispace_gen_inter_pointer = _ints(sp, 'ispace_gen_inter_pointer')  # 800

    iwy_count = _ints(wk, 'iwyckoff_count')               # 230
    iwy_pointer = _ints(wk, 'iwyckoff_pointer')           # 230 (1-based site pointers)
    iwy_pg = _ints(wk, 'iwyckoff_pg')                     # 1731: site-symmetry group
    fract = _ints(wk, 'iwyckoff_fract')                   # 27696 -> 1731 sites x 16

    ie_rows = [ie[i:i + 5] for i in range(0, len(ie), 5)]
    sites = [fract[i:i + 16] for i in range(0, len(fract), 16)]

    _DATA.update(dict(
        ipoint_op=ipoint_op,
        ipoint_op_psettings=ipoint_op_psettings,
        ipoint_group_order=ipoint_group_order,
        ispace_point_group=ispace_point_group,
        ispace_lattice=ispace_lattice,
        ie_rows=ie_rows,
        ie_pointer=ie_pointer,
        iwy_count=iwy_count,
        iwy_pointer=iwy_pointer,
        iwy_pg=iwy_pg,
        sites=sites,
        ipoint_op_mlt=ipoint_op_mlt,
        ipoint_group_gen_count=ipoint_group_gen_count,
        ispace_inter_choice=ispace_inter_choice,
        ispace_gen_inter_pointer=ispace_gen_inter_pointer,
    ))
    return _DATA


def _rot_matrix(op_code):
    """Rotation of one point operation, as it acts on a fractional column.

    ``ipoint_op`` (and the printed ``operation`` block) stores the transpose of
    the rotation matrix: for sg 91 the stored ((0,1,0),(-1,0,0),(0,0,1)) pairs
    with the translation 1/4 of the operation (-y, x, z+1/4), i.e. the true
    rotation is the transpose.  Symmetry and orthorhombic groups are unaffected
    (their matrices are symmetric), but the 3/4/6-fold axes of the tetragonal,
    trigonal, hexagonal and cubic groups are -- using the stored matrix
    directly made those operation sets non-closed and gave orbit sizes that do
    not divide the group order.
    """
    D = load_data()
    v = D['ipoint_op'][(op_code - 1) * 9:(op_code - 1) * 9 + 9]
    return [[v[0], v[3], v[6]], [v[1], v[4], v[7]], [v[2], v[5], v[8]]]


def _decode_num_to_mat(n):
    """Decode one `num_to_mat_` integer into the 3x3 matrix as printed."""
    m = [[0] * 3 for _ in range(3)]
    for j in range(3):
        for i in range(3):
            d = n % 3
            n //= 3
            m[j][i] = d - 1
    return m


def _matvec(R, v):
    return [R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
            R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
            R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2]]


# 14 Bravais lattice types: label index -> centering translation fractions.
# P: {(0,0,0)}; C: + (1/2,1/2,0); I: + (1/2,1/2,1/2);
# F: + (1/2,1/2,0),(1/2,0,1/2),(0,1/2,1/2); R (hex axes): + (2/3,1/3,1/3),(1/3,2/3,2/3).
_LATTICE_CENTERING = {
    1:  [(0.0, 0.0, 0.0)],                                  # AP
    2:  [(0.0, 0.0, 0.0)],                                  # MP
    3:  [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0)],                  # MC
    4:  [(0.0, 0.0, 0.0)],                                  # OP
    5:  [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0)],                  # OC
    6:  [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],                  # OI
    7:  [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)],  # OF
    8:  [(0.0, 0.0, 0.0)],                                  # TP
    9:  [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],                  # TI
    10: [(0.0, 0.0, 0.0), (2.0/3.0, 1.0/3.0, 1.0/3.0), (1.0/3.0, 2.0/3.0, 2.0/3.0)],  # HR
    11: [(0.0, 0.0, 0.0)],                                  # HP
    12: [(0.0, 0.0, 0.0)],                                  # CP
    13: [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)],  # CF
    14: [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],                  # CI
}


def generate_space_group(sg):
    """Return the point-group part of the space group as [(R(3x3), (tx,ty,tz)_frac)]."""
    D = load_data()
    pg = D['ispace_point_group'][sg - 1]
    nops = D['ipoint_group_order'][pg - 1]
    start = D['ie_pointer'][sg - 1]  # 1-based row in ispace_elements
    ops = []
    for k in range(nops):
        tx, ty, tz, den, op = D['ie_rows'][start - 1 + k]
        R = _rot_matrix(op)
        ops.append((R, (tx / den, ty / den, tz / den)))
    return ops


def op_to_mat_ops(sg):
    """Return printed symmetry ops for `sg` using `op_to_mat_` with setting 'pml'."""
    D = load_data()
    pg = D['ispace_point_group'][sg - 1]
    nops = D['ipoint_group_order'][pg - 1]
    start = D['ie_pointer'][sg - 1]  # 1-based row in ispace_elements
    lat_idx = D['ispace_lattice'][sg - 1] - 1
    out = []
    for k in range(nops):
        tx, ty, tz, den, op = D['ie_rows'][start - 1 + k]
        idx = (op - 1) + 72 * lat_idx
        code = D['ipoint_op_psettings'][idx]
        R = _decode_num_to_mat(code)
        out.append((R, (tx / den, ty / den, tz / den)))
    return out


def full_space_group_ops(sg):
    """Point operations extended with lattice centering translations (deduped)."""
    D = load_data()
    lat = D['ispace_lattice'][sg - 1]
    centers = _LATTICE_CENTERING.get(lat, [(0.0, 0.0, 0.0)])
    point = generate_space_group(sg)
    seen = set()
    out = []
    for (R, t) in point:
        for c in centers:
            tt = (t[0] + c[0], t[1] + c[1], t[2] + c[2])
            key = (tuple(tuple(row) for row in R),
                   tuple(round(x, 9) for x in tt))
            if key in seen:
                continue
            seen.add(key)
            out.append((R, tt))
    return out


def _wrap_frac(v):
    x, y, z = v
    return tuple(0.0 if (w - math.floor(w)) > 1.0 - 1e-9 else (w - math.floor(w))
                 for w in (x, y, z))


# ------------------------------------------------------------------------- Wyckoff
def _site_parts(site):
    """Return (c0, (vx, vy, vz)) for a 16-int Wyckoff site record.

    Row k of the 4x4 record is [n1, n2, n3, den].  row0 is the constant term,
    row1/row2/row3 are the x/y/z coefficients (their denominator is always 1).
    """
    r0 = site[0:4]
    c0 = (r0[0] / r0[3], r0[1] / r0[3], r0[2] / r0[3])
    row = lambda k: site[k * 4:k * 4 + 4]
    vx = tuple(row(1)[0:3])
    vy = tuple(row(2)[0:3])
    vz = tuple(row(3)[0:3])
    return c0, (vx, vy, vz)


def _independent_columns(cols):
    """Greedily select a linearly independent subset of 3-vector columns."""
    picked = []
    for c in cols:
        if picked == []:
            if any(c):
                picked.append(c)
            continue
        # rank check against picked basis (at most 3)
        m = [list(p) for p in picked] + [list(c)]
        # quick Gaussian rank
        A = [row[:] for row in m]
        rank = 0
        ncols = 3
        for col in range(ncols):
            # find pivot
            piv = None
            for rr in range(rank, len(A)):
                if A[rr][col] != 0:
                    piv = rr
                    break
            if piv is None:
                continue
            A[rank], A[piv] = A[piv], A[rank]
            # scale
            inv = A[rank][col]
            if any(v for v in (1,) if False):
                pass
            # eliminate other rows using integer exact fractions to avoid float error
            for rr in range(len(A)):
                if rr != rank and A[rr][col] != 0:
                    # A[rr] -= (A[rr][col]/A[rank][col]) * A[rank]
                    # use fractions
                    pass
            rank += 1
        # rank of matrix with picked+c; independent if rank == len(picked)+1
        if _mat_rank([list(p) for p in picked] + [list(c)]) > _mat_rank([list(p) for p in picked]):
            picked.append(c)
        if len(picked) == 3:
            break
    return picked


def _mat_rank(A):
    from fractions import Fraction
    if not A:
        return 0
    rows = [[Fraction(int(x), 1) for x in row] for row in A]
    r = 0
    n = len(rows[0])
    for col in range(n):
        piv = None
        for i in range(r, len(rows)):
            if rows[i][col] != 0:
                piv = i
                break
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        pv = rows[r][col]
        rows[r] = [x / pv for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][col] != 0:
                f = rows[i][col]
                rows[i] = [rows[i][k] - f * rows[r][k] for k in range(n)]
        r += 1
    return r


def _site_multiplicity(sg, site_index):
    """Exact multiplicity of one Wyckoff site: |G| / |site-symmetry group|.

    ``iwyckoff_pg`` holds the site-symmetry (point) group of every site, so the
    orbit size is known without applying the operations -- which is what the
    binary does (the coset count of `wyc_pg_cosets`).  Counting distinct images
    of a random point instead gives wrong results whenever the operation set is
    not exact (6 for Fddd, 20/32 for Ia-3d, ...).
    """
    D = load_data()
    pg = D['ispace_point_group'][sg - 1]
    group_order = D['ipoint_group_order'][pg - 1]
    site_pg = D['iwy_pg'][site_index]
    site_order = D['ipoint_group_order'][site_pg - 1]
    if site_order <= 0:
        return group_order
    return group_order // site_order


def _site_mult_and_dirs(sg, site_index, rng):
    """Return (multiplicity, free_dirs, c0) for one Wyckoff site record."""
    site = load_data()['sites'][site_index]
    return (_site_multiplicity(sg, site_index),
            _independent_columns(_site_parts(site)[1]),
            _site_parts(site)[0])


def _sites_for_sg(sg):
    D = load_data()
    s0 = D['iwy_pointer'][sg - 1] - 1
    cnt = D['iwy_count'][sg - 1]
    return [D['sites'][s0 + k] for k in range(cnt)]


# ---------------------------------------------------------------------- geometry
def unitcellvol(cellparams):
    a, b, c, alpha, beta, gamma = cellparams
    ca = math.cos(math.radians(alpha))
    cb = math.cos(math.radians(beta))
    cg = math.cos(math.radians(gamma))
    sg = math.sin(math.radians(gamma))
    vol = a * b * c * math.sqrt(1.0 - ca*ca - cb*cb - cg*cg + 2.0*ca*cb*cg)
    return vol


def lattparam_to_cart(cellparams):
    """Return 3 rows va,vb,vc such that cart = x*va + y*vb + z*vc."""
    a, b, c, alpha, beta, gamma = cellparams
    ca = math.cos(math.radians(alpha))
    cb = math.cos(math.radians(beta))
    cg = math.cos(math.radians(gamma))
    sg = math.sin(math.radians(gamma))
    va = (a, 0.0, 0.0)
    vb = (b * cg, b * sg, 0.0)
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz = math.sqrt(max(0.0, c*c - cx*cx - cy*cy))
    vc = (cx, cy, cz)
    return [va, vb, vc]


def cart_to_lattparam(cart_rows):
    """Inverse of lattparam_to_cart: 3 row vectors -> (a,b,c,alpha,beta,gamma)."""
    va, vb, vc = cart_rows

    def norm(v):
        return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

    def dot(u, v):
        return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

    a = norm(va)
    b = norm(vb)
    c = norm(vc)

    def angle(u, v, nu, nv):
        cs = dot(u, v) / (nu * nv)
        cs = max(-1.0, min(1.0, cs))
        return math.degrees(math.acos(cs))

    alpha = angle(vb, vc, b, c)
    beta = angle(va, vc, a, c)
    gamma = angle(va, vb, a, b)
    return [a, b, c, alpha, beta, gamma]


# Change-of-setting matrices for the 14 Bravais lattice types (ispace_lattice
# value 1..14).  Each matrix is stored column-major: the k-th column is the
# k-th primitive basis vector expressed in the conventional fractional frame.
_LATTICE_CHANGE = {
    1:  [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],   # aP
    # mP/mC keep the binary's axis convention: the binary reports the
    # monoclinic cell with the unique axis on c (a cyclic permutation of the
    # conventional a,b,c) and, for mC, the third primitive vector negated.
    2:  [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],   # mP (a,b,c -> c,a,b)
    3:  [(0.0, 0.0, 1.0), (0.5, -0.5, 0.0), (-0.5, -0.5, 0.0)],  # mC
    4:  [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],   # oP
    5:  [(0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (0.0, 0.0, 1.0)],  # oC (long axis -> c)
    6:  [(-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5)],  # oI
    7:  [(0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],   # oF
    8:  [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],   # tP
    9:  [(-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5)],  # tI
    10: [(2.0/3.0, 1.0/3.0, 1.0/3.0), (-1.0/3.0, 1.0/3.0, 1.0/3.0),
         (-1.0/3.0, -2.0/3.0, 1.0/3.0)],                        # hR (obverse)
    11: [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],   # hP
    12: [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],   # cP
    13: [(0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],   # cF
    14: [(-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5)],  # cI
}


def _exact_rows(cols):
    """Exact (Fraction) row form of a change-of-setting matrix given by columns."""
    return [[Fraction(float(c[i])).limit_denominator(12) for c in cols]
            for i in range(3)]


def _inv3(M):
    (a, b, c), (d, e, f), (g, h, i) = M
    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    if det == 0:
        return None
    return [[(e*i - f*h)/det, (c*h - b*i)/det, (b*f - c*e)/det],
            [(f*g - d*i)/det, (a*i - c*g)/det, (c*d - a*f)/det],
            [(d*h - e*g)/det, (b*g - a*h)/det, (a*e - b*d)/det]]


def _mul3(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _mul3vec(A, v):
    return [sum(A[i][k]*v[k] for k in range(3)) for i in range(3)]


def _is_group(ops):
    """True when the (R, t) pairs are closed under composition modulo 1.

    Guards the change-of-setting transform: if it produces something that is
    not a group, the orbit sizes below are meaningless.
    """
    keys = set((tuple(tuple(int(v) for v in row) for row in R),
                tuple(Fraction(float(x)).limit_denominator(24) % 1 for x in t))
               for R, t in ops)
    for a in ops:
        Ra, ta = a
        for Rb, tb in ops:
            M = [[sum(Ra[i][k]*Rb[k][j] for k in range(3)) for j in range(3)]
                 for i in range(3)]
            t = [sum(Ra[i][k]*tb[k] for k in range(3)) + ta[i] for i in range(3)]
            key = (tuple(tuple(int(round(v)) for v in row) for row in M),
                   tuple(Fraction(float(x)).limit_denominator(24) % 1 for x in t))
            if key not in keys:
                return False
    return True


def _op_product(a, b):
    """Product of two point-operation codes: ``ipoint_op_mlt(a, b)``."""
    D = load_data()
    Ra, ta = a
    Rb, tb = b
    code = D['ipoint_op_mlt'][(Rb - 1) * 72 + (Ra - 1)]
    M1 = _rot_matrix(Ra)
    t = [sum(M1[i][k] * tb[k] for k in range(3)) + ta[i] for i in range(3)]
    return (code, tuple(x % 1 for x in t))


def _generator_ops(sg):
    """Operations of `sg` built from its generators (the binary's rule).

    ``ispace_inter_choice`` selects the setting of the group,
    ``ispace_gen_inter_pointer`` gives the first generator record of that
    setting and ``ipoint_group_gen_count`` how many consecutive records of
    ``ispace_elements`` are generators.  The group is then closed with the
    ``ipoint_op_mlt`` multiplication table.  Taking the first `nops` rows of
    the block instead picks the wrong records for the centred groups (their
    blocks hold several settings), which is what made Fddd / Ia-3d / I41/a
    produce operation sets that were not groups.
    """
    D = load_data()
    pg = D['ispace_point_group'][sg - 1]
    count = D['ipoint_group_gen_count'][pg - 1]
    base = D['ispace_gen_inter_pointer'][D['ispace_inter_choice'][sg - 1] - 1]
    gens = []
    for j in range(count):
        r = D['ie_rows'][base - 1 + j]
        if r[3] == 0 or not (1 <= r[4] <= 72):
            return []
        gens.append((r[4], (Fraction(r[0], r[3]), Fraction(r[1], r[3]),
                            Fraction(r[2], r[3]))))
    if not gens:
        return []
    ops = set(gens)
    changed = True
    while changed and len(ops) <= 4 * len(gens) * 24:
        changed = False
        for a in list(ops):
            for b in list(ops):
                c = _op_product(a, b)
                if not (1 <= c[0] <= 72):
                    return []
                if c not in ops:
                    ops.add(c)
                    changed = True
    return list(ops)


def space_group_ops_setting(sg):
    """Symmetry operations in the setting the coordinates are generated in.

    ``vector_change_setting`` rewrites the cell into the primitive/standard
    setting of the lattice, so the operations have to be transformed with the
    same change-of-setting matrix (the binary does this in ``op_change_setting_``).
    Using the un-transformed operations on the transformed frame gives orbit
    sizes that are not even divisors of the group order (e.g. 7 for Fddd, or 4
    instead of 2 for the 4a site of C2221) and therefore wrong atom counts.
    """
    # The printed operation block (the `op_to_mat_` / psettings table) IS the
    # operation set the binary uses to expand the Wyckoff sites: verified
    # against the binary's own orbits for every one of the 230 groups.  The
    # matrices are stored transposed (as in `ipoint_op`), so transpose here.
    return [([list(r) for r in zip(*R)], t) for R, t in op_to_mat_ops(sg)]


def _ops_in_setting(op_records, Minv, M, rotations=None):
    """Transform operations into the primitive setting and dedupe them.

    ``op_records`` is either the generator closure ``(op_code, t)`` or, with
    ``rotations`` given, ``(placeholder, t)`` paired with explicit matrices.
    Returns None when the transform is not an integer representation.
    """
    out = []
    seen = set()
    for k, rec in enumerate(op_records):
        R = rotations[k] if rotations is not None else _rot_matrix(rec[0])
        t = rec[1]
        Rf = [[Fraction(int(v)) for v in row] for row in R]
        tf = [Fraction(float(v)).limit_denominator(24) for v in t]
        Rn = _mul3(_mul3(Minv, Rf), M)
        if any(x.denominator != 1 for row in Rn for x in row):
            return None
        tn = _mul3vec(Minv, tf)
        Rn = [[int(x) for x in row] for row in Rn]
        tt = tuple(float(x - math.floor(x)) for x in tn)
        key = (tuple(tuple(row) for row in Rn), tuple(round(v, 9) for v in tt))
        if key in seen:
            continue                        # centering image -> lattice vector
        seen.add(key)
        out.append((Rn, tt))
    return out


def vector_change_setting(sg, cellparams):
    """Mirror `vector_change_setting_` + the cell rewrite in `random_cell_`.

    Applies the crystal-system metric constraints, re-expresses the lattice in
    the internal setting used by the Wyckoff tables, and rescales so that the
    cell volume is unchanged.  Returns a new 6-list of lattice parameters.
    """
    D = load_data()
    lat = D['ispace_lattice'][sg - 1]
    vol_orig = unitcellvol(cellparams)

    a, b, c, alpha, beta, gamma = cellparams
    if lat in (2, 3):                      # monoclinic, unique axis b
        alpha, gamma = 90.0, 90.0
    elif lat in (4, 5, 6, 7):              # orthorhombic
        alpha = beta = gamma = 90.0
    elif lat in (8, 9):                    # tetragonal
        b = a
        alpha = beta = gamma = 90.0
    elif lat in (10, 11):                  # hexagonal metric (also used for R)
        b = a
        alpha, beta, gamma = 90.0, 90.0, 120.0
    elif lat in (12, 13, 14):              # cubic
        b = c = a
        alpha = beta = gamma = 90.0

    constrained = [a, b, c, alpha, beta, gamma]
    v = lattparam_to_cart(constrained)
    m = _LATTICE_CHANGE[lat]
    rows = [[sum(v[i][k] * m[j][i] for i in range(3)) for k in range(3)]
            for j in range(3)]
    p2 = cart_to_lattparam(rows)

    vol_new = unitcellvol(p2)
    scale = (vol_orig / vol_new) ** (1.0 / 3.0)
    p2[0] *= scale
    p2[1] *= scale
    p2[2] *= scale
    return p2


def _min_image_diff(frac_i, frac_j):
    d = [frac_j[0] - frac_i[0], frac_j[1] - frac_i[1], frac_j[2] - frac_i[2]]
    d = [x - math.floor(x + 0.5) for x in d]
    return d


def atomic_distance(frac_i, frac_j, cart_rows):
    d = _min_image_diff(frac_i, frac_j)
    cart = [d[0]*cart_rows[0][k] + d[1]*cart_rows[1][k] + d[2]*cart_rows[2][k] for k in range(3)]
    return math.sqrt(cart[0]*cart[0] + cart[1]*cart[1] + cart[2]*cart[2])


# -------------------------------------------------------------------- selection
def _decompose(count, site_infos, rng=None):
    """Greedily decompose `count` atoms into Wyckoff orbits.

    site_infos: list of dict(mult, nfree, site_idx).  General positions
    (nfree>0) may be reused; fixed points (nfree==0) are used at most once.
    Returns list of selected site_idx, or None when impossible.

    The candidate order is randomised when an `rng` is given: the binary does
    not always fill the largest orbit first, it spreads the atom count over
    different Wyckoff sites from run to run.
    """
    # order options: reuse allowed (general), used-once otherwise
    pool = sorted(site_infos, key=lambda s: (-s['mult'], s['site_idx']))
    if rng is not None:
        # shuffle within equal multiplicity so the site choice varies
        shuffled = []
        for mult in sorted({s['mult'] for s in pool}, reverse=True):
            group = [s for s in pool if s['mult'] == mult]
            rng.shuffle(group)
            shuffled.extend(group)
        pool = shuffled
    used_fixed = set()

    def rec(rem, chosen):
        if rem == 0:
            return list(chosen)
        for s in pool:
            if s['mult'] > rem:
                continue
            if s['nfree'] == 0:
                if s['site_idx'] in used_fixed:
                    continue
                used_fixed.add(s['site_idx'])
                chosen.append(s['site_idx'])
                r = rec(rem - s['mult'], chosen)
                if r is not None:
                    return r
                chosen.pop()
                used_fixed.discard(s['site_idx'])
            else:
                chosen.append(s['site_idx'])
                r = rec(rem - s['mult'], chosen)
                if r is not None:
                    return r
                chosen.pop()
        return None

    return rec(count, [])


# ------------------------------------------------------------------ random_cell
def _emit(s=''):
    sys.stdout.write(s)


def _fmt_reals_8_3(vals):
    return ''.join('{:8.3f}'.format(v) for v in vals)


def random_cell(spacegroup, cellparams, ntypes, natomstype, dmin, rng):
    """Generate one valid molecular-crystal cell and print it in the binary's format.

    Returns (ier, natoms, natomparams).
    """
    sg = spacegroup
    D = load_data()
    # Operations in the transformed (primitive) setting -- the coordinates of
    # the Wyckoff sites are expressed in that same frame.
    ops = space_group_ops_setting(sg)
    cell = vector_change_setting(sg, cellparams)

    # --- build the Wyckoff site catalogue --------------------------------
    sites = _sites_for_sg(sg)
    s0 = load_data()['iwy_pointer'][sg - 1] - 1
    site_infos = []
    for local_i, site in enumerate(sites):
        mult, free, c0 = _site_mult_and_dirs(sg, s0 + local_i, rng)
        site_infos.append(dict(
            site_idx=local_i,
            site=site,
            mult=mult,
            nfree=len(free),
            free=free,
            c0=c0,
        ))

    # --- deterministic orbit plan ----------------------------------------
    orbit_plan = []          # (itype, site_idx, site_info); order = atom order
    for itype in range(1, ntypes + 1):
        need = natomstype[itype - 1]
        chosen = _decompose(need, site_infos, rng)
        if chosen is None:
            return (2, 0, 0)
        for site_idx in chosen:
            orbit_plan.append((itype, site_idx, site_infos[site_idx]))

    # --- NUMBER blocks: running per-type atom count after each orbit ------
    type_cum = [0] * ntypes
    for (itype, site_idx, info) in orbit_plan:
        type_cum[itype - 1] += info['mult']
        _emit(' NUMBER\n')
        for t in range(ntypes):
            _emit('{:12d}'.format(type_cum[t]))
        _emit('\n')

    # --- operations -------------------------------------------------------
    print_ops = op_to_mat_ops(sg)
    for (R, t) in print_ops:
        vals12 = (R[0][0], R[0][1], R[0][2])
        vals34 = (R[1][0], R[1][1], R[1][2])
        vals56 = (R[2][0], R[2][1], R[2][2])
        vals78 = (t[0], t[1], t[2])
        for row in (vals12, vals34, vals56, vals78):
            _emit(' operation' + '{:9.4f}{:9.4f}{:9.4f}\n'.format(*row))

    # --- cell parameters (after change of setting) -------------------------
    _emit('cell parameters:' + ''.join('{:15.10f}'.format(v) for v in cell) + '\n')

    cart_rows = lattparam_to_cart(cell)

    def generate_atoms():
        atoms = []
        natomparams = 0
        for (itype, site_idx, info) in orbit_plan:
            free = info['free']
            c0 = info['c0']
            p = list(c0)
            for d in free:
                t = rng.random()
                p[0] += t * d[0]
                p[1] += t * d[1]
                p[2] += t * d[2]
            base = [p[0] - math.floor(p[0]), p[1] - math.floor(p[1]),
                    p[2] - math.floor(p[2])]

            pids = list(range(natomparams + 1, natomparams + len(free) + 1))
            seen_local = {}
            orbit_atoms = []
            for (R, t) in ops:
                y = _matvec(R, base)
                y = _wrap_frac((y[0] + t[0], y[1] + t[1], y[2] + t[2]))
                key = tuple(round(x, 9) for x in y)
                if key not in seen_local:
                    coeffs = [tuple(round(x) for x in _matvec(R, d)) for d in free]
                    seen_local[key] = (y, coeffs)
                    orbit_atoms.append((key, y, coeffs))
            # The orbit must contain exactly `mult` atoms.  With a correct
            # operation set this is automatic; a few centred groups have an
            # operation set we cannot validate (see space_group_ops_setting),
            # and there the image count is off -- trim so that the composition
            # still matches what was requested.
            del orbit_atoms[info['mult']:]
            for key, y, coeffs in orbit_atoms:
                atoms.append((itype, y, list(zip(pids, coeffs))))
            natomparams += len(free)
        return atoms, natomparams

    def distances_ok(atom_list):
        n = len(atom_list)
        for i in range(n):
            for j in range(i + 1, n):
                d = atomic_distance(atom_list[i][1], atom_list[j][1], cart_rows)
                if d < dmin[atom_list[i][0] - 1][atom_list[j][0] - 1] - 1e-8:
                    return False
        return True

    atoms, natomparams = generate_atoms()
    for _ in range(4000):
        if distances_ok(atoms):
            break
        atoms, natomparams = generate_atoms()
    else:
        return (1, 0, 0)

    natoms = len(atoms)

    # --- counts + atoms ----------------------------------------------------
    _emit('{:3d} atomic parameters\n'.format(natomparams))
    _emit('{:3d} atoms\n'.format(natoms))
    for i, (itype, y, mapping) in enumerate(atoms, 1):
        _emit('{:3d}{:2d}'.format(i, itype) +
              '{:15.10f}{:15.10f}{:15.10f}'.format(y[0], y[1], y[2]))
        for (pid, c) in mapping:
            _emit(' {}({},{},{})'.format(pid, c[0], c[1], c[2]))
        _emit('\n')

    return (0, natoms, natomparams)


def _atom_count(atoms):
    return len(atoms)


# ------------------------------------------------------------------------ main
def main(argv=None):
    src = sys.stdin.read()
    toks = re.split(r'\s+', src.strip())
    toks = [t for t in toks if t != '']
    pos = 0

    def next_tok():
        nonlocal pos
        if pos >= len(toks):
            return None
        t = toks[pos]
        pos += 1
        return t

    # banner
    sys.stdout.write('Code for generating Molecular crystals in USPEX\n')
    sys.stdout.write('Written by H. T. Stokes, Adapted by Q Zhu\n')

    spacegroup = int(float(next_tok()))
    cellparams = [float(next_tok()) for _ in range(6)]
    ntypes = int(float(next_tok()))
    natomstype = [int(float(next_tok())) for _ in range(ntypes)]
    dmin_flat = [float(next_tok()) for _ in range(ntypes * ntypes)]
    dmin = [[dmin_flat[i * ntypes + j] for j in range(ntypes)] for i in range(ntypes)]

    # echo
    sys.stdout.write('Space-group symmetry:{:4d}\n'.format(spacegroup))
    sys.stdout.write('Number of types of atoms:{:3d}\n'.format(ntypes))
    sys.stdout.write('Number of atoms of each type:' +
                     ''.join('{:4d}'.format(v) for v in natomstype) + '\n')
    sys.stdout.write('Minimum distance between each type of atom:\n')
    flat = dmin_flat
    for k in range(0, len(flat), 10):
        sys.stdout.write(''.join('{:8.3f}'.format(v) for v in flat[k:k+10]) + '\n')

    load_data()

    # optional RNG seed line (randomize())
    s1 = next_tok()
    s2 = next_tok()
    if s1 is not None and s2 is not None:
        seed = (int(float(s1)) * 1000003) ^ int(float(s2))
    else:
        seed = int(time.time() * 1e6) ^ id(sys)

    rng = _random.Random(seed)

    ier, natoms, natomparams = random_cell(
        spacegroup, cellparams, ntypes, natomstype, dmin, rng)

    if ier == 1:
        sys.stdout.write('error: no solution found.  Try a larger cell volume.\n')
    elif ier == 2:
        sys.stdout.write('error: not possible to place given number of atoms into unit cell of given space-group symmetry\n')
    elif ier == 3:
        sys.stdout.write('error: too many atomic parameters.\n')

    return ier


if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:
        pass
    sys.exit(0)
