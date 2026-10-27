# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""
uspex_fast_core.pyx - USPEX molecular fingerprint acceleration core.

Cython implementation of makeMatrices + fingerprint_calc for computing
the USPEX structural fingerprint (order, fing, atom_fing) from a crystal
structure.  Replaces the Octave hot path with a single nogil loop.

Build via ``pip install .`` (uses setup.py cythonize).
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs, exp, ceil, M_PI, log, pow

ctypedef np.float64_t f64_t
ctypedef np.int32_t   i32_t


# ═══════════════════════════════════════════════════════════════
# 1. 距离矩阵构建 (makeMatrices) — 返回原子索引
# ═══════════════════════════════════════════════════════════════

cdef inline f64_t _dist2(f64_t x, f64_t y, f64_t z,
                          f64_t a11, f64_t a12, f64_t a13,
                          f64_t a21, f64_t a22, f64_t a23,
                          f64_t a31, f64_t a32, f64_t a33) noexcept nogil:
    cdef f64_t cx = x*a11 + y*a21 + z*a31
    cdef f64_t cy = x*a12 + y*a22 + z*a32
    cdef f64_t cz = x*a13 + y*a23 + z*a33
    return cx*cx + cy*cy + cz*cz


def build_distance_matrix(
    np.ndarray[f64_t, ndim=2] coords,     # (N, 3) 分数坐标
    np.ndarray[f64_t, ndim=2] lattice,    # (3, 3) 晶格
    f64_t Rmax,
    np.ndarray[i32_t, ndim=1] numIons,    # (n_species,)
    np.ndarray[i32_t, ndim=1] atomType,   # (n_species,)
):
    """构建距离矩阵，返回距离 + 原子索引 + 类型索引

    与 Octave makeMatrices.m 对应：
      dist_matrix(j, i) = 近邻 j 到晶胞内原子 i 的距离
      typ_i(j) = 近邻 j 所属原子类型 (1-indexed)
      typ_j(j) = 晶胞内原子 i 所属原子类型 (1-indexed)

    但我们用扁平数组 + cc_idx/bc_idx 替代二维矩阵，
    避免 Octave vertcat 的 O(n²) 内存重分配。

    Returns:
        dist_array: (n_pairs,)  距离值
        cc_idx:     (n_pairs,)  current_cell 原子索引 (0-indexed)
        bc_idx:     (n_pairs,)  basic_cell 原子索引 (0-indexed)
        typ_i_arr:  (n_pairs,)  current_cell 类型 (1-indexed)
        typ_j_arr:  (n_pairs,)  basic_cell 类型 (1-indexed)
        N_out:      (N,)        每原子所属类型的原子数
        V:          float       晶胞体积
        N:          int         原子数
    """
    cdef i32_t N = coords.shape[0]
    cdef i32_t n_species = numIons.shape[0]

    # typ_i_map: 原子索引 -> 类型 (1-indexed)
    cdef np.ndarray[i32_t, ndim=1] typ_i_map = np.zeros(N, dtype=np.int32)
    cdef np.ndarray[f64_t, ndim=1] N_out = np.zeros(N, dtype=np.float64)
    cdef i32_t idx = 0, sp, a
    for sp in range(n_species):
        for a in range(numIons[sp]):
            if idx < N:
                typ_i_map[idx] = sp + 1
                N_out[idx] = <f64_t>numIons[sp]
                idx += 1

    # 预计算 13 个格子向量长度
    cdef f64_t lv[3][3]
    cdef i32_t vi
    for vi in range(3):
        lv[0][vi] = lattice[0, vi]
        lv[1][vi] = lattice[1, vi]
        lv[2][vi] = lattice[2, vi]

    cdef f64_t v_abs[13]
    v_abs[0] = sqrt(lv[0][0]**2 + lv[0][1]**2 + lv[0][2]**2)
    v_abs[1] = sqrt(lv[1][0]**2 + lv[1][1]**2 + lv[1][2]**2)
    v_abs[2] = sqrt(lv[2][0]**2 + lv[2][1]**2 + lv[2][2]**2)
    v_abs[3] = sqrt((lv[0][0]+lv[1][0])**2+(lv[0][1]+lv[1][1])**2+(lv[0][2]+lv[1][2])**2)
    v_abs[4] = sqrt((lv[0][0]-lv[1][0])**2+(lv[0][1]-lv[1][1])**2+(lv[0][2]-lv[1][2])**2)
    v_abs[5] = sqrt((lv[0][0]+lv[2][0])**2+(lv[0][1]+lv[2][1])**2+(lv[0][2]+lv[2][2])**2)
    v_abs[6] = sqrt((lv[0][0]-lv[2][0])**2+(lv[0][1]-lv[2][1])**2+(lv[0][2]-lv[2][2])**2)
    v_abs[7] = sqrt((lv[2][0]+lv[1][0])**2+(lv[2][1]+lv[1][1])**2+(lv[2][2]+lv[1][2])**2)
    v_abs[8] = sqrt((lv[2][0]-lv[1][0])**2+(lv[2][1]-lv[1][1])**2+(lv[2][2]-lv[1][2])**2)
    v_abs[9] = sqrt((lv[0][0]+lv[1][0]+lv[2][0])**2+(lv[0][1]+lv[1][1]+lv[2][1])**2+(lv[0][2]+lv[1][2]+lv[2][2])**2)
    v_abs[10]= sqrt((lv[0][0]+lv[1][0]-lv[2][0])**2+(lv[0][1]+lv[1][1]-lv[2][1])**2+(lv[0][2]+lv[1][2]-lv[2][2])**2)
    v_abs[11]= sqrt((lv[0][0]-lv[1][0]+lv[2][0])**2+(lv[0][1]-lv[1][1]+lv[2][1])**2+(lv[0][2]-lv[1][2]+lv[2][2])**2)
    v_abs[12]= sqrt((-lv[0][0]+lv[1][0]+lv[2][0])**2+(-lv[0][1]+lv[1][1]+lv[2][1])**2+(-lv[0][2]+lv[1][2]+lv[2][2])**2)

    cdef f64_t max_abs = 0.0, min_abs = 1e100
    for vi in range(13):
        if v_abs[vi] > max_abs: max_abs = v_abs[vi]
        if v_abs[vi] < min_abs: min_abs = v_abs[vi]

    cdef i32_t L = <i32_t>ceil((Rmax + max_abs) / min_abs) + 1

    # signum 和 condition
    cdef i32_t signum[8][3]
    signum[0][0]=1;  signum[0][1]=1;  signum[0][2]=1
    signum[1][0]=-1; signum[1][1]=1;  signum[1][2]=1
    signum[2][0]=1;  signum[2][1]=-1; signum[2][2]=1
    signum[3][0]=1;  signum[3][1]=1;  signum[3][2]=-1
    signum[4][0]=-1; signum[4][1]=-1; signum[4][2]=1
    signum[5][0]=1;  signum[5][1]=-1; signum[5][2]=-1
    signum[6][0]=-1; signum[6][1]=1;  signum[6][2]=-1
    signum[7][0]=-1; signum[7][1]=-1; signum[7][2]=-1

    cdef i32_t cond[8][3]
    cond[0][0]=0;cond[0][1]=0;cond[0][2]=0
    cond[1][0]=1;cond[1][1]=0;cond[1][2]=0
    cond[2][0]=0;cond[2][1]=1;cond[2][2]=0
    cond[3][0]=0;cond[3][1]=0;cond[3][2]=1
    cond[4][0]=1;cond[4][1]=1;cond[4][2]=0
    cond[5][0]=0;cond[5][1]=1;cond[5][2]=1
    cond[6][0]=1;cond[6][1]=0;cond[6][2]=1
    cond[7][0]=1;cond[7][1]=1;cond[7][2]=1

    # 预分配缓冲区（保守估计）
    cdef i32_t max_pairs = N * N * 300
    cdef np.ndarray[f64_t, ndim=1] dist_arr = np.zeros(max_pairs, dtype=np.float64)
    cdef np.ndarray[i32_t, ndim=1] cc_arr = np.zeros(max_pairs, dtype=np.int32)
    cdef np.ndarray[i32_t, ndim=1] bc_arr = np.zeros(max_pairs, dtype=np.int32)
    cdef np.ndarray[i32_t, ndim=1] ti_arr = np.zeros(max_pairs, dtype=np.int32)
    cdef np.ndarray[i32_t, ndim=1] tj_arr = np.zeros(max_pairs, dtype=np.int32)
    cdef np.ndarray[np.uint8_t, ndim=1] shift_arr = np.zeros(max_pairs, dtype=np.uint8)  # 1=有周期位移, 0=零位移

    cdef bint is_zero_shift
    cdef f64_t Rmax2 = Rmax * Rmax
    cdef i32_t pair_idx = 0
    cdef i32_t qi, qj, qk, q, cc, bc
    cdef f64_t dx, dy, dz, d2
    cdef f64_t a11=lattice[0,0], a12=lattice[0,1], a13=lattice[0,2]
    cdef f64_t a21=lattice[1,0], a22=lattice[1,1], a23=lattice[1,2]
    cdef f64_t a31=lattice[2,0], a32=lattice[2,1], a33=lattice[2,2]

    cdef f64_t[:,::1] coords_v = coords
    cdef f64_t[::1] dist_v = dist_arr
    cdef i32_t[::1] cc_v = cc_arr
    cdef i32_t[::1] bc_v = bc_arr
    cdef i32_t[::1] ti_v = ti_arr
    cdef i32_t[::1] tj_v = tj_arr
    cdef np.uint8_t[::1] shift_v = shift_arr
    cdef i32_t[::1] tim = typ_i_map

    with nogil:
        for qi in range(L + 1):
            for qj in range(L + 1):
                for qk in range(L + 1):
                    for q in range(8):
                        if cond[q][0]*(qi==0) + cond[q][1]*(qj==0) + cond[q][2]*(qk==0) != 0:
                            continue
                        # 零位移条件: qi=0, qj=0, qk=0, q=0 (第一个象限, signum=[1,1,1], condition=[0,0,0])
                        is_zero_shift = (qi == 0 and qj == 0 and qk == 0 and q == 0)
                        for cc in range(N):
                            for bc in range(N):
                                dx = coords_v[cc,0] + signum[q][0]*qi - coords_v[bc,0]
                                dy = coords_v[cc,1] + signum[q][1]*qj - coords_v[bc,1]
                                dz = coords_v[cc,2] + signum[q][2]*qk - coords_v[bc,2]
                                d2 = _dist2(dx, dy, dz, a11,a12,a13, a21,a22,a23, a31,a32,a33)
                                if d2 < Rmax2:
                                    if pair_idx < max_pairs:
                                        dist_v[pair_idx] = sqrt(d2)
                                        cc_v[pair_idx] = cc
                                        bc_v[pair_idx] = bc
                                        ti_v[pair_idx] = tim[cc]
                                        tj_v[pair_idx] = tim[bc]
                                        shift_v[pair_idx] = 0 if is_zero_shift else 1
                                        pair_idx += 1

    # 体积
    cdef f64_t V = fabs(
        lattice[0,0]*(lattice[1,1]*lattice[2,2] - lattice[1,2]*lattice[2,1]) -
        lattice[0,1]*(lattice[1,0]*lattice[2,2] - lattice[1,2]*lattice[2,0]) +
        lattice[0,2]*(lattice[1,0]*lattice[2,1] - lattice[1,1]*lattice[2,0])
    )

    return (dist_arr[:pair_idx].copy(),
            cc_arr[:pair_idx].copy(),
            bc_arr[:pair_idx].copy(),
            ti_arr[:pair_idx].copy(),
            tj_arr[:pair_idx].copy(),
            shift_arr[:pair_idx].copy(),
            N_out, V, N)


# ═══════════════════════════════════════════════════════════════
# 2. 指纹计算 (fingerprint_calc) — 纯 Cython nogil
# ═══════════════════════════════════════════════════════════════

cdef inline f64_t _erf_approx(f64_t x) noexcept nogil:
    """erf 近似 (Abramowitz & Stegun 7.1.26, 精度 ~1e-7)"""
    cdef f64_t sign = 1.0 if x >= 0 else -1.0
    x = fabs(x)
    if x > 5.0:
        return sign
    cdef f64_t t = 1.0 / (1.0 + 0.3275911*x)
    cdef f64_t y = 1.0 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t
                          - 0.284496736)*t + 0.254829592)*t*exp(-x*x)
    return sign * y


def fingerprint_calc(
    np.ndarray[f64_t, ndim=1] dist_arr,    # (n_pairs,)
    np.ndarray[i32_t, ndim=1] cc_idx,      # (n_pairs,) current_cell 原子索引
    np.ndarray[i32_t, ndim=1] bc_idx,      # (n_pairs,) basic_cell 原子索引
    np.ndarray[i32_t, ndim=1] typ_i_arr,   # (n_pairs,) current_cell 类型
    np.ndarray[i32_t, ndim=1] typ_j_arr,   # (n_pairs,) basic_cell 类型
    np.ndarray[f64_t, ndim=1] N_out,       # (N,)
    f64_t V,
    np.ndarray[i32_t, ndim=1] numIons,     # (n_species,)
    f64_t Rmax=12.0,
    f64_t sigma=0.05,
    f64_t delta=0.08,
    i32_t dimension=3,
):
    """指纹计算 — 纯 Cython nogil 循环

    对应 Octave fingerprint_calc.m:
      for bins = 2:numBins
        for i = 1:Ncell       (basic_cell 原子)
          for j = 1:Nfull     (所有近邻)
            if dist(j,i) > 0 && |dist(j,i) - bin_center| < 4*sigma
              delt = 0.5*(erf(upper) - erf(lower))
              atom_fing(i, typ_j(j), bins) += delt / (N(j) * R0^2)
              fing(typ_i(j), typ_j(j), bins) += delt / R0^2

    关键：用 cc_idx/bc_idx 直接索引，无需 Python mask
    """
    cdef i32_t N = N_out.shape[0]
    cdef i32_t n_species = numIons.shape[0]
    cdef i32_t n_pairs = dist_arr.shape[0]

    if N == 0:
        return (np.array([]), np.array([]), np.array([]), V)

    cdef i32_t numBins = <i32_t>(Rmax / delta)
    # Octave: 2:numBins 包含 numBins, Python range(2, numBins+1)

    cdef f64_t sigma_eff = sigma / sqrt(2.0 * log(2.0))
    cdef f64_t normaliser = 1.0 if dimension != 0 else 0.0
    cdef f64_t total_ions = 0.0
    cdef i32_t s
    for s in range(n_species):
        total_ions += <f64_t>numIons[s]

    cdef f64_t sqrt2_sigma = sqrt(2.0) * sigma_eff
    cdef f64_t window = 4.0 * sigma_eff
    cdef f64_t vc_third = pow(V / <f64_t>N, 1.0/3.0)
    cdef f64_t inv_4pi_delta = 1.0 / (4.0 * M_PI * delta)

    # 输出数组 — numBins 列, 0-based 索引 bins-1 对应 Octave 1-based bins
    cdef np.ndarray[f64_t, ndim=2] fing = np.zeros((n_species * n_species, numBins), dtype=np.float64)
    cdef np.ndarray[f64_t, ndim=1] order = np.zeros(N, dtype=np.float64)
    cdef np.ndarray[f64_t, ndim=3] atom_fing = np.zeros((N, n_species, numBins), dtype=np.float64)

    if dimension != 0:
        fing[:, 0] = -1.0  # Octave fing(:,1) = -1, 0-based col 0

    # Cython memoryviews
    cdef f64_t[::1] dist_v = dist_arr
    cdef i32_t[::1] cc_v = cc_idx
    cdef i32_t[::1] bc_v = bc_idx
    cdef i32_t[::1] ti_v = typ_i_arr
    cdef i32_t[::1] tj_v = typ_j_arr
    cdef f64_t[::1] N_v = N_out
    cdef i32_t[::1] ni_v = numIons
    cdef f64_t[:,::1] fing_v = fing
    cdef f64_t[::1] order_v = order
    cdef f64_t[:,:,::1] af_v = atom_fing

    cdef i32_t bins, j, cc, bc, ti, tj, si, sj
    cdef f64_t bin_center, bin_lower, d, upper, lower, delt, weight

    with nogil:
        for bins in range(2, numBins + 1):  # Octave 2:numBins (包含 numBins)
            bin_center = delta * (bins - 0.5)
            bin_lower = delta * (bins - 1.0)

            for j in range(n_pairs):
                d = dist_v[j]
                if d <= 0.0:
                    continue
                if fabs(d - bin_center) >= window:
                    continue

                # erf 积分
                # Octave: interval(2) = (delta*bins - R0) / (sqrt(2)*sigm)  [bin 上边界]
                #         interval(1) = (delta*(bins-1) - R0) / (sqrt(2)*sigm) [bin 下边界]
                upper = (delta * bins - d) / sqrt2_sigma
                lower = (delta * (bins - 1) - d) / sqrt2_sigma
                if upper > 5.0: upper = 5.0
                elif upper < -5.0: upper = -5.0
                if lower > 5.0: lower = 5.0
                elif lower < -5.0: lower = -5.0

                delt = 0.5 * (_erf_approx(upper) - _erf_approx(lower))

                cc = cc_v[j]  # current_cell 原子 (近邻, Octave 行 j)
                bc = bc_v[j]  # basic_cell 原子 (晶胞内, Octave 列 i)
                ti = ti_v[j]  # 近邻类型 (对应 Octave typ_j)
                tj = tj_v[j]  # basic_cell 类型 (对应 Octave typ_i)

                # Octave: atom_fing(i, typ_j(j), bins) += delt / (Ni(j) * R0^2)
                #   i = basic_cell, typ_j(j) = 近邻类型 = ti
                af_v[bc, ti - 1, bins - 1] += delt / (N_v[cc] * d * d)

                # Octave: fing((typ_i(i)-1)*species + typ_j(j), bins) += delt / R0^2
                #   typ_i(i) = basic_cell 类型 = tj, typ_j(j) = 近邻类型 = ti
                fing_v[(tj - 1) * n_species + (ti - 1), bins - 1] += delt / (d * d)

            # 归一化 atom_fing
            for bc in range(N):
                for tj in range(n_species):
                    af_v[bc, tj, bins - 1] = V * af_v[bc, tj, bins - 1] * inv_4pi_delta - normaliser

            # order 贡献
            for tj in range(n_species):
                weight = <f64_t>ni_v[tj] / total_ions
                for bc in range(N):
                    order_v[bc] += weight * delta * af_v[bc, tj, bins - 1] * af_v[bc, tj, bins - 1] / vc_third

            # 归一化全局指纹
            for si in range(n_species):
                for sj in range(n_species):
                    s = si * n_species + sj
                    if ni_v[si] > 0 and ni_v[sj] > 0:
                        fing_v[s, bins - 1] = V * fing_v[s, bins - 1] / (4.0 * M_PI * <f64_t>ni_v[si] * <f64_t>ni_v[sj] * delta) - normaliser

    # sqrt(order)
    cdef i32_t i
    for i in range(N):
        if order_v[i] < 0.0:
            order_v[i] = 0.0
        order_v[i] = sqrt(order_v[i])

    return (order, fing, atom_fing, V)


# ═══════════════════════════════════════════════════════════════
# 3. 一站式计算接口
# ═══════════════════════════════════════════════════════════════

def compute_all(
    np.ndarray[f64_t, ndim=2] coords,
    np.ndarray[f64_t, ndim=2] lattice,
    np.ndarray[i32_t, ndim=1] numIons,
    np.ndarray[i32_t, ndim=1] atomType,
    f64_t Rmax=12.0,
    f64_t sigma=0.05,
    f64_t delta=0.08,
    i32_t dimension=3,
):
    """一站式计算：makeMatrices + fingerprint_calc"""
    import time as _time
    t0 = _time.time()

    dist_arr, cc_idx, bc_idx, ti_arr, tj_arr, shift_arr, N_out, V, N = build_distance_matrix(
        coords, lattice, Rmax, numIons, atomType
    )
    t1 = _time.time()

    order, fing, atom_fing, _ = fingerprint_calc(
        dist_arr, cc_idx, bc_idx, ti_arr, tj_arr, N_out, V, numIons,
        Rmax, sigma, delta, dimension
    )
    t2 = _time.time()

    return {
        'order': order,
        'fing': fing,
        'atom_fing': atom_fing,
        'V': V,
        'n_pairs': len(dist_arr),
        'time_matrix': t1 - t0,
        'time_fingerprint': t2 - t1,
        'time_total': t2 - t0,
    }