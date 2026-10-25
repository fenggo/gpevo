#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总所有 resultsN 文件夹的 DFT 计算结果
优先读取 density.log; 若某文件夹没有 density.log, 则遍历其纯数字命名的子
文件夹中的 siesta.traj, 用 ase 获取能量 (atoms.get_potential_energy()) 并参考
density.py 计算密度, 按其它 density.log 的格式生成一份 density.log, 再读取。
密度与能量都接近 (重复结构) 的点会被忽略。
散点图: Density vs Energy, 按文件夹着色, 标注结构ID
"""
import os
import re
import glob
import argparse
import sys
import numpy as np
from ase.io import read, write
from ase.io.trajectory import TrajectoryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde


parser = argparse.ArgumentParser(description='plot the dft results: ./plot_dft.py --res=results')
parser.add_argument('--res', default='results', type=str, help='results folder prefix')
parser.add_argument('--a', default=0, type=int, help='auto (keep for back-compat)')
parser.add_argument('--dtol', default=1e-3, type=float, help='density tolerance for dedup (g/cm3)')
parser.add_argument('--etol', default=1e-3, type=float, help='energy tolerance for dedup (eV)')
args = parser.parse_args(sys.argv[1:])
res  = args.res


base = os.path.dirname(os.path.abspath(__file__))

# ── 密度 / 能量过滤阈值 (density.log 与 siesta.traj 两路共用) ──
MIN_DENSITY = 1.84
MAX_ENERGY = -14613.0

# ── 去重阈值: 密度与能量都落在此容差内则视为重复结构并忽略 ──
DENSITY_TOL = args.dtol
ENERGY_TOL = args.etol


def keep(density, energy):
    """过滤低密度或未收敛 (能量过正) 的结构"""
    return density > MIN_DENSITY and energy < MAX_ENERGY

def num_suffix(name):
    """从 'results12' / 'results-12' 中提取数字后缀，用于排序"""
    m = re.search(r'(\d+)$', name)
    return int(m.group(1)) if m else 0

def is_close(density, energy, seen):
    """判断 (density, energy) 是否与已保留点足够接近 (重复结构)"""
    for sd, se in seen:
        if abs(density - sd) <= DENSITY_TOL and abs(energy - se) <= ENERGY_TOL:
            return True
    return False

def generate_density_log(folder):
    """依据纯数字命名子文件夹中的 siesta.traj, 按其它 density.log 的格式生成 density.log.

    返回写入的条目数; 若没有找到任何 siesta.traj 则返回 0 且不创建文件.
    """
    rows = []
    for sub in sorted(glob.glob(os.path.join(folder, '[0-9]*'))):
        if not os.path.isdir(sub) or not os.path.basename(sub).isdigit():
            continue
        traj = os.path.join(sub, 'siesta.traj')
        if not os.path.isfile(traj):
            continue
        try:
            atoms = read(traj, index=-1)
            energy = atoms.get_potential_energy()
            volume = atoms.get_volume()
            masses = np.sum(atoms.get_masses())
            density = masses / volume / 0.602214129  # 参考 density.py
            rows.append((int(os.path.basename(sub)), density, energy))
        except Exception as ex:
            print(f'[skip] {traj}: {ex}', file=sys.stderr)

    if not rows:
        return 0

    fname = os.path.join(folder, 'density.log')
    with open(fname, 'w') as f:
        f.write('# Crystal_id Density Energy\n')
        for cid, density, energy in rows:
            f.write(f'{cid:5d}  {density:9.6f} {energy:15.8f}\n')
    return len(rows)


# ── 收集所有 resultsN 文件夹（按数字后缀排序）──
pat = re.compile(re.escape(res) + r'[-_]?(\d+)$')
folders = []
for d in glob.glob(os.path.join(base, res + '*')):
    m = pat.match(os.path.basename(d))
    if m:
        folders.append((int(m.group(1)), d))
folders = [d for _, d in sorted(folders)]
folder_paths = {os.path.basename(d): d for d in folders}

data = {}  # folder_name -> (ids, densities, energies)
all_d, all_e, all_label = [], [], []
seen = []   # 已保留的 (density, energy), 用于去重
n_dup = 0   # 被忽略的重复点数量

for folder in folders:
    fname = os.path.join(folder, 'density.log')
    name = os.path.basename(folder)
    if not os.path.isfile(fname):
        n = generate_density_log(folder)
        if n:
            print(f'[generate] {name}: wrote {n} entries to density.log', file=sys.stderr)
        else:
            print(f'[skip] {name}: no siesta.traj found', file=sys.stderr)
            continue
    ids, ds, es = [], [], []
    with open(fname) as f:
        next(f)  # skip header
        for line in f:
            p = line.split()
            if len(p) >= 3:
                density, energy = float(p[1]), float(p[2])
                if not keep(density, energy):
                    continue
                if is_close(density, energy, seen):
                    n_dup += 1
                    continue
                seen.append((density, energy))
                ids.append(int(p[0]))
                ds.append(density)
                es.append(energy)
    if ids:
        data[name] = (ids, ds, es)
        all_d.extend(ds)
        all_e.extend(es)
        all_label.extend([name]*len(ds))

all_d = np.array(all_d)
all_e = np.array(all_e)
all_label = np.array(all_label)

print(f'\nTotal DFT points: {len(all_d)} from {len(data)} folders ({n_dup} duplicates ignored)')
print(f'{"Folder":<16} {"Crystal_ID":>10} {"Density":>10} {"Energy":>16}')
print('-' * 56)
for name in sorted(data, key=num_suffix):
    ids, ds, es = data[name]
    for cid, d, e in zip(ids, ds, es):
        print(f'{name:<16} {cid:>10} {d:>10.4f} {e:>16.5f}')
print('-' * 56)
print(f'{"Total":<16} {len(all_d):>10}\n')

folder_names = sorted(data.keys(), key=num_suffix)

# ── 保存每个绘图点对应的结构到 structures.traj ──
def read_last_or_none(path):
    """读取轨迹的最后一帧; 读取失败时返回 None 而不是终止整个脚本."""
    try:
        return read(path, index=-1)
    except Exception as ex:
        print(f'[warn] cannot read {path}: {ex}', file=sys.stderr)
        return None


# if os.path.isfile('structures.traj'):
#     os.remove('structures.traj')

n_saved = 0
n_missing = 0
traj    = TrajectoryWriter('structures.traj',mode='w')
for name in folder_names:
    ids = data[name][0]
    folder = folder_paths[name]
    for cid in ids:
        sub = os.path.join(folder, str(cid))
        atoms = None

        id_traj = os.path.join(sub, f'id_{cid}.traj')
        if os.path.isfile(id_traj):
            atoms = read_last_or_none(id_traj)

        if atoms is None:
            siesta_traj = os.path.join(sub, 'siesta.traj')
            if os.path.isfile(siesta_traj):
                atoms = read_last_or_none(siesta_traj)

        if atoms is None:
            print(f'[skip structure] {name}/{cid}: no readable id_*.traj or siesta.traj',
                  file=sys.stderr)
            n_missing += 1
            continue

        # write('structures.traj', atoms, append=True)
        traj.write(atoms=atoms)
        
        n_saved += 1
traj.close()
print(f'\nSaved {n_saved} structures to structures.traj ({n_missing} missing)')

# ── 绘图 ──
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel(r'$Density$ ($g/cm^3$)', fontsize=13)
ax.set_ylabel(r'$Relative\ Energy$ ($eV$)', fontsize=13)
# ax.set_title(r'TNT$_4$·CL-20$_4$ DFT Results (4:4 molar ratio)', fontsize=13)

# 每个文件夹用不同颜色
cmap = plt.cm.tab10
n_folders = len(data)

for i, name in enumerate(folder_names):
    ids, ds, es = data[name]
    ax.scatter(ds, es, s=80, alpha=0.85, marker='o',
               color=cmap(i % 10), edgecolors='k', linewidths=0.5,
               label=name, zorder=3)
    # 标注结构ID
    for j in range(len(ids)):
        ax.annotate(str(ids[j]), (ds[j], es[j]), fontsize=6,
                    ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points', color=cmap(i % 10))

# 标注最优结构
best_idx = np.argmin(all_e)
ax.scatter(all_d[best_idx], all_e[best_idx], marker='*', s=300,
           facecolors='none', edgecolors='red', linewidths=2, 
           zorder=5, label=f'Best (D={all_d[best_idx]:.4f})')

# ── 边际 KDE ──
divider = make_axes_locatable(ax)
ax_top   = divider.append_axes("top",   size="15%", pad=0.18)
ax_right = divider.append_axes("right", size="15%", pad=0.08)

xl, yl = ax.get_xlim(), ax.get_ylim()

kde_d = gaussian_kde(all_d)
xd = np.linspace(xl[0], xl[1], 300)
ax_top.plot(xd, kde_d(xd), color='steelblue', lw=1.5)
ax_top.fill_between(xd, kde_d(xd), alpha=0.25, color='steelblue')
ax_top.set_xlim(xl); ax_top.set_xticks([]); ax_top.set_yticks([])
for sp in ['top','right','left']: ax_top.spines[sp].set_visible(False)

kde_e = gaussian_kde(all_e)
xe = np.linspace(yl[0], yl[1], 300)
ax_right.plot(kde_e(xe), xe, color='coral', lw=1.5)
ax_right.fill_between(kde_e(xe), xe, alpha=0.25, color='coral')
ax_right.set_ylim(yl); ax_right.set_xticks([]); ax_right.set_yticks([])
for sp in ['top','right','bottom']: ax_right.spines[sp].set_visible(False)

# ax.legend(loc='lower left', fontsize=8, framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.15)

# 信息框
# info = (f'Total DFT structures: {len(all_d)}\n'
#         f'Density: {all_d.min():.4f} – {all_d.max():.4f} g/cm³\n'
#         f'Energy: {all_e.min():.2f} – {all_e.max():.2f} eV\n'
#         f'Best: D={all_d[best_idx]:.4f}, E={all_e[best_idx]:.2f} eV')
# ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=8,
#         verticalalignment='top', horizontalalignment='right',
#         bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

plt.savefig('dft_summary.png', dpi=200, bbox_inches='tight')
plt.savefig('dft_summary.svg', transparent=True, bbox_inches='tight')
print('\nSaved: dft_summary.png, dft_summary.svg')
