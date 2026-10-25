#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USPEX 310 共晶 (TNT₄·CL20₄) 全代密度分析 —— 纯绘图 + 报告.

输入: results1/density_predict.log  (由 compute_gp_parallel.py 全量重算生成)
      results1/Individuals          (晶体 → 代 映射 + ML 密度)
      results1/density.log          (DFT 验证, 可选)

产出:
  1. {prefix}_evolution.png  —— 所有代密度进化 (ML 散点 + 每代 max 曲线 + DFT 点)
  2. {prefix}_pergen.png     —— 每代 GP top-5 高密度 + EI 选中的 2 个 + DFT 散点
  3. 终端报告 —— 每代 top-5 密度 / EI-2 / DFT 精确数值

GP / EI 语义 (与 CalcFitness_310.m 一致):
  density_gp  = density_predict.log 第5列
  σ           = 1.96 × std_den  (gp.csv col7 = 1.96 * std_den)
  f_best      = max(density_gp) 于 residual≤10 的本代有效晶体
  EI          = (μ-f_best)·Φ(Z) + σ·φ(Z),  Z=(μ-f_best)/σ
  每代取 EI 前 k 个 (INPUT.txt --k=2)

用法:
  python3 gp_density_all_gens.py                    # 解析现有 log 画图
  python3 gp_density_all_gens.py --k 2 --top 5
"""

import os
import re
import sys
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNCERT_SCALE = 1.96          # gp.csv uncertainty = 1.96 * pred std_den
RESIDUAL_CUT = 10.0          # CalcFitness_310.m 的 residual>10 过滤


# -----------------------------------------------------------------------------
#  解析
# -----------------------------------------------------------------------------

def strip_brackets(line):
    return re.sub(r"\[[^\]]*\]", "", line)


def parse_individuals(path):
    """→ (all_rows, gen_ids) ; 只保留有效晶体 (fitness < 0, 与 pred 的 f<0 过滤一致).

    all_rows: list[(gen, id, enthalpy, volume, density, fitness)]
    """
    rows = []
    with open(path) as f:
        for line in f:
            l = strip_brackets(line).split()
            if len(l) < 7 or l[0] in ("Gen", "ID"):
                continue
            try:
                gen = int(l[0]); cid = int(l[1])
                enthalpy = float(l[3]); volume = float(l[4])
                density = float(l[5]); fitness = float(l[6])
            except (ValueError, IndexError):
                continue
            if fitness >= 0.0:
                continue
            rows.append((gen, cid, enthalpy, volume, density, fitness))
    gen_ids = {}
    for g, cid, *_ in rows:
        gen_ids.setdefault(g, []).append(cid)
    return rows, gen_ids


def parse_density_predict(path):
    """→ {id: (residual, density_gp, std_den, density_mlp, density_rf, energy)}  (同 id 取最后)"""
    data = {}
    if not os.path.exists(path):
        return data
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 6:
                continue
            try:
                cid = int(p[0])
                residual = float(p[1]); dmlp = float(p[2]); drf = float(p[3])
                dgp = float(p[4]); std = float(p[5])
                eng = float(p[6]) if len(p) >= 7 else float("nan")
            except ValueError:
                continue
            data[cid] = (residual, dgp, std, dmlp, drf, eng)
    return data


def parse_density_log(path):
    """→ {id: (density, energy)}  (DFT)"""
    dft = {}
    if not os.path.exists(path):
        return dft
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 3:
                try:
                    dft[int(p[0])] = (float(p[1]), float(p[2]))
                except ValueError:
                    continue
    return dft


# -----------------------------------------------------------------------------
#  EI
# -----------------------------------------------------------------------------

def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _Phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def expected_improvement(mu, sigma, f_best):
    diff = mu - f_best
    if sigma < 1e-10:
        return max(0.0, diff)
    z = diff / sigma
    return diff * _Phi(z) + sigma * _phi(z)


# -----------------------------------------------------------------------------
#  逐代分析
# -----------------------------------------------------------------------------

def build_per_gen(gen_ids, pred, dft, k=2, top=5):
    out = {}
    for gen, ids in gen_ids.items():
        rows = [(cid,) + pred[cid] for cid in ids if cid in pred]
        valid = [r for r in rows if r[1] <= RESIDUAL_CUT]          # r[1]=residual
        f_best = max((r[2] for r in valid), default=float("-inf"))  # r[2]=density_gp

        ei_list = []
        for cid, residual, dgp, std, dmlp, drf, eng in valid:
            sigma = UNCERT_SCALE * std
            ei = expected_improvement(dgp, sigma, f_best)
            ei_list.append((ei, cid, dgp, sigma, residual))
        ei_list.sort(key=lambda r: -r[0])
        ei_topk = ei_list[:k]

        top_rows = sorted(valid, key=lambda r: -r[2])[:top]
        top_n = [(r[0], r[2], r[3]) for r in top_rows]              # (id, dgp, std)

        dft_in_gen = {cid: dft[cid] for cid in ids if cid in dft}

        out[gen] = {
            "all_rows": rows,
            "valid": valid,
            "f_best": f_best,
            "ei_topk": ei_topk,
            "top_n": top_n,
            "dft": dft_in_gen,
            "broken": [r for r in rows if r[1] > RESIDUAL_CUT],
        }
    return out


# -----------------------------------------------------------------------------
#  绘图
# -----------------------------------------------------------------------------

def jitter(n, span=0.30, seed=0):
    """确定性 jitter, 把同代多个点横向铺开."""
    if n <= 1:
        return [0.0]
    return list(np.linspace(-span, span, n))


def plot_evolution(all_rows, dft_by_gen, out_png):
    gens = sorted({r[0] for r in all_rows})
    fig, ax = plt.subplots(figsize=(11, 6))

    # 每代 max 密度曲线 (不画所有晶体散点)
    best_by_gen = {}
    for r in all_rows:
        g, d = r[0], r[4]
        if g not in best_by_gen or d > best_by_gen[g]:
            best_by_gen[g] = d
    ax.plot(gens, [best_by_gen[g] for g in gens], "o-", color="#2980b9",
            lw=2.0, ms=6, zorder=3, label="best (max density) per generation")

    # DFT 验证点叠加
    if dft_by_gen:
        dft_g = sorted(dft_by_gen)
        dft_x = [g for g in dft_g for _ in dft_by_gen[g]]
        dft_y = [dft_by_gen[g][cid][0] for g in dft_g for cid in dft_by_gen[g]]
        ax.scatter(dft_x, dft_y, s=120, marker="D", color="#c0392b",
                   edgecolors="#641e16", linewidths=1.0, zorder=6, label="DFT density")

    ax.set_xlabel("Generation")
    ax.set_ylabel(r"Density (g/cm$^3$)")
    ax.set_title(r"USPEX density evolution  |  TNT$_4$·CL20$_4$ (228 atoms)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(gens)
    ax.set_xlim(min(gens) - 0.5, max(gens) + 0.5)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_pergen(per_gen, gen_ids, pred, dft, id2ml, out_png, k=2, top=5):
    gens = sorted(gen_ids.keys())
    fig, ax = plt.subplots(figsize=(12, 6.8))

    # ============ GP 预测 (top-5 + EI 更新) ============
    x_by_id = {}        # (gen, cid) -> x 位置, DFT 点对齐用
    for gi, gen in enumerate(gens):
        top_n = per_gen[gen]["top_n"]
        ei_topk = per_gen[gen]["ei_topk"]
        offs = jitter(len(top_n), span=0.30)
        eoffs = jitter(len(ei_topk), span=0.16)

        # top-5 高密度: 橙色星 + ±1.96σ 误差棒
        for (cid, dgp, std), j in zip(top_n, offs):
            x = gen + j
            x_by_id[(gen, cid)] = x
            sigma = UNCERT_SCALE * std
            ax.errorbar(x, dgp, yerr=sigma, fmt="none", ecolor="#e67e22",
                        elinewidth=1.0, alpha=0.55, capsize=0, zorder=3)
            ax.scatter(x, dgp, s=72, marker="*", color="#e67e22",
                       edgecolors="#7e5109", linewidths=0.6, zorder=4,
                       label="GP top-5 (pred ±1.96σ)" if gen == gens[0] else "")
            dy = 11 if gi % 2 == 0 else -15
            ax.annotate(f"#{cid}", (x, dgp), textcoords="offset points",
                        xytext=(0, dy), fontsize=6.5, color="#8a6d1a",
                        ha="center", fontweight="bold", zorder=8)

        # EI 选中: 绿色圆圈圈住原 GP 预测星
        for (ei, cid, dgp, sigma, residual), j in zip(ei_topk, eoffs):
            x = gen + j
            x_by_id.setdefault((gen, cid), x)
            ax.scatter(x, dgp, s=360, facecolors="none", edgecolors="#27ae60",
                       linewidths=1.8, zorder=5,
                       label="EI-selected" if gen == gens[0] else "")
            # 更新高斯值: DFT 完成后该点纳入训练集, 无观测噪声 GP
            # 在训练点处后验均值 = 观测值 (DFT 密度)
            if cid in dft:
                d_upd = dft[cid][0]
                ax.scatter(x, d_upd, s=95, marker="^", color="#16a085",
                           edgecolors="#0e6655", linewidths=0.7, zorder=7,
                           label="updated GP (after DFT included)"
                           if gen == gens[0] else "")
                ax.plot([x, x], [dgp, d_upd], color="#aab7b8", lw=0.8,
                        alpha=0.6, zorder=4)

    # ============ DFT: 与同一晶体的 GP 预测值纵向对齐 + 连线 ============
    for gen in gens:
        dft_g = {cid: v for cid, v in per_gen[gen]["dft"].items() if cid in pred}
        top_ei_ids = {t[0] for t in per_gen[gen]["top_n"]} | \
                     {t[1] for t in per_gen[gen]["ei_topk"]}
        # EI 且已 DFT 的晶体: 更新值已用三角画过, 这里不再画红菱形
        ei_dft_ids = {t[1] for t in per_gen[gen]["ei_topk"] if t[1] in dft_g}
        # 不在 top-5/EI 中的 DFT 晶体: 在代右侧等距排开, 补画其 GP 小星
        extra_ids = [cid for cid in dft_g if (gen, cid) not in x_by_id]
        for i, cid in enumerate(extra_ids):
            x_by_id[(gen, cid)] = gen + 0.46 + i * 0.14

        for cid, (dd, de) in dft_g.items():
            if cid not in pred:
                continue    # 无对应 GP 预测值, 不画 DFT 点
            x = x_by_id[(gen, cid)]
            dgp = pred[cid][1]
            if cid not in top_ei_ids:
                ax.scatter(x, dgp, s=40, marker="*", color="#e67e22",
                           edgecolors="#7e5109", linewidths=0.5,
                           zorder=4, alpha=0.85)
            if cid in ei_dft_ids:
                # 更新值三角 + 圆圈已画, 跳过红菱形 (对比线也由 EI 段画过)
                continue
            # GP 预测 → DFT 实际 对比竖线
            ax.plot([x, x], [dgp, dd], color="#7f8c8d", lw=0.9,
                    linestyle="--", alpha=0.7, zorder=4)
            ax.scatter(x, dd, s=130, marker="D", color="#c0392b",
                       edgecolors="#641e16", linewidths=1.0, zorder=6,
                       label="DFT density" if gen == gens[0] else "")
            ax.annotate(f"#{cid}", (x, dd), textcoords="offset points",
                        xytext=(0, 9), fontsize=7, color="#922b21",
                        ha="center", fontweight="bold", zorder=8)

    ax.set_xlabel("Generation")
    ax.set_ylabel(r"Density (g/cm$^3$)")
    ax.set_title("GP top-5 high-density + EI-selected (updated) + DFT",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    by_lab = {}
    for h, lb in zip(handles, labels):
        if lb and lb not in by_lab:
            by_lab[lb] = h
    ax.legend(by_lab.values(), by_lab.keys(), loc="lower right",
              fontsize=9, framealpha=0.95, ncol=1)
    ax.set_xticks(gens)
    # 右侧可能排有非 top-5 的 DFT 点 (gen+0.46 起, 每个间距 0.14)
    ax.set_xlim(min(gens) - 0.5, max(gens) + 1.1)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


# -----------------------------------------------------------------------------
#  报告
# -----------------------------------------------------------------------------

def report(per_gen, gen_ids, k=2, top=5):
    print("\n======== 逐代报告 ========")
    for gen in sorted(gen_ids.keys()):
        pg = per_gen[gen]
        top_n = pg["top_n"]
        ei_topk = pg["ei_topk"]
        dft_g = pg["dft"]
        broke = len(pg["broken"])
        print(f"\ngen {gen:2d}  预测={len(pg['all_rows'])}/{len(gen_ids[gen])} "
              f"(broken={broke})  f_best={pg['f_best']:.4f}")
        print(f"    top-{top} 密度: " + ", ".join(
            f"#{cid}(gp={dgp:.4f},σ={1.96*std:.4f})" for cid, dgp, std in top_n))
        print(f"    EI-{k} 选中:  " + ", ".join(
            f"#{cid}(EI={ei:.5f},gp={dgp:.4f},σ={sigma:.4f})"
            for ei, cid, dgp, sigma, residual in ei_topk))
        if dft_g:
            print(f"    DFT:      " + ", ".join(
                f"#{cid}(ρ={dd:.4f},E={de:.4f})" for cid, (dd, de) in sorted(dft_g.items())))
        else:
            print(f"    DFT:      (无)")


# -----------------------------------------------------------------------------
#  main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="USPEX 全代 GP 密度 + EI + DFT 绘图报告")
    ap.add_argument("--results1", default="/home/feng/uspex_tnt/uspex_gp/results1")
    ap.add_argument("--k", type=int, default=2, help="EI 选择个数 (INPUT --k)")
    ap.add_argument("--top", type=int, default=5, help="每代 GP 高密度 top-N")
    ap.add_argument("--last", type=int, default=None,
                    help="只绘制最后 N 代 (与 --gen-range 互斥; 两者都不给=全部)")
    ap.add_argument("--gen-range", default=None, metavar="LO-HI",
                    help="只绘制 LO~HI 代 (含端点), 例如 20-25")
    ap.add_argument("--out", default=None, help="输出前缀 (默认 results1/gp_all_gens)")
    args = ap.parse_args()

    res_dir = args.results1
    ind_path = os.path.join(res_dir, "Individuals")
    pred_path = os.path.join(res_dir, "density_predict.log")
    dft_path = os.path.join(res_dir, "density.log")

    all_rows, gen_ids = parse_individuals(ind_path)
    pred = parse_density_predict(pred_path)
    dft = parse_density_log(dft_path)

    # 代范围筛选: --gen-range LO-HI 或 --last N
    if args.gen_range:
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", args.gen_range)
        if not m:
            ap.error(f"--gen-range 格式应为 LO-HI, 收到 {args.gen_range!r}")
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            ap.error(f"--gen-range 起点 > 终点: {lo}-{hi}")
        gen_list = [g for g in sorted(gen_ids) if lo <= g <= hi]
        missing = [g for g in range(lo, hi + 1) if g not in gen_ids]
        if missing:
            print(f"警告: 代 {missing} 不存在, 已跳过")
    elif args.last:
        gen_list = sorted(gen_ids.keys())[-args.last:]
    else:
        gen_list = sorted(gen_ids.keys())
    gen_ids = {g: gen_ids[g] for g in gen_list}
    all_rows = [r for r in all_rows if r[0] in gen_list]

    if not gen_ids:
        sys.exit("错误: 指定范围内没有任何代")
    total = sum(len(v) for v in gen_ids.values())
    have = sum(1 for cid in pred if any(cid in ids for ids in gen_ids.values()))
    print(f"代数 {len(gen_ids)} (gen {min(gen_ids)}–{max(gen_ids)}), "
          f"有效晶体 {total}, 已有 GP 预测 {have} ({have/total*100:.1f}%)")

    id2ml = {r[1]: (r[0], r[4]) for r in all_rows}      # id -> (gen, ml_density)

    per_gen = build_per_gen(gen_ids, pred, dft, k=args.k, top=args.top)

    # DFT 按代归组 (供进化图叠加)
    dft_by_gen = {}
    for cid, (dd, de) in dft.items():
        if cid in id2ml:
            dft_by_gen.setdefault(id2ml[cid][0], {})[cid] = (dd, de)

    prefix = args.out if args.out else os.path.join(res_dir, "gp_all_gens")
    plot_evolution(all_rows, dft_by_gen, prefix + "_evolution.png")
    plot_pergen(per_gen, gen_ids, pred, dft, id2ml, prefix + "_pergen.png",
                k=args.k, top=args.top)
    report(per_gen, gen_ids, k=args.k, top=args.top)
    print("\nDone.")


if __name__ == "__main__":
    main()