#!/usr/bin/env python
"""随机抽取 feature.csv / feature_mlp.csv / structures.traj 中相同比例的数据。

CSV 的第 i 行(去掉表头后)与 structures.traj 的第 i 帧一一对应。
随机抽出指定比例的行/帧，并将三者同步保存到新文件。

用法:
    python sample_data.py --ratio 0.3
    python sample_data.py --ratio 0.5 --seed 42 \
        --feature feature.csv --feature_mlp feature_mlp.csv --traj structures.traj
"""
import argparse
import csv
import random

from ase.io.trajectory import Trajectory, TrajectoryWriter


def read_csv(path):
    """返回 (header, rows)，rows 为 list[list[str]]，顺序即行号。"""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path} 为空")
    return rows[0], rows[1:]


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="按比例随机抽取 feature.csv / feature_mlp.csv / structures.traj 的对应数据")
    parser.add_argument("--ratio", type=float, default=0.5,
                        help="抽取比例 (0, 1]，如 0.3 表示抽 30%")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，保证可复现；不指定则每次不同")
    parser.add_argument("--feature", default="feature.csv",
                        help="feature.csv 路径 (默认 ./feature.csv)")
    parser.add_argument("--feature_mlp", default="feature_mlp.csv",
                        help="feature_mlp.csv 路径 (默认 ./feature_mlp.csv)")
    parser.add_argument("--traj", default="structures.traj",
                        help="structures.traj 路径 (默认 ./structures.traj)")
    parser.add_argument("--out_feature", default="feature_sample.csv",
                        help="输出 feature 文件名")
    parser.add_argument("--out_feature_mlp", default="feature_mlp_sample.csv",
                        help="输出 feature_mlp 文件名")
    parser.add_argument("--out_traj", default="structures_sample.traj",
                        help="输出 traj 文件名")
    args = parser.parse_args()

    if not (0.0 < args.ratio <= 1.0):
        raise SystemExit("ratio 必须在 (0, 1] 范围内")

    # 1. 读取 CSV
    feat_header, feat_rows = read_csv(args.feature)
    mlp_header, mlp_rows = read_csv(args.feature_mlp)
    n = len(feat_rows)
    if len(mlp_rows) != n:
        raise SystemExit(
            f"feature.csv ({n} 行) 与 feature_mlp.csv ({len(mlp_rows)} 行) 行数不一致")

    # 2. 读取轨迹，校验帧数
    images = Trajectory(args.traj)
    if len(images) != n:
        raise SystemExit(
            f"structures.traj ({len(images)} 帧) 与 feature.csv ({n} 行) 数量不一致")

    # 3. 随机抽取索引
    rng = random.Random(args.seed)
    k = int(round(n * args.ratio))
    if k < 1:
        k = 1
    if k > n:
        k = n
    selected = sorted(rng.sample(range(n), k))

    # 4. 写出 CSV（保持原表头，行按抽样顺序输出）
    write_csv(args.out_feature, feat_header, [feat_rows[i] for i in selected])
    write_csv(args.out_feature_mlp, mlp_header, [mlp_rows[i] for i in selected])

    # 5. 写出对应的轨迹帧
    writer = TrajectoryWriter(args.out_traj, mode="w")
    for idx in selected:
        writer.write(images[idx])
    writer.close()

    print(f"总共 {n} 条数据，抽取 {k} 条 (ratio={args.ratio})")
    print(f"  -> {args.out_feature}")
    print(f"  -> {args.out_feature_mlp}")
    print(f"  -> {args.out_traj}")


if __name__ == "__main__":
    main()
