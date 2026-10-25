# GPEvo

**GPEvo** 是 USPEX-GP（基于 USPEX-9.4.4）分子晶体结构预测路径（`calculationType=310`）的纯 Python 实现，内置高斯过程（GP）引导的主动学习。不需要 MATLAB / OCTAVE，从 `INPUT.txt` 和分子模板直接运行。

- 上游项目：[USPEX](https://uspex-team.org)（MATLAB 原版 `uspex-gp`）
- 本仓库：<https://gitee.com/fenggo/gpevo>

---

## 功能

- **分子晶体进化搜索（310）**：heredity / random / softmutation / permutation / latmutation / rotmutation 全部遗传算子 + 精英保留，与 MATLAB 版同一套流程。
- **GP 主动学习**：每个结构经 `uspexkit.gp()` 预测密度/能量与不确定度；`CalcFitness` 阶段按 EI（或不确定度/随机）选 top-K 送 DFT，结果追加进训练集后重训、重预测，再回写适应度。
- **对称性与随机结构生成**：230 个空间群的 Wyckoff 展开，`random_cell_mol` 的 Cython 移植（`pip install .` 时自动编译）。
- **集群提交**：本地子进程与远程调度器两条路径，MPI 环境变量自动处理。
- **自动绘图**：第一代评估完成、GP 训练结束后，在 `results*/` 生成四面板 GP 状态图（`gp_gen1.png`）。
- **结果对比工具**：`compare_runs.py` 多运行目录的密度/能量分布对比，`plot_dft.py` DFT 结果绘图。

## 安装

依赖：Python ≥ 3.9，以及 numpy / scipy / networkx / spglib；编译 Cython 扩展需要一个 C 编译器（GCC 即可）。

```bash
git clone https://gitee.com/fenggo/gpevo.git
cd gpevo
pip install .                          # 有网络（构建隔离自动拉取 setuptools/cython）
pip install . --no-build-isolation     # 离线，用本机已有的 setuptools + Cython
```

安装后得到：

| 项目 | 名称 |
|---|---|
| 发行包（distribution） | `GPEvo` |
| 导入包 | `gpevo` |
| 命令 | `gpevo` |

```bash
python -c "import gpevo; print(gpevo.__file__)"   # 验证安装位置
```

**免安装运行**：在仓库根目录放好 `INPUT.txt` 后直接 `python gpevo.py`（会把仓库根加入 `sys.path`）。

## 运行

1. 准备工作目录，放入 `INPUT.txt` 和分子模板 `MOL_1`、`MOL_2`……（310 分子晶体路径的全部输入）。
2. 在该目录执行：

```bash
gpevo                # 或 python -m gpevo
```

3. 产出（均在工作目录下）：

| 文件/目录 | 内容 |
|---|---|
| `OUTPUT.txt` | 运行日志，含逐代 `ID Origin Composition Enthalpy Volume KPOINTS SYMMETRY` 表；GP 开启时追加 `Enthalpy(GP) UC Density(GP) UC` 四列 |
| `results1/`（`results2/`…） | `gatheredPOSCARS`、`Individuals`、`gp_gen1.png`、`density_predict.log` |
| `CalcFold1/`… | 每个结构的弛豫/GP 计算目录，`gp.csv` 记录每次 GP 预测 |
| `ei.log` | 每代 EI 选中的晶体 ID |

4. 运行结束后用 `compare_runs.py` 对比多次运行：

```bash
python compare_runs.py --dirs tnt/results2 uspex-gp/results1 --labels gpevo octave
```

## GP 主动学习（INPUT.txt 选项）

在 `INPUT.txt` 中加下面几行即可开启（`value : keyword` 与 `keyword : value` 两种顺序都认）：

```
1      : gp          # 开关：1=启用，弛豫后追加一步 GP 评估
24     : ncpu        # GP 训练并行核数；不写则取 % numProcessors 最后一项，再缺省 8
data   : data11_22   # 训练数据目录（需含 feature_mlp.csv / feature.csv / structures.traj）
den    : 1.8         # 密度门：gp() 的 --den，同时是 CalcFitness EI 门的密度下限
0.04   : u           # 不确定度阈值（±1.96σ），超过才送 DFT
2      : k           # 每代 EI top-K 个数
ei     : mode        # ei | random | uncertainty
```

流程：GULP 弛豫 → `uspexkit.gp()` 预测（进程内调用其 Python API，不走命令行）→ `CalcFitness` 里 EI top-K 筛选 → DFT → 追加训练集 → 重训 → 重预测 → 回写适应度。第一代结束后自动画 `results*/gp_gen1.png`（种群分布 / 探索地图 / 预测 vs 采用值 / 外推检查四面板）。

> GP 内核来自 [uspexkit](https://gitee.com/fenggo/uspexkit)（源码在 `~/uspexkit`），以进程内 API 方式调用；改内核需重装 uspexkit。

## 目录结构

```
gpevo/                        # Python 包
  FunctionFolder/
    USPEX/                    # 核心进化算法（310 路径）
    AbinitCode/               # GULP / SIESTA 接口与 I/O
    Submission/               # 作业提交（本地子进程 / 集群调度器）
    spacegroup/               # 对称性分析 + random_cell_mol（Cython 移植）
    sys/  Tool/               # 系统工具函数
  cli.py                      # 控制台入口
  runtime.py  inputparser.py  log_utils.py  matrix.py  importers.py
gpevo.py                      # 免安装入口（仓库根目录运行）
compare_runs.py  plot_dft.py  # 结果绘图工具
ei_select.py  clean.py        # EI 日志 / 清理脚本
example/  Specific/  train/  ebind/  docs/   # 示例、力场与文档
```

## 遗留说明

MATLAB / OCTAVE 源码树不在本仓库内，Python 路径也不需要它们：不需要 `USPEXPATH` 环境变量，不需要配置 MATLAB path，入口只有 `gpevo` / `python -m gpevo` / `python gpevo.py` 三种。

## License

MIT，见 [LICENSE](LICENSE)。

## 链接

- 仓库：<https://gitee.com/fenggo/gpevo>
- 作者：FengGo
