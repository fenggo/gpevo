# GPEvo

**GPEvo** 是 USPEX-GP（基于 USPEX-9.4.4）分子晶体结构预测路径（`calculationType=310`）的纯 Python 实现，内置高斯过程（GP）引导的主动学习。不需要 MATLAB / OCTAVE，从 `INPUT.txt` 和分子模板直接运行。

| 项目 | 说明 |
|---|---|
| 上游项目 | [USPEX](https://uspex-team.org)（MATLAB 原版 `uspex-gp`） |
| 本仓库 | <https://gitee.com/fenggo/gpevo> |
| 许可 | MIT，见 [LICENSE](LICENSE) |

## 目录

- [功能概览](#功能概览)
- [安装](#安装)
- [快速开始](#快速开始)
- [运行与输出](#运行与输出)
  - [输出文件一览](#输出文件一览)
  - [`Individuals.traj`](#individualsla_traj)
  - [`origin`](#origin)
- [INPUT.txt 扩展选项](#inputtxt-扩展选项)
  - [GP 主动学习](#gp-主动学习)
  - [swarm 算子（可选）](#swarm-算子可选)
  - [强化学习算子/参数优化（可选）](#强化学习算子参数优化可选)
- [分析与绘图工具](#分析与绘图工具)
- [目录结构](#目录结构)
- [设计文档](#设计文档)
- [与 MATLAB 版的关系](#与-matlab-版的关系)

---

## 功能概览

- **分子晶体进化搜索（310）**：heredity / random / softmutation / permutation / latmutation / rotmutation 全部遗传算子 + 精英保留，与 MATLAB 版同一套流程；另含一个可选的 **swarm 算子**（分子感知刚体移动，见 [swarm 算子](#swarm-算子可选)）。
- **GP 主动学习**：每个结构经 `uspexkit.gp()` 预测密度/能量与不确定度；训练集**自动生长**——第一代结束时用参考方法给最优的若干个结构打标签建立数据库，之后每代按 EI（或不确定度/随机）选 top-K 打标签追加进训练集、失效模型缓存后重训、重预测，再回写适应度。
- **对称性与随机结构生成**：230 个空间群的 Wyckoff 展开，`random_cell_mol` 的 Cython 移植（`pip install .` 时自动编译）。
- **集群提交**：本地子进程与远程调度器两条路径，MPI 环境变量自动处理。
- **GP 训练集自举**：不需要预先准备 `feature_mlp.csv` / `feature.csv` / `structures.traj`，空目录（甚至不存在的目录）会在第一代结束时自动建起来。
- **自动绘图**：每代评估完成后，在 `results*/` 生成四面板 GP 状态图（`gp_gen*.png`）。
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

## 快速开始

`example/` 下有一份 310 路径的样例输入（`INPUT.txt`、`MOL_1`、`MOL_2`、`POSCAR` 等），可以直接复制出来当工作目录的模板：

```bash
mkdir -p run && cp example/INPUT.txt example/MOL_* run/
cd run
gpevo                       # 等价于 python -m gpevo
```

## 运行与输出

1. 准备工作目录，放入 `INPUT.txt` 和分子模板 `MOL_1`、`MOL_2`……（310 分子晶体路径的全部输入）。
2. 在该目录执行：

```bash
gpevo                # 或 python -m gpevo
```

### 输出文件一览

| 文件/目录 | 内容 |
|---|---|
| `OUTPUT.txt` | 运行日志，含逐代 `ID Origin Composition Enthalpy Volume KPOINTS SYMMETRY` 表；GP 开启时追加 `Enthalpy(GP) UC Density(GP) UC` 四列 |
| `results1/`（`results2/`…） | `gatheredPOSCARS`、`Individuals`、`Individuals.traj`、`origin`、`gp_gen*.png`、`density_predict.log` |
| `CalcFold1/`… | 每个结构的弛豫/GP 计算目录，`gp.csv` 记录每次 GP 预测 |
| `results*/<ID>/` | 某个结构走参考方法（SIESTA）打标签的工作目录；上一轮中断留下的空目录会在下次选点时自动清理 |
| `ei.log` | 每代 EI 选中的晶体 ID |

### `Individuals.traj`

`gatheredPOSCARS` 的 ASE 可读版本：每个结构算完即追加一帧，帧序与 `Individuals` 表一致（`numParallelCalcs=1` 时严格一致），第 *i* 帧就是 `Individuals` 中 ID 为 *i+1* 的那个结构。帧的 `atoms.info` 带有 `Number / generation / howCome / symg / enthalpy / density`（有 GP 预测时还有 `energy_pred / uncertainty_eng / density_gp / uncertainty`），焓同时挂为单点能量，可直接分析：

```python
from ase.io import read
frames = read("results1/Individuals.traj", index=":")
print(frames[0].info)                          # 与 Individuals 行对应的元数据
print(frames[0].get_potential_energy())        # 焓 (eV)
```

### `origin`

记录每个结构由哪个算子产生、父本是谁（`WriteOUTPUT.m` 的 `resFolder/origin`），是 `makeFigures.m` 给结构按算子着色的数据源：

```
 ID    Origin    Enthalpy   Parent-E   Parent-ID
   1   Random      -0.829    -0.829  [         0]
   2   Seeds       -1.956    [          ]
 223 keptBest      -1.988    -1.988  [       151]
```

`Enthalpy`/`Parent-E` 都是**每原子**焓（`Enthalpies(end)/sum(numIons)`）；无父本的结构（random）`Parent-ID` 为 `0` 并重复自身焓，种子结构父本信息为空，故 `Parent-E` 留空。格式与 MATLAB 版逐字节一致（含空父本时的空白列）。

## INPUT.txt 扩展选项

以下三组都是**可选扩展**，不写即以默认行为运行。推荐统一用 `value : keyword`（USPEX 的规范写法），`keyword : value` 也认；注释用 `%`、`!` 或 `#` 都可以。

### GP 主动学习

```
1         : gp            % 开关：1=启用，弛豫后追加一步 GP 评估
24        : ncpu          % GP 训练并行核数；不写则取 % numProcessors 最后一项，再缺省 8
data11_22 : data          % 训练数据目录（已有则直接用于 gp() 训练，没有则由本运行自己建）
1.8       : den           % 密度门：gp() 的 --den，同时是 CalcFitness EI 门的密度下限
0.04      : u             % 不确定度阈值（±1.96σ），超过才送 DFT
2         : k             % 每代 EI top-K 个数
ei        : mode          % ei | random | uncertainty
2         : nboot         % 训练集为空时首批打标签的结构数（默认 2，需 ≥2）
```

> 若 `gp` 一行因写法问题没被读入（例如关键字被注释污染），开关会**静默**保持 0——整条 GP 链都不发生。启动时会打印告警列出有问题的行。

流程：

1. **自举（第一代末）**：训练库不足 2 行时，按适应度取前 `nboot` 个结构，`uspexkit.calc()` 用参考方法（SIESTA）算它们的能量/密度，自动建起 `<homePath>/<data>/{feature_mlp.csv, feature.csv, structures.traj}`。这一代 GP 步整体跳过（无任何结构失败）。
2. **逐代生长（第二代起）**：`CalcFitness` 里按 `mode` 规则（EI / 不确定度 / 随机）取 top-`k`，再过 `u`（±1.96σ）与 `den` 两道门，选中的结构同样送参考方法打标签 → 追加进训练库。
3. **重训**：确实有新行入库才清理模型缓存——这些 pickle（`gpr_density.pkl` / `gpr_energy.pkl` / `rfr.pkl`，散落在 `CalcFold*`、`results*`、`data/` 三处）只按内核版本做戳，不清就不会因为数据变多而重训。随后 `uspexkit.pred()` 重预测并把新值回写适应度。

数据真的在长吗？看日志里这两行：

```
  GP training set has only 0 labelled row(s) in /home/feng/tnt/data_exp: seeding it with the best structures of generation 1
  GP training set: 0 -> 2 rows; dropped 3 cached model(s)
```

没长出来时也会说明卡在哪一道门（`No crystals above…` / `density threshold gate` / `calc failed` / `training set is still N row(s)`）。常见原因是参考方法跑不起来：标签靠 `siesta`（`mpirun`）+ `<homePath>/Specific/*.psf`，缺任一项启动时会提前告警。每代画图 `results*/gp_gen*.png`（种群分布 / 探索地图 / 预测 vs 采用值 / 外推检查四面板）。

> GP 内核来自 [uspexkit](https://gitee.com/fenggo/uspexkit)（源码在 `~/uspexkit`），以进程内 API 方式调用；改内核需重装 uspexkit。

### swarm 算子（可选）

分子感知的刚体移动，默认关闭（不写就完全不改变现有算子配比）：

```
0.1   : fracSwarm    % 每代额外产生的 swarm 结构数 = round(populationSize × 该值)
0.5   : swarmStep    % 分子向最优结构对应分子移动的位移比例上限（0~1）
30    : swarmAngle   % 移动后附加的刚体随机转动角度上限（度）
```

它把每个**分子整体**当作一个刚体，向当代最优结构中对应分子的位置移动一个随机比例的位移，再叠加一个随机刚体转动——`MOLCOORS` 只做刚体变换，分子内几何**逐位不变**，因此不像原子级软模位移那样会把分子撕开。位移会做最小镜像折叠，晶胞本身不变（交给后续弛豫）。

两点说明：

- 这是 PSO **思想**（向最优个体靠拢 + 随机步长）在分子晶体上的实现，**不是** uspex-gp 那个 PSO 模块的移植：后者是逐原子随机放置 + 原子软模位移，不支持分子。
- swarm 结构是**额外**加在 `populationSize` 之上的（与精英保留追加的结构同理），不会挤占现有算子的配额；未设 `fracSwarm` 时计数为 0，行为与之前完全一致。

### 强化学习算子/参数优化（可选）

`RL_strategy ≥ 1` 开启一个分层 contextual bandit：外层学 6 个算子的配额，内层对 Rotation / LatMutate / softmutate 三个算子的**连续变异幅度**做 Thompson 采样。状态存 `rl_agent_state.pkl`，可跨任务导出的先验放在 `RLPriors/`。

```
2      : RL_strategy    % 0=关；>=1 开启 bandit
0.9    : RL_forget      % 遗忘因子 λ（历史统计衰减）
1.0    : RL_w1          % 奖励权重：入选精英集
0.5    : RL_w2          % 奖励权重：焓相对增益
0.6    : RL_armCap      % 单个算子占比上限
1.0    : RL_kappa       % 内层探索系数，越大幅度探索越激进
1      : RL_autoExport  % 导出跨任务先验
```

内层学的四个连续参数，也都有 INPUT 入口（`RL_strategy=0` 时**直接固定为你设的值**；开启 RL 时作为**学习起点**）：

```
45     : rotationAngleMax   % 旋转变异角度上限，**单位度**（内部转弧度）
0.3    : translationMax     % 旋转后的刚体平移上限
0.15   : mutationRate       % LatMutation 的应变率
1.5    : softStepScale      % softmutate 步长缩放
```

RL 开启时的搜索范围固定为：角度 15°–90°、平移 0.1–0.8、应变 0.05–0.30、步长缩放 0.5–2.0（超出的设置值会被夹到区间端点）。

## 分析与绘图工具

```bash
# 多次运行的密度/能量分布对比（输出 compare_runs.png）
python compare_runs.py --dirs tnt/results2 uspex-gp/results1 --labels gpevo octave

# DFT 结果绘图 / 空间群覆盖度统计 / EI 日志 / 清理
python plot_dft.py
python sg_coverage.py
python ei_select.py
python clean.py
```

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
sg_coverage.py                # 空间群覆盖度统计
ei_select.py  clean.py        # EI 日志 / 清理脚本
example/  Specific/  train/  ebind/  docs/   # 示例、力场与文档
```

## 设计文档

`docs/` 下有若干设计与原理说明：

| 文件 | 内容 |
|---|---|
| `ea_flowchart.md` / `ea_mechanism.png` / `draw_ea.py` | 进化算法流程图与绘制脚本 |
| `uq_active_learning.md` / `uq_active_learning_flowchart.md` | GP 主动学习（UQ）流程 |
| `fingerprint_method.md` | 指纹方法说明 |
| `rl_bandit_design.md` | 强化学习 bandit 设计 |
| `uspexkit-gp内核C优化原理与运行机制.pptx` | uspexkit GP 内核的 C 优化原理 |

## 与 MATLAB 版的关系

MATLAB / OCTAVE 源码树不在本仓库内，Python 路径也不需要它们：不需要 `USPEXPATH` 环境变量，不需要配置 MATLAB path，入口只有 `gpevo` / `python -m gpevo` / `python gpevo.py` 三种。

## License

MIT，见 [LICENSE](LICENSE)。

## 链接

- 仓库：<https://gitee.com/fenggo/gpevo>
- 作者：FengGo
