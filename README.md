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
1         : cache         % 1（默认）= pred 复用 GP 步已算出的 10 维特征；0 = 每个晶体重算
1         : explore       % 1（默认）= 完整 EI；0 = 去掉 σ·φ(Z) 项，只按提升量排序
```

#### `explore : 0` —— 关掉 EI 的探索项

EI = (μ − f_best)·Φ(Z) + **σ·φ(Z)**。后一项是唯一"奖励未知"的项：一个 GP 从没见过的晶体自带很宽的后验，σ·φ(Z) 可以盖过提升项，于是**预测值明明比 incumbent 还差的低质量晶体**反而排到前面，白白花掉一次昂贵的参考计算。关掉之后只剩纯粹的利用项——提升幅度 × 提升概率：

```
1.9 是观测到的 incumbent（feature.csv 里 DFT 密度的最大值）
  晶体 A: μ=1.82, σ=0.50  →  EI = −0.035 + 0.197 = 0.162   ← 竟然赢了
  晶体 B: μ=1.95, σ=0.05  →  EI =  0.042 + 0.012 = 0.054
explore : 0 后
  晶体 A: −0.035   ← 归位
  晶体 B:  0.042   ← 赢
```

代价是失去探索：训练库很久没有新区域时，σ 大的晶体不再被优先采样，模型可能收敛到一个局部区域。想保留探索就保持 `1`，或改用 `mode : uncertainty`（那本来就是纯 σ 排序，是另一回事）。`mode` 标签会打印成 `EI (no exploration)` 以示区别。

`cache : 1` 省掉的是每代每个晶体 1 次 ReaxFF 重优化 + 4 次 GULP（特征计算）。前提是 GP 步读的 `gulp.cif` 与 `Individuals.traj` 帧是**同一个弛豫后的结构**（`WriteIndividualsTraj` 写的就是 `s.COORDINATES`），所以复用不会换结构，只是不再重新走一遍极小化。日志会打出实际复用个数：

```
  pred: reusing the GP-step feature for 132/134 crystal(s), skipping their ReaxFF re-optimisation
```

> 若 `gp` 一行因写法问题没被读入（例如关键字被注释污染），开关会**静默**保持 0——整条 GP 链都不发生。启动时会打印告警列出有问题的行。

EI 用哪个量做采集函数，由 `optType` 决定：

| optType | 采集量 μ | σ |
|---|---|---|
| `1`（焓极小） | GP 预测能量 `energy_gp` 取负（极小化） | `uncertainty_eng` |
| `-1`（焓极大） | 同上取正（极大化） | `uncertainty_eng` |
| 其它（默认 11 等） | GP 预测密度 `density_gp`（极大化） | `uncertainty` |

`den`（密度门）不受此影响，始终作用在预测密度上；`u`（±1.96σ）则作用在对应采集量的 σ 上，因此 optType 1 时它的单位是能量（eV），按能量不确定度的量级调即可。整代没有一个结构带可用的能量预测时（老 gp.csv 或 gp() 失败），会打印一行提示并退回密度。

流程：

1. **自举（第一代末）**：训练库不足 2 行时，按适应度取前 `nboot` 个结构，`uspexkit.calc()` 用参考方法（SIESTA）算它们的能量/密度，自动建起 `<homePath>/<data>/{feature_mlp.csv, feature.csv, structures.traj}`。这一代 GP 步整体跳过（无任何结构失败）。
2. **逐代生长（第二代起）**：`CalcFitness` 里按 `mode` 规则（EI / 不确定度 / 随机）取 top-`k`，再过 `u`（±1.96σ）与 `den` 两道门，选中的结构同样送参考方法打标签 → 追加进训练库。
3. **重训**：确实有新行入库才清理模型缓存——这些 pickle（`gpr_density.pkl` / `gpr_energy.pkl` / `rfr.pkl`，散落在 `CalcFold*`、`results*`、`data/` 三处）只按内核版本做戳，不清就不会因为数据变多而重训。随后 `uspexkit.pred()` 重预测并把新值回写适应度。

数据真的在长吗？看日志里这两行：

```
  GP training set has only 0 labelled row(s) in /home/feng/tnt/data_exp: seeding it with the best structures of generation 1
  GP training set: 0 -> 2 rows; dropped 3 cached model(s)
```

每代结束还会打一行学习曲线，**这一行是判断"学习到底有没有在发生"的依据**（被门挡住、一个标签都没打时是 `+0`）：

```
  GP learning curve (gen 3): rows 216 -> 220 (+4)  LOO RMSE density=0.0312 g/cm^3  energy=0.4123 eV
```

`LOO RMSE` 用帽子矩阵恒等式从已拟合模型的 Cholesky 因子直接算（Rasmussen & Williams 5.12，无需重拟合），超参是全程数据上拟合的那套，所以偏乐观一点；库超过 2000 行或找不到 uspexkit 缓存的模型时显示 `n/a`。

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

### AntiSeeds 反种子（可选）

把**已经过度采样**的结构指纹区域设为排斥源：与反种子指纹越像的结构，适应度被抬得越高（越不容易被选中做父代），从而把搜索面推开。对应 `uspex-gp` 的 `pickAntiSeeds.m` + `AntiSeedsCorrection.m`，在 `CalcFitness_310` 之后、`FitnessRankingCorrection` 之前生效。

**用法**：把不希望再采到的结构（若干个 POSCAR 顺次拼接）放进 `<homePath>/AntiSeeds/POSCARS`，启动时（新跑和重启都会）读入并算出各自指纹作为排斥源。

```
0.5   : antiSeedsMax         % 惩罚强度系数；0（默认）表示整个 AntiSeed 关闭
0.02  : antiSeedsSigma       % 惩罚宽度系数
3     : antiSeedsActivation  % |值|：>0 从该代起把全种群都加为反种子，<0 只加最优结构
```

惩罚公式（`AntiSeedsCorrection.m:57`）：每个种群成员的适应度加上

```
Δ = Max · exp( −cos_dist² / (2·Sigma²) )
```

其中 `cos_dist` 是该结构与反种子指纹的余弦距离：完全一样（距离 0）吃到满额 `Max`，差异越大衰减越快。

- `Max` 每代自适应：**当前值 = `antiSeedsMax` ×（最优 `bestFrac` 部分适应度的均值 − 最小值）**，即随适应度展宽而涨；`generation > antiSeedsActivation + 1` 后还有一半衰减保底，不会掉得比 `0.5×上一代` 更快。
- `Sigma` 每代自适应：**当前值 = `antiSeedsSigma` × 最优区结构两两指纹的平均余弦距离**，第 1 代用它统一标定所有预置反种子。
- 到达 `antiSeedsActivation` 后开始自学习：`>0` 追加全部种群指纹，`<0` 只追加当代最优那一个——这样搜索会自动避开已经挤在一起的区域。
- 状态存在 `<homePath>/Current_ANTISEEDS.pkl`（与 `Current_ORG/POP.pkl` 同族），重启时优先加载而不是重读 POSCARS；`gpevo clean` 会把它清掉。

**调参提示**：`antiSeedsSigma` 直接决定排斥范围，`Sigma ≈ antiSeedsSigma × 最优区平均指纹距离`。实测一组 TNT 类结构（到反种子的余弦距离依次 0.00 / 0.33 / 0.46）在 `antiSeedsMax=2` 下抬高量：

| `antiSeedsSigma` | 完全一样 | 比较像(0.33) | 很不一样(0.46) |
|---|---|---|---|
| 0.001（默认） | 1.000 | 0.000 | 0.000 |
| 0.02 | 1.000 | 0.000 | 0.000 |
| **0.5** | 1.011 | 0.143 | 0.036 |
| 1.0 | 1.327 | 0.903 | 0.729 |

默认的 `0.001`（以及 0.02 这种小值）实际是"**只排开几乎重复的结构**"，想让 replusion 有梯度、把比较像的结构一起推开，建议取 **0.3~1.0**（约 0.5 时区分度最好）；再大就变成全局都罚了。`antiSeedsMax` 控制抬高多少，`0`（默认）整个功能关闭。

> 与 MATLAB 的一处已知差异：参考实现给反种子算指纹时**不做分子内过滤**，而 gpevo 里分子体系（`molecule=1`，即 310 场景）的种群指纹会过滤掉分子内原子对。为保持一致，`pickAntiSeeds` 目前也按参考实现走（不过滤）；若发现距离被系统性抬高、惩罚偏弱，可在这条路径上补 `intra_map`。

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
0.15   : mutationRate       % LatMutation 应变 σ（正态分布，对齐 lattice_Mutation）
1.5    : softStepScale      % softmutate 步长缩放
```

RL 开启时的搜索范围固定为：角度 15°–90°、平移 0.1–0.8、应变 0.05–0.30、步长缩放 0.5–2.0（超出的设置值会被夹到区间端点）。

## 分析与绘图工具

```bash
# 多次运行的密度/能量分布对比（输出 compare_runs.png）
python compare_runs.py --dirs tnt/results2 uspex-gp/results1 --labels gpevo octave

# DFT 结果绘图 / 空间群覆盖度统计 / EI 日志
python plot_dft.py
python sg_coverage.py
python ei_select.py
```

## 命令行

```bash
gpevo               # 跑搜索（等价于 gpevo run，与旧行为一致）
gpevo clean         # 清理运行目录
gpevo clean -n      # 只列出会删什么，不动文件
gpevo clean -a      # 连 results*/ 一起删
gpevo clean -p DIR  # 清理别的目录（不是当前目录）
gpevo clean -f      # 目录里没有 INPUT.txt / CalcFold 也照样清
gpevo ei-report     # 逐代密度演化 + EI 收敛性（见下）
gpevo ei-report -p DIR --out PREFIX --trials 300
```

### `gpevo ei-report`

读一个运行目录里**已经落盘**的东西（不重算任何 GP/ReaxFF）——`results1/Individuals`（每代每个结构的密度）、`results1/density_predict.log`（GP μ/σ，**逐代追加、跨代累积**）、`results1/density.log`（DFT 参考）、`ei.log`（每代 EI 送 DFT 的 id）、`log`（`GP learning curve` 行）——画三张图并打一份逐代报告：

- **evolution**：全部结构散点 + 每代最优曲线 + best-so-far 阶梯，EI 选中的晶体画绿圈、DFT 验证值画红菱形；
- **convergence**：EI 选择趋势图，横轴是**代次**、纵轴是**密度**——每个被 EI 选中的晶体按所在代次落点（绿点），带 **GP 不确定度误差棒（±1.96σ）**；蓝阶梯线是该代次为止的最佳密度，红菱形是该晶体的 DFT 真值；
- **learning**：每代平均 GP σ、打标签数与 LOO RMSE（有记录才画）。

误差棒的 σ 来自 `density_predict.log`。早期版本每代都会**重写**这个文件，所以只有最后一代的晶体留有 σ，`convergence` 图上就只有最新那个点有误差棒（此时 `ei-report` 会打印 `σ for n/m EI-selected crystals` 提示）；现在它改为逐代追加，新跑的运行每个点都有误差棒。已经跑完的旧运行里，往代的 σ 没有落盘，无法事后补回。

真实运行上它直接回答"EI 到底有没有用"：某个 TNT·CL-20 运行（14 代、690 结构、26 个标签）上，EI 第 1 个标签就到 1.918、第 4 个到 1.94，随机基线要到第 ~8 个标签才追平，最终 1.9470 对随机的 90 分位 1.9460。

`gpevo clean` 删的是新一轮不该继承的东西：`AntiSeeds/`、`CalcFold*`（全部，不只是 `CalcFold1`）、`results*/<ID>/`（打标签留下的单结构目录）、`*.pkl` / `*.mat` / `*.backup`，以及 `still_running`、`USPEX_IS_DONE`、`octave-workspace`、`Warnings`、`ei.log` 这些状态文件（`ei.log` 是追加写的，留着会把两轮运行选中的晶体混在一个文件里，`gpevo ei-report` 也会一起画进去）。

永远不删：`Specific/`、`Seeds/`、`INPUT.txt`、`MOL_*`、`results*/`（除 `-a`）、GP 训练数据目录 `<data>/`。目录里既没有 `INPUT.txt` 也没有 `CalcFold*` 时会拒绝执行，确认无误再加 `-f`。

## 目录结构

```
gpevo/                        # Python 包
  FunctionFolder/
    USPEX/                    # 核心进化算法（310 路径）
    AbinitCode/               # GULP / SIESTA 接口与 I/O
    Submission/               # 作业提交（本地子进程 / 集群调度器）
    spacegroup/               # 对称性分析 + random_cell_mol（Cython 移植）
    sys/  Tool/               # 系统工具函数
  cli.py                      # 控制台入口（gpevo run / gpevo clean）
  tools/                      # 子命令实现（clean.py）
  runtime.py  inputparser.py  log_utils.py  matrix.py  importers.py
gpevo.py                      # 免安装入口（仓库根目录运行）
compare_runs.py  plot_dft.py  # 结果绘图工具
sg_coverage.py                # 空间群覆盖度统计
ei_select.py                  # EI 日志
example/  Specific/  train/  ebind/  docs/   # 示例、力场与文档
```

## 特征工程结论（有对照实验支撑）

想知道"把结构指纹并入 GP 的 10 维特征能不能提精度"，答案是否定的，完整过程与脚本见
[`docs/feature_ablation.md`](docs/feature_ablation.md)：

- 216 个真实 DFT 样本、5 折 CV、用生产核（ARD）实测：基线 R² 0.650（密度）/ 0.602（能量），**并入指纹后没有提升，甚至略降**。
- **打乱指纹的对照给出同样（甚至更好）的分数** → 说明没有增量信号：10 维里的密度代理已与 DFT 密度 r=0.707，与指纹信息冗余。
- 过程中还记录了一个陷阱：多维 GP **必须用 ARD 长度尺度**，各向同性核会塌成"预测均值"（R²≈0），曾据此得出过相反的错误结论。

复现：`python docs/feature_ablation.py --data ~/tnt22/data22`

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
