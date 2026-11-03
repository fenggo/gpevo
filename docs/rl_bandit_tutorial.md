# gpevo 强化学习模块教程（原理 + 使用）

> 配套设计文档：[`rl_bandit_design.md`](rl_bandit_design.md)（含更完整的理论推导与路线图）。
> 本文件面向**使用者**：讲清楚它是什么、为什么这样设计、怎么在 `INPUT.txt` 里开关，
> 以及如何解读结果。所有结论都对应到具体代码行，方便溯源。

---

## 1. 这是什么

gpevo 的进化搜索（USPEX 风格的 310 流程）每代都要决定：

1. **6 个变异算子各占多少配额**（遗传 Heredity / 随机 Random / 置换 Permutate / 旋转 Rotate / 晶格突变 LatMutate / soft 突变 softmutate）；
2. **连续变异幅度取多大**（旋转角上限、刚体平移上限、晶格应变率、soft 步长缩放）。

传统做法是**静态比例**（`AutoFrac=0`）或**启发式自适应**（`AutoFrac=1`）。
强化学习模块把这两件事都交给一个**分层 contextual bandit** 在线学习：

- **外层（离散）**：在 6 个算子间用 Thompson Sampling 学配额；
- **内层（连续）**：对 Rotation / LatMutate / softmutate 三个算子的幅度做 reward-weighted 高斯后验采样。

一句话：**GP/EI 在「结构空间」做贝叶斯优化，RL 在「算子 + 参数空间」做贝叶斯优化**。

开关由 `RL_strategy` 控制，默认 `0`（关闭，行为与传统流程逐行一致）。
模块入口：`gpevo/FunctionFolder/USPEX/src/RL_agent.py`，在
`gpevo/FunctionFolder/USPEX/src/update_STUFF.py:146`（`RL_strategy >= 1` 时）被每代调用。

---

## 2. 原理

### 2.1 为什么用 bandit，而不是 DQN / 策略梯度

单次搜索一般只有约 `50 代 × 200 个体 ≈ 10⁴` 次动作，而每次动作的代价是一次
多核力场弛豫。深度 RL 需要 `10⁶+` 样本才稳定，在这里样本量严重不足。

本方案是**无模型、无梯度、约 350 行、样本效率高、可落盘审计、可无损回退**的
contextual bandit —— 正好匹配「少样本 + 高代价」的晶体结构搜索场景。
跨任务先验（第 2.6 节）进一步缓解样本稀缺。

### 2.2 两层动作空间

**外层 6 个离散臂**（`RL_agent.py:21` 的 `KEYS`），与 310 流程的算子一一对应：

| 臂 a | 算子 | `howCome` 归因关键字 |
|---|---|---|
| 1 | Heredity（遗传） | `Heredity` |
| 2 | Random（随机） | `Random` |
| 3 | Permutation（置换） | `Permutate` |
| 4 | Rotate（旋转） | `Rotate` |
| 5 | LatMutate（晶格突变） | `LatMutate` |
| 6 | softmutate（soft 突变） | `softmutate` |

归因靠每个个体身上的 `howCome` 字段反推它来自哪个算子（`which_arm`，`RL_agent.py:92`）。
`keptBest / convexHull / Seeds` 等不参与归因（其 `howCome` 匹配不到任何臂，被跳过）。

**内层每个臂的连续参数**（归一化到 `u ∈ [0,1]`，运行时再映射回物理量）：

| 臂 | 参数 | 物理范围 [lo, hi] | 映射 |
|---|---|---|---|
| 4 Rotate | `rotationAngleMax`（角度上限） | `[π/12, π/2]`（15°–90°） | 线性 |
| 4 Rotate | `translationMax`（刚体平移上限） | `[0.1, 0.8]` | 线性 |
| 5 LatMutate | `mutationRate`（应变 σ） | `[0.05, 0.30]` | 线性 |
| 6 softmutate | `softStepScale`（步长缩放） | `[0.5, 2.0]`（乘基准） | **指数** `exp` |

> 注意：Rotation 实际施加角不是固定的 `±rotationAngleMax/3`（旧版）。
> 经 MATLAB 对拍修正后，刚体旋转每轴角度 = `±rotationAngleMax * ref / b[loop]`
> （`Rotation_310.py:136`），最轻轴可转满 `rotationAngleMax`（默认 90°），重轴
> 按比例缩小。此外若模板声明了柔性二面角，还会叠加**基团扭转**（±90°/±60°/±30°
> 渐进随机，见 [`rotation_operator.md`](rotation_operator.md)）。

每个连续参数在内部维护一个在线高斯 `{m, s, neff}`（`RL_agent.py:109` 的 `_fresh`），
`s` 在代码里不参与更新，方差由有效样本数 `neff` 决定（见 2.4）。

### 2.3 外层 Thompson Sampling

每臂一个 `Beta(α_a, β_a)`，初始 `α = β = 1`（均匀先验）。

**结算上一代**（`RL_agent.py` 主循环）：

```
对每个来自臂 a 的个体 i：
    trials[a] += 1
    reward_i  > 0  →  wins[a] += reward_i
```

**更新后验**：

```
α_a ← λ·α_a + wins_a
β_a ← λ·β_a + (trials_a − wins_a)
```

`λ = RL_forget`（默认 `0.9`）是遗忘因子，让旧统计指数衰减，以跟踪种群收敛带来的
非平稳性（`λ = 1` 会记一辈子，`λ < 1` 才能换策略）。

**抽下一代配额**（`RL_agent.py` 的 `for _ in range(popSize)`）：

```
每个臂抽 θ_a ~ Beta(α_a, β_a)，α/β 各抽一个 Gamma 再相除
本轮选 θ 最大的臂，计数 +1
```

这样按种群规模抽 `popSize` 次，自然得到各臂整数配额 `f[a]`。

**配额约束**（保证搜索不塌缩）：

- 关键臂下限：`f[1] ≥ 0.10`（heredity）、`f[2] ≥ 0.10`（random）、`f[6] ≥ 0.10`（softmutate）；
- 单臂上限：`safe_fractions`（`RL_agent.py:183`）把每个 `f[a]` 夹到 `RL_armCap`（默认 `0.6`），
  超出的部分按比例补给未达上限的臂，最终归一化使 `sum(howMany) = populationSize`。

### 2.4 内层连续参数：reward-weighted 高斯后验

对每个有连续参数的臂、每个参数 `k`，维护 reward 加权的在线高斯（无梯度 Thompson）：

**结算时更新**（`RL_agent.py` 主循环的 `if reward > 0 and arm in (4,5,6)`）：

```
n_old = λ · neff
n_new = n_old + reward
m ← (m_old · n_old + u · reward) / n_new
neff ← n_new
```

即：用奖励当权重做加权矩更新，奖励越高的 `u` 越把均值往那边拉。

**决策时抽样**（下一代的幅度，`RL_agent.py` 内层循环）：

```
sd = κ / sqrt(neff) + 0.05
u  ~ clip( N(m, sd²), 0, 1 )
p  = lo + u · (hi − lo)        # softStepScale 用 exp 映射
```

`κ = RL_kappa`（默认 `1.0`）是探索系数：**样本越少 `neff` 越小、`sd` 越大、探索越宽**；
某区域奖励高则 `m` 集中、探索自然收窄。等价于一维连续 Thompson Sampling。

### 2.5 奖励设计

对臂 `a` 上一代产出的每个结构 `i`：

```
H_i = 个体 Enthalpies 末位（失败/分解/损坏记 H_i ≥ 90000 的跳过）
reward_i = w1 · 1[i 进精英池] + w2 · clip((H_mean − H_i) / H_std, 0, 1)
           w1 = RL_w1 (默认 1.0)，w2 = RL_w2 (默认 0.5)
```

- 「进精英池」= 当代 `ranking` 前 `bestFrac`（默认 0.5）且通过新颖性去重的个体；
- `gain` 项衡量该结构比种群平均好多少（用焓的 z-score 截断到 [0,1]）；
- 损坏/分解/失败（`H_i ≥ 90000`）记 0 奖励，**天然惩罚坏算子 / 坏幅度**。

每条奖励同时归属到「外层臂 a」和「内层实际使用的参数 u」，两层各自更新。

### 2.6 跨任务先验

状态文件 `rl_agent_state.pkl` 存在**运行目录根**，是单任务在线统计量。
跨任务先验放在 **`RLPriors/operator_prior.pkl`**——优先找工作目录，找不到再查
`~/.config/uspex-gp/RLPriors/operator_prior.pkl`（`RL_agent.py` 的 `_read_prior_file`）。

- **有则融合、无则跳过**：先验的 `α,β` 作为伪计数叠加（`α += max(0, prior.α − 1)`），
  内层的 `m, neff` 也并入；
- 不相关先验不会强行带偏——本任务数据一多，先验影响自动被稀释；
- 任务结束若 `RL_autoExport = 1`（默认），会把后验导出到 `RLPriors/operator_prior.pkl`，
  形成跨任务积累；
- 全程 `try/except`：先验损坏 / 版本不符 → 告警并按均匀先验启动，**不报错中断**。

---

## 3. 快速上手

### 3.1 开启 bandit

在 `INPUT.txt` 里加：

```
1      : RL_strategy    % 0=关；>=1 开启分层 bandit
0.9    : RL_forget      % 遗忘因子 λ
1.0    : RL_w1          % 奖励权重：入选精英集
0.5    : RL_w2          % 奖励权重：焓相对增益
0.6    : RL_armCap      % 单个算子占比上限
1.0    : RL_kappa       % 内层探索系数
1      : RL_autoExport  % 任务结束导出跨任务先验
```

`RL_strategy` 设成任意 `≥ 1` 的值即开启（设计稿里曾用 `2` 表示 L2 完整模式，
当前实现只判 `>= 1`）。`RL_strategy = 0` 时路径与原始流程完全一致。

### 3.2 连续参数入口（也作 RL 关闭时的固定值）

这四个参数无论是否开 RL 都能在 `INPUT.txt` 设；`RL_strategy = 0` 时**直接固定为你设的值**，
开启 RL 时则作为**学习的起点**（首次运行的 `u` 初值由这些值映射得到）：

```
45     : rotationAngleMax   % 旋转变异角度上限，单位「度」（内部转弧度）
0.3    : translationMax     % 旋转后的刚体平移上限
0.15   : mutationRate       % LatMutation 应变 σ（正态分布）
1.5    : softStepScale      % softmutate 步长缩放
```

开启 RL 时，搜索范围被夹到固定区间：角度 15°–90°、平移 0.1–0.8、应变 0.05–0.30、
步长缩放 0.5–2.0；超出区间的设置值会被夹到端点（见 `_fresh` 的 `_u` 映射）。

---

## 4. 只让 RL 学一个参数（`RL_inner`)

默认（`RL_inner` 缺省或 `all`）内层每代对**四个**幅度一起做 Thompson 采样。
想做单参数消融，用 `RL_inner` **逐个点名**，没点到的保持 `INPUT.txt` 里的固定值：

```
angle        : RL_inner   % 只学 rotationAngleMax（平移/应变/步长固定）
translation  : RL_inner   % 只学 translationMax
mutation     : RL_inner   % 只学 mutationRate
softstep     : RL_inner   % 只学 softStepScale
rotate       : RL_inner   % arm 4 的两个一起学（角度 + 平移）
angle,softstep : RL_inner % 多个用逗号/空格分隔（不要写成 "angle, softstep"）
none         : RL_inner   % 内层完全不学，四个都用 INPUT 的固定值
```

- 识别**不分大小写**，也认 `all / on / 1`、`none / off / 0`，以及别名
  `latmut / softmut / rotationangle` 等（见 `RL_agent.py:35` 的 `INNER_TOKENS`）；
- 拼错的名字会告警并退回 `all`（`RL_agent.py:84`）。

**「只测旋转角」的最小写法**：

```
1      : RL_strategy
angle  : RL_inner        % 内层只动旋转角
45     : rotationAngleMax % RL 的学习起点
0.3    : translationMax   % 以下三个写死为你要的固定值
0.15   : mutationRate
1.5    : softStepScale
```

> 两点提醒：
> 1. 外层 6 臂 bandit **仍在学算子配额**。想连算子配额也定住，只能 `0 : RL_strategy`
>    走纯固定值扫描。
> 2. 对照实验要换独立目录或先删 `rl_agent_state.pkl`（存在时 INPUT 的初值不再生效），
>    并把 `RLPriors/operator_prior.pkl` 与 `~/.config/uspex-gp/RLPriors/operator_prior.pkl`
>    挪开，否则跨任务先验会覆盖你的起点。

---

## 5. 状态与先验文件

| 文件 | 位置 | 作用 |
|---|---|---|
| `rl_agent_state.pkl` | 运行目录根 | 单任务在线统计量（`α,β`、`inner` 高斯、`history`），每代 `safesave` 落盘 |
| `RLPriors/operator_prior.pkl` | 运行目录或 `~/.config/uspex-gp/RLPriors/` | 跨任务先验，读入融合、任务结束导出（受 `RL_autoExport` 控制） |

`history` 每条记录一代的 `generation / trials / wins / f / u`，可用于审计和作图
（看配额轨迹、各幅度 `u` 的演化）。

控制台每代会打印两行：

```
RL bandit: f = [ 0.10 0.10 0.08 0.22 0.30 0.20 ]   % 六臂配额
RL bandit: inner = rotationAngleMax                 % 仅当 RL_inner 非 all 时
```

若加载了跨任务先验，还会打印 `RL bandit: loaded cross-task prior from ...`。

---

## 6. 怎么判断它在正常工作

1. 看每代 `RL bandit: f = [...]`：六个数之和应为 1，关键臂不会低于 0.10，单臂不超 0.60；
2. 看 `rl_agent_state.pkl` 的 `history`：随代数推进，好算子对应的 `wins/trials` 应上升，
   坏算子的配额应被压低；
3. 做单参数消融时，确认 `ORG.rotationAngleMax` 等属性确实在变（被内层改写），
   而未点名的参数保持 `INPUT.txt` 的常数；
4. 任何异常（先验损坏、状态反序列化失败）都会降级：
   `update_STUFF.py:146` 的 `try/except` 捕获后打印 `RL bandit disabled: ...`，
   退回原本的 `AutoFrac /` 静态配比，搜索不中断。

---

## 7. 参数速查表

| `INPUT.txt` 键 | 默认 | 含义 |
|---|---|---|
| `RL_strategy` | `0` | `≥1` 开启 bandit；`0` 关闭（传统流程） |
| `RL_forget` | `0.9` | 遗忘因子 `λ`，历史统计指数衰减 |
| `RL_w1` | `1.0` | 奖励权重：入选精英集 |
| `RL_w2` | `0.5` | 奖励权重：焓相对增益 |
| `RL_armCap` | `0.6` | 单个算子配额上限 |
| `RL_kappa` | `1.0` | 内层探索系数，越大探索越激进 |
| `RL_autoExport` | `1` | 任务结束导出跨任务先验 |
| `RL_inner` | `all` | 内层可动幅度点名（`all`/`none`/具体参数） |
| `rotationAngleMax` | `90`（度） | 旋转角度上限（RL 起点 / 固定值） |
| `translationMax` | `0.5` | 刚体平移上限 |
| `mutationRate` | `0.08` | 晶格应变 σ |
| `softStepScale` | `1.0` | softmutate 步长缩放 |

---

## 8. 常见坑

- **改了 `INPUT.txt` 的初值却没生效**：运行目录里已有 `rl_agent_state.pkl`，
  会从它恢复而不是重新读初值。清掉状态文件（及可能干扰的 `RLPriors`）再跑。
- **想固定某个幅度却一直变**：检查 `RL_inner` 是否意外包含了它，或没写 `none`。
- **`RL_strategy=0` 时四个连续参数完全不起作用？**：不会——它们始终作为固定值传入
  Rotation / LatMutate / softmutate，只是不参与学习。
- **跨任务先验把结果带偏**：先验是「伪计数」式叠加，不会覆盖本任务数据；若仍不放心，
  设 `RL_autoExport = 0` 并挪开 `RLPriors/` 即可。
