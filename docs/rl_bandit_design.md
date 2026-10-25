# 算子级强化学习（Hierarchical Contextual Bandit）引入方案 — L2

> 状态：**设计稿，未动代码**。在不改变现有进化计算逻辑的前提下，
> 让 RL 同时学习 (a) **选哪个变异算子**，(b) **该算子用多大的连续变异参数**
> （旋转角幅度、晶格应变率、softmode 步长）。
> L1（只学算子配额）是本方案的退化特例。

## 1. 动机与定位

现有三种算子配额策略：静态比例（`AutoFrac=0`→`update_STUFF_old.m`）、
启发式自适应（`AutoFrac=1`→`update_STUFF.m` 第 34–101 行）、以及本方案（`RL_strategy=2`）。

`AutoFrac` 已是一个无状态、单步、硬编码的 multi-armed bandit（有动作/奖励/更新），
但只学"选谁"，且算子内部幅度是写死的。L2 把两层都交给数据：

- **外层（离散）**：在 6 个算子中选择 —— Thompson Sampling；
- **内层（连续）**：对选中算子的连续参数做贝叶斯探索 —— reward-weighted 高斯后验。

哲学：**GP/EI 在"结构空间"做贝叶斯优化；本方案在"算子 + 参数空间"做贝叶斯优化。**

## 2. 为什么不用 DQN / 策略梯度

单次搜索仅 50 代 × 200 ≈ 10⁴ 次动作，每次代价是一次 24 核弛豫；深度 RL 需 10⁶+ 样本。
本方案全程无模型、无梯度、约百行、样本效率高、可落盘审计、可无损回退。
跨任务先验（第 6 节）进一步缓解样本稀缺。

## 3. 动作空间

### 3.1 外层：6 个离散臂（与 `EA_310.m` 第 82–95 行一致，不含 TransMutate）

| 臂 a | 算子 | `howCome` 归因关键字（已核实） |
|---|---|---|
| 1 | Heredity_310 | ` Heredity ` |
| 2 | Random_310 | ` Random ` |
| 3 | Permutation_310 | ` Permutate ` |
| 4 | Rotation_310 | `Rotate` |
| 5 | LatMutation_310 | ` LatMutate ` |
| 6 | SoftModeMutation_310 | `softmutate` |

`keptBest/convexHull/Seeds/COPEX` 不参与归因。

### 3.2 内层：每个臂的连续参数（归一化到 u∈[0,1]，再映射物理范围）

| 臂 | 学的参数 | 当前来源 | 物理范围 [lo, hi] | 默认 |
|---|---|---|---|---|
| 4 Rotation | `rotationAngleMax` | **函数内硬编码 pi/2**（第 12 行） | [π/12, π/2] | π/2 |
| 4 Rotation | `translationMax` | **硬编码 0.5**（第 13 行） | [0.1, 0.8] | 0.5 |
| 5 LatMutation | `mutationRate`（应变 σ） | `ORG_STRUC.mutationRate` | [0.05, 0.30] | 现有值 |
| 6 SoftMode | `softStepScale`（步长缩放，作用于 normfac） | `ORG_STRUC.howManyMut` | [0.5, 2.0]（乘基准） | 1.0 |

- 其余臂（heredity/random/permutation）无可调连续参数，只走外层；
- 参数以 `u` 存（0–1，跨体系可比），运行时 `p = lo + u·(hi−lo)`（softStepScale 用指数映射）；
- 旋转两个硬编码变量需提升为 `ORG_STRUC.rotationAngleMax / .translationMax`，
  **默认值 = 现硬编码值**，不开启 RL 时行为不变（详见第 7 节）。

## 4. 奖励

对臂 a 上一代产出的每个结构 i：

```
r_i = w1·1[i 进精英池] + w2·clip((H_parentMean − H_i)/H_scale, 0, 1)
    w1 = 1, w2 = 0.5
```

- "进精英池"= ranking 前 bestFrac 且通过 `SameStructure_order` 新颖性去重（与 AutoFrac 同口径）；
- 损坏/分解/失败（H_i≥90000）记 0，天然惩罚坏算子/坏幅度。

每条奖励同时归属到：**外层臂 a** 与 **内层实际使用的参数 u**，两层各自更新。

## 5. 更新与决策

### 5.1 外层 Thompson Sampling

```
每臂 Beta(α_a,β_a)；结算: α_a←λ α_a+wins_a, β_a←λ β_a+(trials_a−wins_a)
决策: θ_a~Beta(α_a,β_a)，按 pop 次抽样 argmax 得整数配额
遗忘因子 λ = 0.9（应对种群收敛的非平稳性）
臂上限 0.6、关键臂下限 0.10（复用 update_STUFF 第 90–92 行规则），归一化后 sum(howMany)=pop
```

### 5.2 内层连续参数：reward-weighted 高斯后验（无梯度 Thompson）

对每个有连续参数的臂、每个参数 k，维护 reward 加权的在线高斯：

```
观测 (u, r)：以奖励为权重做加权矩更新，得后验 N(m_a,k, s_a,k²)，
以有效样本数 n_eff 收缩方差；遗忘因子同样乘 λ。
决策：u ~ 截断到[0,1]的 N(m, (κ/√n_eff)² + s²)
      —— 样本越少探索越宽，奖励高区域均值越集中。
```

- 每个后代实例独立抽参数可保持多样性；等价于一维连续 Thompson Sampling；κ=1.0（PoC）。
- 记录每个后代实际 (臂, u) 供下代结算。

### 5.3 执行顺序（每代）

1. `update_STUFF.m` 触发 → RL 模块先**结算上一代**（外层 + 内层）；
2. **抽样**：外层得配额，内层对连续臂抽参数；
3. 写回：`fracXxx/howManyXxx`（复用现有写回）+ 本代
   `ORG_STRUC.rotationAngleMax/translationMax/mutationRate`；softmode 通过 `softStepScale` 读取；
4. `EA_310.m` `eval(算子)` 时算子读本代参数 —— **主循环不变**。

> PoC 最小改动：每代对每个连续臂**抽一个代表参数**供该臂本代使用（先打通链路）；
> "每后代独立参数"作为紧随细化（只需把参数随个体写入 OFF_STRUC）。

## 6. 跨任务算子先验（仿 Seeds 文件夹）

- 固定文件夹 **`RLPriors/`**（先在工作目录找；找不到再查全局
  `~/.config/uspex-gp/RLPriors/`），放 `operator_prior.mat`；
- **有则读取、没有则跳过**，逻辑与 `pick_Seeds` 的 `exist('POSCARS')` 一致，不报错；
- 先验内容 = 外层 `α,β`（伪计数）+ 内层各参数加权高斯 `m,s,n_eff`；
- 加载即作为伪计数/先验矩与本任务初值**融合**（不覆盖，权重可配），
  不相关先验不会强行带偏，随本任务数据增多其影响自动稀释；
- 任务结束可**导出**后验到 `RLPriors/operator_prior.mat`（开关控制），形成跨任务积累；
- 全程 `try/catch`：先验损坏/版本不符 → 告警并按均匀先验启动。

## 7. 代码改动清单（可开关、可回退）

| 文件 | 改动 |
|---|---|
| `createORGDefault.m` | 新增默认 `RL_strategy=0`；`rotationAngleMax=pi/2`、`translationMax=0.5`、`softStepScale=1.0`（第 88 行 AutoFrac 附近） |
| `createORG_EA.m` | 解析 `RL_strategy` 及先验/导出开关，仿第 41–43 行 |
| `createORGStruc.m` | 白名单元组加入新字段（仿第 9 行 AutoFrac） |
| `RL_agent.m` | **新增**：`RL_agent(action)`，load/save 状态、结算、外层 TS + 内层高斯、配额约束、先验融合 |
| `RL_load_prior.m` / `RL_export_prior.m` | **新增**：`RLPriors/` 读取/导出，仿 Seeds 的 exist 检查 |
| `update_STUFF.m` | 第 14 行分支加第三支：`RL_strategy==2` 调 `RL_agent`，**复用现有第 87–101 行**写回 |
| `Rotation_310.m` | 第 12–13 行硬编码改读 `ORG_STRUC.rotationAngleMax/translationMax`（默认=现值） |
| `SoftModeMutation_310.m` | normfac 处乘 `ORG_STRUC.softStepScale`（默认 1，行为不变） |
| `INPUT.txt`（示例） | 加 `2 : RL_strategy`（缺省 0，现状不变） |
| `docs/` | 本文件 + 流程图（同 `uq_active_learning.excalidraw` 风格） |

**原则**：`RL_strategy=0` 时路径与现在逐行一致；新参数字段取默认值时，
Rotation/SoftMode 数值与硬编码完全相同。RL 模块任何异常都降级为 AutoFrac/静态。

## 8. 落盘格式

`rl_agent_state.mat`（工作目录根，`safesave`）：

```
RL.alpha(1,6) RL.beta(1,6)        # 外层 Beta（初值 1）
RL.inner(a).param(k).m s n_eff    # 内层加权高斯
RL.forget=.9  RL.kappa=1.0  RL.version
RL.history(:) # 每代: generation, context, f, trials, wins, 各臂实际u（审计/作图）
```

仅标量与小数组，避开大矩阵与 shell ARG_MAX，不通过命令行传数组。

## 9. 验证计划（先离线、后廉价在线）

1. **外层离线**：静态 6 臂（固定不同胜率）→ ~20 代内集中到最优臂；
   非平稳（中途交换最优臂）→ λ=.9 能跟踪、λ=1 不能。
2. **内层离线**：造"奖励随参数单峰"的一维环境 → 高斯均值收敛到峰、探索宽随 n_eff 收窄。
3. **归因单测**：合成含 6 类 `howCome` 的 POP，trials/wins 正确，keptBest/Seeds 不计入。
4. **先验单测**：有/无 `RLPriors/operator_prior.mat`，断言融合数值正确、缺失不报错、损坏降级。
5. **配额/参数边界**：极端后验下断言上下限、`sum(howMany)=pop`、u 截断在 [0,1]。
6. **廉价在线 PoC（当前 GULP+ReaxFF，缩小种群）**：
   复制当前目录为最小测试目录，`populationSize/initialPopSize` 调小（建议 16）、3–4 代；
   三组对比 static / AutoFrac / RL，分别在有/无有效先验下跑；
   确认状态文件、先验读取、参数确实传入 Rotation/Lat/SoftMode、能正常结束与导出。
7. **正式评价**：best-so-far 焓曲线/AUC、配额与参数轨迹、单位弛豫精英结构数；
   多随机种子下稳定优于 AutoFrac 才默认开启。

## 10. 已确认的决策

- 采用 **L2**（同时学算子与连续参数）；
- λ = 0.9；w1 = 1 / w2 = 0.5；臂上限 0.6；
- 支持**从 `RLPriors/` 文件夹加载跨任务先验，有则读、无则跳过**（仿 Seeds）；
- 在线 PoC 用**当前 GULP+ReaxFF 链路、缩小种群**。

## 11. 实现前仍需确认

1. 全局先验路径 `~/.config/uspex-gp/RLPriors/`、文件名 `operator_prior.mat` 是否合适？
2. 任务结束是否**默认导出**后验（自动积累），还是默认不导出、用开关控制？
3. PoC 复制为 `~/uspex-gp-poc`（不动正式目录）、种群 16、4 代是否合适？

## 12. 路线图

- **L2（本方案）**：分层 bandit，学算子 + 连续参数，跨任务先验。
- **L2.5**：接入上下文（归一化代数、焓统计、goodFrac、S_order、组成熵）做 contextual（LinTS/线性奖励塑形）。
- **L2.7**：参数细化为"每后代独立"，扩展更多可调参数（softmode 频率选择权重等）。
- **L3（暂缓）**：从晶体指纹直接到动作的深度策略，需先用 L2/L2.5 跨任务积累离线数据。
