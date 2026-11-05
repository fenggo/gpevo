# gpevo 全盘审查：算法与方法改进建议

> 本文整理自一次全仓代码审查对话，回答「gpevo 还有什么算法 / 方法上的改进空间」。
> 状态：**§2.1/§2.2/§2.4/§2.5 已实施；§2.3（残差标准化）已于 2026-11-05 撤销**，
> 残差恢复原始特征空间距离、阈值恢复 10.0（见 §2.3 末尾的撤销记录）。
> §3 方法层面仍是待办建议。
> 下文所有 `file:line` 都是**实施前**提交 `c2ffa07` 的行号，实施后已整体后移；
> 请按函数名检索（`UNCERT_SCALE`、`_to_sigma`、`observed_density_incumbent`、
> `_setup_rng`、`_record_gp_calibration`、`_ei_preview`）。
>
> 姊妹篇（已讨论过、本文不重复）：
> - `ei_theoretical_basis.md` / `expected_improvement.md`：EI 理论依据、top-K vs q-EI、den 硬门 vs 约束 EI(cEI)、u 门、σ 校准前提。
> - `uq_active_learning.md`：UQ 主动学习闭环与核函数选型。
> - `feature_ablation.md`：结构指纹并入 10 维特征无效（含打乱对照），结论是「数据量比特征设计更值得投入」。
> - `fingerprint_method.md`：RDF 指纹与 Cython 加速。
> - `rl_bandit_design.md` / `rl_bandit_tutorial.md`：L2 分层 bandit 的设计与消融。
> - `molecule_damage_check.md`：分子损坏漏检与 4 条建议（**尚未实现，优先级最高**，见 §4）。
> - `rotation_operator.md` / `strain_matrix.md`：旋转算子默认行为、strainMatrix 谱系字段。

---

## 0. 一句话结论

当前最值得做的**不是加新算法**，而是先修 4 个口径 / 数值问题（§2.1–§2.4），再补一个可复现种子（§2.5）；
方法层面的改动（§3）多数会破坏与 MATLAB USPEX 的字符级 / 行为对拍，需要你先点头。

### 0.1 已实施（顺序：种子 → σ 口径 → f_best → bandit → residual）

| 条目 | 做法 | 关键决定 |
|---|---|---|
| §2.5 种子 | 复用 MATLAB 已有的 `fixRandSeed`：`createORG_System.py` 解析 `N : fixRandSeed`；`Start.py:_setup_rng` 新运行 `rng(seed+1)`、重启 `rng(seed+POP.generation)`，`0` 保持时钟种子；终端打印有效种子 | **不新增 `seed` 选项、不写 OUTPUT.txt**，因此不破坏字符级对拍；空间群生成器的 `RandSeeds` 由同一条 numpy 流产生，随之可复现 |
| §2.1 σ 口径 | `gp_ml.py` 引入 `UNCERT_SCALE=1.96` 与 `_to_sigma()`；`apply_gp_metrics` / `apply_pred_metrics` 同时存 `s.sigma_gp`（后验 σ，给 EI）与 `s.uq`（1.96σ，给 OUTPUT.txt / 误差棒 / `u` 门）；`_metrics_row` 与 `_read_gp_csv` 统一产出 σ；`gp_plot.py` 的 `_ei_preview` 用 `r["sd"]` 算 EI | 经确认，**`u` 门保留 1.96σ 标度**（`s[1]*UNCERT_SCALE > u`），现有 INPUT.txt 的 `u` 值无需重标、EI 触发行为不变；只修正 EI 公式里的数学错误 |
| §2.2 f_best | 抽出 `_best_observed()`、新增 `observed_density_incumbent()`；`_ei_preview` 接受 observed incumbent，状态报告与真实选择用同一个 `f_best`；`_select_by_ei` 每代打印 `f_best_observed` 与 `max(μ)` 及「EI 退化为探索」告警；`_record_gp_calibration()` 用**DFT 标注前**的 μ 与 `density.log` 配对，累计 ≥3 条后打印 `y ~ μ` 回归 | **没有改采集目标本身**（μ 修正 / GP 标度 incumbent / 标度无关 EI 三个选项都保留待定）；`optType=1` 时状态报告仍用 `max(dgp)`，因为它的 incumbent 是能量、与密度不同量纲 |
| §2.4 bandit reward | `RL_agent.py` 在 Beta 更新前把 `wins/(w1+w2)` 归一化到 `[0, trials]` 的 fractional successes；内层连续参数的高斯加权更新仍用原始 reward；`history` 同时记录原始 `wins` 与实际喂给 Beta 的 `successes` | 只修数值饱和，不改奖励定义；补单测断言 `beta` 不触底 |
| §2.3 residual **（已撤销）** | 曾经把 `gp()` / `pred()` / `calc()` 的残差改成标准化空间距离、阈值改成 `RESIDUAL_CUT = 5.0`；2026-11-05 撤销，残差恢复 `sum((d-feature)²)` 原始空间距离，阈值恢复 **`RESIDUAL_CUT = 10.0`**。共享常量仍保留在 `gpevo/runtime.py`，`core.py` 的 `raw_residual()` 只额外产出 GP `predict()` 需要的标准化查询行 `X_` | 保留 EI 对 OOD 的硬剔除；optType 13 适应度、gp.csv / density_predict.log / OUTPUT.txt 的 Residual 列数值随之回到旧语义。撤销原因见 §2.3 末尾 |

> 验证：`tests/` 全量 **154 passed**；`plot_gp_generation` 已用 ct22 风格数据端到端跑通。
另外 `molecule_damage_check.md` 里那组结构合法性检查仍未落地，它比任何精度优化都更该先做。

---

## 1. 已经写过、本文不重复的内容

`ei_theoretical_basis.md` §6 已列出两处偏离教科书口径的改进：用 q-EI / Kriging Believer 替代 top-K 排序、
把 `den` 硬门改成乘性可行概率 (cEI)；§4 已强调 σ 校准是 EI 的命门。
`feature_ablation.md` §7 已给出「基线 R² 0.65 / 0.60，数据量是当前最大杠杆」的实证结论。
下文只补这些文档**没有覆盖**的点。

---

## 2. 新发现的硬问题（有代码 / 数据实证）

### 2.1 EI 把 1.96σ 当成 σ 用（尺度错误，影响所有 EI 排序）

证据链：

| 环节 | 位置 | 值 |
|---|---|---|
| GP 写出不确定度 | `uspexkit/core.py:342,350,374,377` | `1.96*std_prediction[0]`（95% 置信半宽） |
| 存到个体 | `gp_ml.py:265` | `s.uq = result["uncertainty"]`，未除 1.96 |
| 当作 σ 列 | `gp_ml.py:346` (`_UNCERT=5`)、`_metrics_row` | 第 5 列原样进 `table` |
| 进 EI 公式 | `gp_ml.py:492,497` | `z = diff/sigma`、`sigma*pdf_z` |
| 状态图预览 | `gp_plot.py:78,101` | `sigma = _f("uncertainty")` 直接喂 `_expected_improvement` |
| 作者留下的意图 | `gp_plot.py:35` | `UNCERT_SCALE = 1.96  # s.gp['uncertainty'] already is 1.96 * std` —— **全仓仅此一处定义，从未被使用** |

`_expected_improvement` (`gp_ml.py:478-497`) 是标准 EGO 闭式解，其中 `Z=(μ-f_best)/σ`
里的 σ 必须是**后验标准差**。现在传入的是 1.96σ，于是：Z 被压小 1.96 倍、`Φ(Z)` 被系统性抬高、
探索项 `σφ(Z)` 被放大 1.96 倍。`u` 门 (`gp_ml.py:667`) 比的也是 1.96σ，
所以 `u=0.04` 实际对应 `σ_GP > 0.0204`，与文档里「σ 下界」的说法差一个因子。

**已实施（§0.1）**：`apply_gp_metrics` / `apply_pred_metrics` 存 `s.sigma_gp = uncertainty/1.96`
（`_to_sigma()`），`_metrics_row` / `_read_gp_csv` / `_ei_preview` 全部改用后验 σ；
显示与误差棒继续用 `s.uq = 1.96σ`。**`u` 门按确认结果保留 1.96σ 标度**
（`s[1]*UNCERT_SCALE > u`），故现有 `u` 值无需重标；EI 排序仍会变（这是修 bug 的目的）。
`gp_plot.py` 里原先定义了却从未使用的 `UNCERT_SCALE` 已改为从 `gp_ml` 导入并真正使用。

### 2.2 EI 的 incumbent 与 GP 预测不在同一口径（ct22 现象的根因补充）

`_observed_incumbent` (`gp_ml.py:522-565`) 按教科书取 `feature.csv` 里的 **DFT 实测最优**作
`f_best`，理由是「incumbent 必须是已观测值」。这本身没错，但前提是代理与实测在同一标度上。

你贴的 ct22/results2 日志里：

- 实际选择用的 incumbent：`EI incumbent: observed den_gp = 1.8991`（DFT 实测最优密度）；
- 状态报告里的 `f_best (GP density) = 1.8032 [#825]`（`gp_plot.py:_ei_preview` 的 `f_best = max(dgp)`，**GP 预测标度**）。

两者差 ~0.1。后果有两层：

1. 对当前代所有候选 `μ ≤ 1.8032 < 1.8991`，故 `diff = μ - f_best < 0`、`Z<0` 恒成立，
   利用项 `(μ-f_best)Φ(Z)` 几乎不起作用，EI 退化成 `≈ σφ(Z)`（近似不确定性采样）。
   再叠加 §2.1 的 1.96σ 放大，探索倾向被双重放大。
2. **预览和实际选择用的是两个不同的 `f_best`**：预览用 GP 标度 1.8032，实际选择用 DFT 标度 1.8991。
   这解释了为什么报告里能排出 `EI top-1: #829`、而实际 `gp_active_learning` 却整代无标注。
   > 更正：上一轮对话里说「预览排全种群、实际只看当代」是**错的**。`gp_plot.collect_records(POP)`
   > 遍历的就是 `G["POP_STRUC"]` 当前代的 `POPULATION`，与实际选择的候选集是同一批结构。
   > 两个 `f_best` 不一致（再加 σ 口径错误）才是全部原因。

   实测补充：统一 incumbent 后，ct22 风格数据下 `#829` 的 EI 从预览里的 0.02099 掉到 ≈0
   （μ 比 incumbent 低 ~0.1、σ 又按 §2.1 缩小 1.96 倍，`Φ(Z)` 与 `φ(Z)` 同时趋零）。
   这不是新问题，而是把「EI 在这代其实几乎没有区分度」如实显示出来。

> 不能据此断言「GP 系统性低估 0.1」——也可能 1.8991 来自数据库里确实更好的结构、当前种群本就不如它。
> 但**无论哪种解释，对 EI 的影响相同**：incumbent 超出候选 μ 的值域时，EI 只剩探索项。

**已实施（§0.1）**：两个 `f_best` 已统一；`_select_by_ei` 每代打印 `f_best_observed`、
`max predicted this generation`、gap 与退化告警；`_record_gp_calibration()` 用标注前的 μ
与 `density.log` 的 DFT 密度配对，累计 ≥3 条后打印 `y = slope*mu + intercept`、
`mean(mu-y)`、RMSE（刻意不用 `density_predict.log`，那是把新标签加进训练集后的**事后重预测**，
属于 in-sample，会人为美化校准）。

**仍待定**：(a) 修正 μ 的 bias、(b) 改用 GP 标度 incumbent、(c) 标度无关采集，三者都未动；
先看几代校准输出确认是否真有系统偏置再选。

### 2.3 `residual` 是原始特征空间的平方欧氏距离，被能量列支配

`core.py:204`：`res = np.sum(np.square(d - feature), axis=1)`，其中 `d = data[:,1:]`
是**未经标准化**的 10 维 ReaxFF 特征。对仓库里两份真实数据库实测各列对平均成对平方距离的贡献：

| 数据库 | 行数 | 能量列占比 | 密度列(第 10 列)占比 |
|---|---|---|---|
| `example/data22` | 212 | 第 2 列 52.6% + 第 5 列 25.7% | ~0.0% |
| `dnmt/data` | 392 | 第 1 列 40.0% + 第 2 列 58.2% = **98.2%** | ~0.0% |

即 `res>10` 实际上几乎只是「ReaxFF 总能量离最近训练点有多远」，而不是「离训练流形有多远」。
这个量被用在：

- `core.py:238` 的 OOD 回退（GP 预测被 RF 或 `density*d_scaler` 取代）；
- `gp_ml.py` 的 `row[1] > 10` 过滤（`_select_by_ei`、`metrics_table`）；
- `CalcFitness_310.py:53-59` 的 **optType 13 适应度本身就是 residual**；
- `gp_plot.py:34,95,152,205,274` 的内 / 外推判定与图 D。

阈值 10 不是标度无关的，换体系、换 ReaxFF 参数化就失效。更矛盾的是：主动学习本该去标注分布外结构，
而 `residual>10` 恰好把这类结构从 EI 里剔除。

**撤销记录（2026-11-05）**：标准化残差在真实运行里改变了 OOD 判定与 Residual 列语义，
按用户要求**全部回退**：

- `core.py` 的 `standardized_residual()` 改为 `raw_residual(feature, d, scaler)`，
  残差重新用 `sum((d - feature)²)`，即**原始（未标准化）10 维 ReaxFF 特征空间**；
  函数仍产出 `X_ = scaler.transform(feature)`，但那只是 GP `predict()` 需要的标准化查询行，
  与残差无关。
- `gpevo/runtime.py` 的 `RESIDUAL_CUT` 从 `5.0` 改回 **`10.0`**；`core.py` 的 OOD 回退
  （`res[imin] > RESIDUAL_CUT`）与 DFT 触发（`res[imin] < RESIDUAL_CUT`）**统一用 10**，
  不再有 `< 5.0` 的硬编码。`gp_ml.py` / `gp_plot.py` / `ei_report.py` / `ei_select.py`
  继续导入该共享常量，无需改动。
- `gp()` / `pred()` / `calc()` 三处调用点同步改回原始残差；测试改为断言
  `raw_residual` 等于朴素平方距离、且不再对列尺度不变。

也就是说，本节开头写的「残差被能量列支配、阈值 10 不是标度无关」这个**问题重新存在**，
属于已知且接受的状态：换体系 / 换 ReaxFF 参数化时阈值需重新标定。

**仍待定**：「OOD 一律不进 EI 降级为降权」以及马氏 / 核距离替代欧氏距离，两者都未做。

### 2.4 RL bandit 的 reward 未归一化，Beta 后验会饱和

`RL_agent.py:256-258,271-272`：

`reward = w1*in_elite + w2*gain`，默认 `w1=1, w2=0.5`，`gain∈[0,1]` ⇒ `reward∈[0,1.5]`；
但更新用的是 Beta-Bernoulli 口径 `alpha += wins`、`beta += trials - wins`，要求 `wins≤trials`。

数值验证（6 个臂、每臂 10 次、平均 reward 1.5）：

```
wins [15. ...]  trials-wins [-5. ...]
beta after update [0.001 ... 0.001]   <- 全部被 max(...,1e-3) 夹到下限
theta mean ~ alpha/(alpha+beta) = [0.9999 ...]
```

只要某臂平均 reward > 1（进精英池比例高时很常见），`beta` 就会被夹到 1e-3，该臂 `θ≈1`
永久占优、不再探索，且遗忘因子 `λ` 救不回来（`λ*beta` 仍很小）。

**已实施（§0.1）**：Beta 更新改用 `successes = min(wins/(w1+w2), trials)`，
保证 `trials - successes >= 0`；内层高斯更新仍用原始 reward；`history` 增加 `successes`
字段便于审计。补了 `test_reward_above_one_does_not_clamp_beta`：6 个全精英子代
（raw wins ≈ 7.1 > trials 6）时断言 `beta > 0.5` 且不触底。

### 2.5 没有可复现的随机种子

`Start.py:24-30` 用 `time.time()` 播种 `np.random` 与 `random`，种子既不写进 OUTPUT.txt、
也不能从 INPUT.txt 指定；`spacegroup/random_cell_mol.py:877-879` 还有一条独立的
`_random.Random`，兜底种子是 `time.time()*1e6 ^ id(sys)`。

后果：无法做**配对比较**（同一随机数流下 static / AutoFrac / RL 对照，或 `k` / `den` 网格），
而 `rl_bandit_design.md` §9 的验证计划明确要求「多随机种子下稳定优于 AutoFrac」——目前做不到。
这是方法研究里性价比最高、风险最低的一处改动。

**已实施（§0.1）**：改用 MATLAB 已有的 `fixRandSeed`（INPUT 关键字，ORG 字段是 `fixRndSeed`），
没有另造 `seed` 选项，也没有往 OUTPUT.txt 写新行——**这是对拍补齐，不是破坏**。
`Start.m:59` / `Initialize.m:24` / `PickUp.m:29` 的 `rng(fixRndSeed+generation, 'twister')`
被逐条移植；`fixRndSeed>0` 时 `symope_310.py` 才会写 `RandSeeds`，空间群生成器那颗独立 RNG
的兜底 `time.time()^id(sys)` 在 gpevo 路径里因而不会再被命中。
`rl_bandit_design.md` §9 现在可以用不同整数 `fixRandSeed` 做 common-random-numbers 配对。

### 2.6 潜在 bug：`f=2` 时训练 / 查询特征缩放不一致

`core.py:190-201`：`f==2` 把**查询** feature 的能量列除以 `nmol`，注释明说
「训练数据 d 保持原始总能量不动」；`pred()` 的 `core.py:708,756` 同样。若启用 `f=2`，
查询向量（~ -67）与训练矩阵（~ -538）不在一个标度，GP 预测与 `res` 全部失真。
`createORG_AbinitCode.py:16` 的 `_GP_OPTIONS` 不含 `f`，所以 gpevo 进程内路径恒为 `f=1`，
但 CLI 可以走到。建议要么对训练特征做同样归一化，要么在 GP 路径里禁用 `f=2`。

---

## 3. 方法层面（多数会破坏 MATLAB 对拍，需先确认）

### 3.1 适应度只用 GP 均值 μ，没有不确定性惩罚 → 代理被「钻空子」

`use_gp_density.py` 只要 `density_gp` 有限就覆盖几何密度；`CalcFitness_310.py:43,49`
于是 `optType 11/12` 的适应度完全由 GP **均值**决定。EA 的选择压力（`build_tournament`
权重 `(N-p)^2`，`variation_common.py:113-121`）+ 精英保留会持续把搜索推向 GP 高估的区域。
每代只有少数结构拿到 DFT 标签，ct22 那种「整代无标注」时模型还会停在旧版本上。

**建议**（按侵入性排序）：
1. 至少给每个结构打一个「模型版本 / 代数」戳，日志里区分「本代重预测」与「沿用旧模型」；
2. 适应度改 `μ − κσ`（密度最大化的 LCB，κ 可随代数衰减），或 `μ` 与几何密度加权混合；
3. 每代强制对 top-1 做一次 DFT 复核，防止「GP 最优 ≠ 真实最优」被精英池固化。

这与 `ei_theoretical_basis.md` §4「σ 必须校准」是同一个问题的两面：那里说采集函数，
这里说适应度。

### 3.2 保证每代最小标注量，避免整代判死

ct22/results2 的症状：`k=1` + `den=1.79` 贴着 GP 最优 1.8032 → 唯一 EI 候选低于密度门 →
`every top-1 crystal is below the density gate ... GP training set unchanged` →
`GP learning curve: 228 -> 228 (+0)`。`ei_theoretical_basis.md` §6.2 已给出 cEI 的治本方案；
工程上更省事的兜底是：**EI 选空时自动降级**为 `mode=uncertainty` 的 top-1（或最高 σ）并放宽
`den`，保证闭环不至于断粮。

### 3.3 Anti-seeds 的「最优集合」用的是 population 顺序而非 ranking 顺序

`EA_310.py:79-80`：`AntiSeedsCorrection(fitness)` 在 `FitnessRankingCorrection` **之前**调用，
此时 `fitness` 仍是种群（插入）顺序、未排序。而 `AntiSeedsCorrection.py` 用
`out[:fit_len]` 当作「best structures」来估 `anti_seeds_max` 与成对指纹距离 `sigma`，
并按 `population[:fit_len]` 取指纹。也就是说「精英区」取的是前 N 个**任意**结构。

需先核对 MATLAB 的调用顺序是否同样如此；若是对拍一致，至少修正注释 / 文档口径，
否则 anti-seed 的幅度与宽度会是噪声。

### 3.4 去重与多样性维持偏弱

- `degradeSimilar` (`sys/degradeSimilar.py`) 用 `SameStructure`；而 `_auto_frac`
  (`update_STUFF.py`) 用的是 `SameStructure_order`，后者（`sys/SameStructure_order.py`）
  **完全不看指纹**，只比「平均有序度 + 每离子焓」是否落在 `0.075 / 0.015` 内。
  两种「相同结构」判据并存，且后者会把焓相近、堆积不同的结构合并 → 算子奖励归因可能失真。
- `fingerprint_utils.py:541,547` 算出的 `struc_entr` / `S_order` 只被 `Correlation`
  和 `SameStructure_order` 消费，**没有进入适应度**（`CalcFitness_310.py:44` 注释写
  「density combined with entropy」，实际是 `enthalpy/factor − 0.5*density`）。
- 陡峭的锦标赛 `(N-p)^2` + 精英保留 + 去重，容易早熟；当前没有逐代多样性指标。

**建议**：每代记录精英池内平均成对指纹余弦距离作为多样性曲线；必要时引入 crowding / fitness sharing，
或把 `struc_entr` 按 MATLAB 口径并入 `optType 12`。

### 3.5 GP 训练数据的卫生与在线校准诊断

`core.py:586-607` 的 `load_gaussian_process` 拿全表行直接拟合，无离群点过滤、无重复行检查、
无运行内的 held-out 校验；超参缓存只按 `_KERNEL_VERSION` 标记（数据增长由
`gp_ml.py:_invalidate_pkl` 负责丢弃，`rfr.pkl` 也在 `_PKL_NAMES` 里，这一点是对的）。
`gp_ml.py:_log_learning_curve` 会报 LOO RMSE，但只在缓存模型存在时。

数据侧值得先查：`dnmt/data/feature_mlp.csv` 前两列 std 达 126 / 152、range 369 / 477，
而 `example/data22` 同两列 std 仅 0.4 / 2.2——差异很大，需确认数据库是否混了不同组分 /
不同 ReaxFF 参数化（Δ-learning 假设单一映射）。结合 `feature_ablation.md` §7 的结论，
**先把标注量与数据同质性理清，比换核 / 加特征更有效**。

**建议**：每代输出一份轻量校准报告（标注行上 `y ~ μ` 的斜率 / 截距、残差直方图、重复 / NaN 计数），
并在数据库混入不同组分时告警。

---

## 4. 优先级建议（先做什么）

| 顺序 | 事项 | 类型 | 影响对拍 |
|---|---|---|---|
| 0 | `molecule_damage_check.md` 的跨分子成键检查 + 连通性聚类 | 正确性（防止损坏结构污染适应度 / 排序 / GP 训练集） | 否（本来就该挡） |
| 1 | ~~§2.1 σ 口径、§2.2 两个 f_best 统一 + 校准诊断~~ **已做** | 数值 / 口径 | 会改变 EI 排序（已接受） |
| 2 | ~~§2.5 可复现种子~~ **已做** | 实验方法 | 否（复用 MATLAB `fixRandSeed`） |
| 3 | ~~§2.4 bandit reward 归一化 + 单测~~ **已做** | 正确性 | 否（RL 开启时） |
| 4 | ~~§2.3 residual 标准化 + 阈值重标定~~ **已撤销**（恢复原始残差 + 阈值 10） | 数值 / 口径 | 已回到旧行为 |
| 5 | §3.2 每代最小标注量兜底 | 方法 | 会（但只影响「本该停摆」的代） |
| 6 | §3.1 适应度 LCB、§3.4 多样性、§3.5 数据卫生 | 方法 | 会 |
| 7 | q-EI / cEI（`ei_theoretical_basis.md` §6） | 方法 | 会 |

**验证顺序**：1–3 已实施、4 已撤销（见 §2.3）。现在可以用 `fixRandSeed` 固定随机流做配对 A/B，
把 §2 的改动与 §3 的方法改动（适应度 LCB、多样性、cEI）分离评估。

---

## 5. 复核入口

- `tests/` 当前 192 passed（`python -m pytest -q tests`，用 anaconda python 且从仓库根跑）。
- 全仓 `pytest` 会被 `dnmt/train/plot_gp_test.py`（未跟踪的数据目录）挡住收集，与代码无关。
- `README.md` §「GP 主动学习」、`docs/ei_theoretical_basis.md` §6 是已有的理论基线，改动前先对齐口径。

---

## 6. 性能补记：`newMolCheck` 的 Cython 内核

> 起因：permutation 算子被认为慢。实测 `Permutation_310.py` 自身的交换 / 平移
> 循环 < 0.05 ms，而一次完整通过的 `newMolCheck` 为 **125 ms**（8 分子、
> ~30 原子/分子、TNT/CL20 量级）；50 次重试上限约 7 s。`newMolCheck` 是
> **7 个变异算子共用**的，且 results2 每代 permutation 只产 2 个结构、
> rotmutation 产 55 个，所以真正吃掉时间的是共享的距离检查，不是 permutation。

**做法**：`gpevo/FunctionFolder/sys/molcheck_cy.pyx` 用 typed memoryviews +
`nogil` 把分子间原子对（约 2.5 万对）与分子内 124 个周期镜像的扫描放进纯 C；
`setup.py` 增加第二个 `Extension`（与已有的 `random_cell_mol_cy` 同一套
`cythonize` 指令：`boundscheck=False` / `wraparound=False` / `cdivision=True` / `-O3`）。
`newMolCheck.py` 保留原纯 numpy 实现作为**参考 oracle 与未编译时的兜底**，
`tests/test_molcheck_cy.py` 用 29 个用例断言两者逐位一致（随机结构、被拒结构、
非正交晶格、半整数分数坐标、空分子、负阈值）。

**实测**（本仓库，anaconda python 3.12）：

| 情形 | 纯 numpy | Cython | 加速 |
|---|---|---|---|
| 完整通过（无提前 return） | 124.7 ms | 1.46 ms | **85.6×** |
| 提前拒绝 | 2.58 ms | 0.078 ms | 33× |
| 非正交晶格 | 114.7 ms | 1.27 ms | 90.4× |
| `newMolCheck` 端到端 | 125.1 ms | 1.50 ms | 83× |

**语义保持**（与 MATLAB 对拍相关，勿随意改）：

- 最小镜像用 `np.round` 的**半偶舍入**（`_rint` 复刻），不是 C `round` 的半远离零；
- 笛卡尔位移是 `dfrac @ lat`（按 `lat` 的列求和）；
- 距离用 `sqrt(x²+y²+z²)` 与阈值比较，等价于 `np.linalg.norm(v, axis=1) <= thr`；
- 首个违规立即返回 0（坏子代仍然便宜）；
- `Permutation_310` / `LatMutation_310` / `Mutation_310` / `Random_310` 仍用
  `valid_molecules(..., 0)`，而 heredity / rotation / swarm 用 `-0.2`、softmode
  用 `-0.1` —— 与 MATLAB `Permutation_310.m:81` 一致，**没有**顺手改阈值。

**部署**：`.so` 与生成的 `.c` 都在 `.gitignore` 里；`pip install -e .`（或
`python setup.py build_ext --inplace`）后生效。未编译时自动回退纯 numpy，
行为不变只是慢。`~/ct22` 这类从运行目录启动的计算用的是 site-packages 的
非 editable 副本，必须重装才会用到新内核。

### 6.1 `uspex_fast_core.pyx` 的评估结论：保持 dormant，不接入

`gpevo/FunctionFolder/sys/uspex_fast_core.pyx`（386 行）是 Octave 时代的指纹
加速内核，被 `pyproject.toml` 列为 package-data，但 `setup.py` 从未编译它。
`fingerprint_utils.py:17` 的 `try: import ... / except: _ufc = None` 与
`:317` / `:349` 的 `_cy_fingerprint` 因此是**永不执行的死代码**。

本轮评估过「接进 `setup.py`」：Cython 3.0.11 与 gcc 都能编译（需 numpy
头文件 + `NPY_NO_DEPRECATED_API` 宏；`setup.py` 的 `_ext()` 目前不传这些，
且 `numpy` 不在 `[build-system].requires`）。实测 228 原子体系快 7.4×、
8 分子中心快 115×，但 `_erf_approx` 与 `scipy.special.erf` 差约 1e-6，
会改变 OUTPUT.txt 的指纹 / `S_order` 数值与边界相似性判定。**结论：不接。**

本轮按「不接、只清理矛盾」处理，**没有删除任何代码或文件**：

- 修正 `uspex_fast_core.pyx` 头部那条不存在的 `setup_fast.py` 编译命令，
  改为明确的 DORMANT 状态说明与「若启用需要什么」；
- 修正 `setup.py` 开头 docstring（被删的是 uspexkit 那份，不是 sys/ 这份）；
- 修正 `fingerprint_utils.py` 头部与 `_cy_fingerprint` 的误导性描述；
- 修正 `docs/feature_ablation.md` 的「已按冗余删除」口径；
- 在 `docs/fingerprint_method.md` 补 §6，把 Octave 的 2,780× 与 gpevo 的
  纯 numpy 现状分开；`pyproject.toml` 给该 `.pyx` 加 dormant 注释。

若将来要真正移除死代码，候选是：`fingerprint_utils.py` 的 `_ufc` import /
`_cy_fingerprint` / 两处调用点，以及 `uspex_fast_core.pyx` 本体。这两项
**没做**，按默认保留原则留待明确指示。


---

## 7. group rotation 占比恒为 0 的根因：`STDMOL` 按类型索引被写成按位置索引

> 完整表格、修复代码与回归测试见姊妹篇 `docs/rotation_operator.md` §5。

现象：`ct22/results3` 每代汇总 `group rotation: 0 / N (0.0%)`。
`Rotation_310.py` 里给分子查模板曾用**胞内位置** `mi` 索引 `std_mols[mi]` /
`ref_bonds[mi]`，并以 `mi < len(std_mols)` 保护。但 `STDMOL` / `molRefBonds`
是**按分子类型**建的（长度 = 分子种类数）。该体系 `numMols=[2,2]`、
`MtypeLIST=[1,1,2,2]`，`STDMOL` 只有 2 项，而 CL20（唯一声明了
`flex_dihedral` 的类型）处在位置 2、3 → 被保护条件跳过，`_twist_dihedrals`
从未调用 → `group_rotated` 恒 False。

诊断要点：

- `flex_dihedral` 来自 MOL 文件第 7 列（`optFlags[i,2]`，见
  `AbinitCode/read_molecules.py`）。`ct22/MOL_2`（CL20）有 6 个原子非零，
  `MOL_1`（TNT）全 0 —— 所以「没声明基团」不是原因。
- `_twist_dihedrals` 对真实 CL20 模板几何 300/300 接受，扭转算法与
  `intramol_bonds(cand) == ref` 判据都没问题。
- `variation_common.recompute_zmatrix()` 早已按 `MtypeLIST[j]-1` 正确索引
  `STDMOL`，可作为对照。

修复：按 `MtypeLIST[mi]-1` 取模板与参考键集（与 MATLAB `STDMOL(k)` 一致），
`MtypeLIST` 缺失时回退按位置。真实弛豫几何验证：`ct22/results3/generation3`
的 8/8 个 CL20 分子扭转成功（模板键集与弛豫几何键集均为 39，一致）。
`tests/` 全量 192 passed。

教训：多拷贝共晶里「分子位置」与「分子类型」不是同一个编号；任何按数组下标
查 `STDMOL` / `molRefBonds` 的地方都要走 `MtypeLIST`。
