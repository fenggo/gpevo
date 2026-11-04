# 高斯过程 EI 选择的原理

> 本文整理自 gpevo 代码审查对话，说明 `gpevo/FunctionFolder/USPEX/src/gp_ml.py` 中
> Expected Improvement（期望改进）主动学习的原理、数学形式、代码实现，以及它与
> 贝叶斯优化的关系。文中 `Lxxx` 行号是**实施 §2.1/§2.2 之前**的，已陈旧，请按函数名检索。
>
> 理论依据与教科书口径的差异（EGO 出处、σ 校准前提、top-K vs q-EI、den 硬门 vs cEI）
> 见姊妹篇 `ei_theoretical_basis.md`；全仓审查与改进清单见 `docs/algorithm_review.md`。
>
> **实施记录（`docs/algorithm_review.md` §0.1）**：EI 现在用真正的后验 σ
> （`s.sigma_gp = uncertainty/1.96`），`u` 门仍用 1.96σ；`f_best` 在真实选择
> 与状态报告之间已统一为观测 DFT 最优；每代新增 incumbent/max(μ) 与 `y~μ` 校准诊断。

---

## 1. EI 要解决的问题

GP 主动学习里存在两个互相冲突的动机：

- **利用（exploitation）**：把 GP 预测最优秀的晶体送去 DFT，把好结果落袋。
- **探索（exploration）**：把 GP 最不确定的晶体送去 DFT，让模型变准，避免漏掉藏起来的好结构。

DFT 预算有限，每代只能算 K 个（`k` 参数）。EI 的作用是**给每个候选晶体算一个分数，
这个分数同时计入"均值有多好"和"不确定性有多大"**，然后选分数最高的 K 个。

---

## 2. 为什么 GP 能这么做

普通回归只给一个点预测。GP 给的是**一个分布**：对晶体 $x$，预测值是高斯
$N(\mu(x), \sigma^2(x))$。$\mu$ 是"最可能的值"，$\sigma$ 是"我有多没把握"。
这是 EI 能用数学精确定义的先决条件。

在 gpevo 中，`density_gp` / `energy_gp` 对应 $\mu$。`uncertainty` / `uncertainty_eng`
是 **$1.96\sigma$（95% 置信半宽）**，`gp_ml.py` 里的 `UNCERT_SCALE=1.96` 与 `_to_sigma()`
把它换算成后验 σ 存进 `s.sigma_gp` / `s.sigma_eng`，**EI 公式吃的是这个 σ**；
$1.96\sigma$ 继续用于 OUTPUT.txt 的 UC 列、误差棒和 `u` 门。§2.1 之前曾把 $1.96\sigma$
直接当 σ 喂给 EI，是本页公式与代码不一致的根源。

---

## 3. EI 的定义

EI = **期望改进**，字面意思：

$$
EI(x) = \mathbb{E}\left[\max(f(x) - f_{best},\, 0)\right]
$$

"如果我去算 $x$，它比当前最好的 $f_{best}$ 好多少？把这个'好多少'对 $f(x)$ 的
所有可能取值（高斯源）做平均。"

因为 $f(x) \sim N(\mu, \sigma^2)$，这个期望有**闭式解**：

$$
EI(x) = (\mu - f_{best})\,\Phi(Z) + \sigma\,\phi(Z),
\qquad Z = \frac{\mu - f_{best}}{\sigma}
$$

其中 $\Phi$ 是标准正态 CDF，$\phi$ 是 PDF。这就是 `gp_ml.py` 里
`_expected_improvement` 那一行（L463-471）：

```python
def _expected_improvement(mu, sigma, f_best):
    diff = mu - f_best
    if sigma < 1e-10:
        return max(0.0, diff)          # σ≈0 → 退化为纯贪婪
    z = diff / sigma
    phi_z = 0.5 * math.erfc(-z / math.sqrt(2.0))   # Φ(Z)
    pdf_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)  # φ(Z)
    return diff * phi_z + sigma * pdf_z
```

---

## 4. 两个分项的物理解释

| 项 | 含义 | 当它主导时 |
|---|---|---|
| $(\mu - f_{best})\,\Phi(Z)$ | **利用项**：均值高出 $f_{best}$ 的期望增益 | 已收敛，只信均值 |
| $\sigma\,\phi(Z)$ | **探索项**：不确定度带来的额外机会 | 去没去过的地方 |

关键性质（这正是 EI"聪明"的地方）：

- **$\sigma \to 0$**：$\phi$ 项消失，$EI \to \max(\mu - f_{best}, 0)$，
  退化成纯贪婪——均值不好就 0 分。
- **$\mu = f_{best}$**：$EI = \sigma\,\phi(0) = \sigma/\sqrt{2\pi} \propto \sigma$，
  完全由不确定度驱动，此时 EI 就是"最值得探索的那个点"。
- 两者之间是连续过渡，**不需要人为调"探索 vs 利用"的权重**——这是 EI 比
  "$\mu + \lambda\sigma$"这类手调公式根本强的地方。

---

## 5. gpevo 中的实现映射

EI 选择由 `gp_ml.py` 里几个函数串起来：

### 5.1 核心公式 → `_expected_improvement`（L463）

见上文第 3 节，即 EI 闭式解。

### 5.2 选哪个量 → `_ei_target`（L474）

```python
if optType == 1:  return _ENG_GP, _UNCERT_ENG, "eng_gp"   # 能量列 (7,8)
return _DEN_GP, _UNCERT, "den_gp"                          # 密度列 (4,5)
```

- `optType = 1`（焓最小化）→ 在**能量**维度上算 EI
- 其它（密度最大化，`optType = 11`）→ 在**密度**维度上算

列索引常量化在 L309-310：`_DEN_GP=4, _UNCERT=5, _ENG_GP=7, _UNCERT_ENG=8`。

### 5.3 方向翻正 → `sign`（L564-565）

EI 公式本质是"最大化"公式。能量要**最小**，所以对 $-E$ 求最大：

```python
opt_sign = params["optSign"]              # 1=最小化, -1=最大化
sign = -opt_sign if target == "eng_gp" else opt_sign
```

于是 EI 计算传入 `sign * mu`（L612），能量维度变成 $-\hat{E}$。

### 5.4 参照点 f_best → `_observed_incumbent`（L498）+ Step 0（L567-589）

这是 EI 唯一需要"外来信息"的地方——$f_{best}$（incumbent，当前最优）是什么。

- **优先**：从 `<data_dir>/feature.csv` 读 DFT 实测标签。`etot` 取 `min`、
  `density` 取 `max`（L540），再乘 `sign` 翻到 EI 空间。
- **回退**：文件缺失时（`observed is None`），`f_best = max(sign * mu)` 用 GP
  自己的预测作参照（L585-589）。

**为什么这一步关键**：$f_{best}$ 错了，所有 $Z$ 都整体偏移，EI 会系统性偏向
利用或探索。经典 EGO 的收敛性前提就是 **incumbent 必须是真实观测到的最优值**，
不能用模型自己猜的值。这也是把"GP 预测值当 f_best"改为"feature.csv 实测值"的
理论动机。

### 5.5 完整流程 → `_select_by_ei`（L544）

1. **定维度 + fallback**（L553-559）：`optType=1` 但拿不到能量预测时，退回密度维度。
2. **算 f_best**（L567-589）：观测优先，预测兜底。
3. **den 门**（L599）：本代 GP 预测最高密度 `best_density < den` → 整代跳过。
4. **逐晶体算 EI**（L604-615）：`residual > RESIDUAL_CUT`（=5.0，标准化空间）的跳过（外推区 μ/σ 不可信），
   其余 `ei = EI(sign*mu, sigma, f_best)`。
5. **排序选 top-K**（L619-638）：`mode=ei` 按 EI 降序、`uncertainty` 按 σ 降序、
   `random` 打乱。
6. **两道后置门**（L639-648）：`1.96σ > u`（`u` 是 95% 半宽的下界，太确定的不花 DFT）、
   `density_gp ≥ den`（不够密的不算）。EI 本身用 σ，但 `u` 门刻意保留 1.96σ 标度，
   以免把既有 INPUT.txt 的 `u` 值等效收紧 1.96 倍。

---

## 6. EI 属于贝叶斯算法吗？

**属于。** EI 是贝叶斯优化（Bayesian Optimization, BO）框架里的标准**采集函数**
（acquisition function）之一。

贝叶斯优化 = 三个部件循环：

```
循环：
  1. [贝叶斯] 用 GP 拟合观测数据 → 得到后验 μ(x), σ(x)
  2. [采集]   采集函数用后验算分数，选出下一个点 x* = argmax EI(x)
  3. [评价]   真实算 f(x*)（DFT），加进数据
```

| BO 部件 | gpevo 代码 |
|---|---|
| GP 后验 | uspexkit 的 `gpr_energy` / `gpr_density` 训练 → 产出 `energy_gp/uncertainty_eng`、`density_gp/uncertainty` |
| 采集函数 | `_expected_improvement` + `_select_by_ei` |
| 真实评价 | `uspexkit calc`（送 DFT）→ 追加进 `feature.csv` |

**两点边界（别把话说过头）**：

1. **EI 只是"一个"采集函数**，不是唯一。同族的还有 UCB、PI、Thompson Sampling，
   以及 gpevo 里的 `mode=uncertainty`（≈不确定性采样）、`mode=random`。它们都是
   基于后验的选点策略，区别只在于怎么用后验。

2. **gpevo 整体不是纯贝叶斯优化器**。外面的遗传算法循环（heredity/变异/选择）是
   进化算法，EI 只嵌在"每一代该送哪几个结构做昂贵 DFT"这一步。准确表述是：

   > **gpevo = 进化算法（主导） + 贝叶斯优化（EI 主动学习，内嵌在标注选择里）**

   EI 那一段是贝叶斯；整个 USPEX-GP 的定性是进化算法，不是 BO。

---

## 7. 关键参数速查

| 参数 | INPUT.txt 写法 | 含义 | 标度注意 |
|---|---|---|---|
| `k` | `2 : k` | 每代 EI top-K 个数 | — |
| `mode` | `ei : mode` | `ei` \| `uncertainty` \| `random` | — |
| `u` | `0.04 : u` | 不确定度门（**1.96σ 下界**，超过才送 DFT） | **跟随 EI 目标维度**：能量维是 eV 量级（data22 实测 σ 中位数 ~0.59 eV），密度维是 g/cm³ 量级（~0.04）。注意这是 95% 半宽的标度，不是 EI 里的 σ；换算关系是 `σ > u/1.96` |
| `den` | `1.8 : den` | 密度门（`density_gp ≥ den` 才送 DFT） | 永远盯预测密度，与 optType 无关 |
| `optType` | `1 : optType` | 1=焓最小化（能量 EI），11=密度最大化（密度 EI） | `-N` 写法=对应最大化/最小化翻转，经 `opt_sign` 进入 `sign` |

---

## 8. 一句话总结

gpevo 的 EI 选择 = **对每个 `residual ≤ 10` 的晶体，在它的优化目标维度上（能量或
密度，方向已由 `sign` 翻正），用 `feature.csv` 里已 DFT 实测的最优值作 incumbent，
按高斯闭式解算一个"期望改进"分数，取 top-K，再过 `u` 门和 `den` 门**。这是标准
EGO 采集函数；叠加的三个工程约束（residual 外推拒绝、u 不确定度门、den 密度门）
是套在 EI 之上的物理筛选，不属于 EI 本身。