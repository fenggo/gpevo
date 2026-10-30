# EI 采集函数的理论依据

> 本文整理自 gpevo 代码审查对话，回答的是「**为什么用 EI 来选晶体**」——即期望改进
> （Expected Improvement, EI）作为采集函数的理论来源、数学形式、成立前提，以及
> gpevo 实现相对教科书口径的偏离。
>
> 姊妹篇 `docs/expected_improvement.md` 讲的是「EI 是什么、在代码里怎么落的」；
> 本文讲的是「凭什么信它、它的边界在哪」。
>
> 行号以 `gpevo-codebuddy` 当前版本为准。

---

## 一句话

EI 是**贝叶斯优化 (Bayesian Optimization)** 的采集函数。在「DFT 很贵 + 高斯过程给出后验分布」的前提下，它把「期望能改进多少」当作效用，一步到位地把**探索**和**利用**折进一个标量，从而决定下一代把昂贵的 DFT 预算花在哪几个晶体上。

---

## 1. 出处与动机

- **Mockus（1970s）** 最早提出用期望改进做序列实验设计；系统化并使之流行的是
  **Jones, Schonlau & Welch (1998)** 的 EGO（*Efficient Global Optimization of
  Expensive Black-Box Functions*）。
- 关键场景：目标函数 $f$ 是**昂贵黑箱**（每算一次 DFT 成本很高）。GP 只给出**信念**
  $f(x)\sim\mathcal N(\mu(x),\sigma^2(x))$，它本身不告诉你下一步该算谁——因此需要
  采集函数把信念转换成决策。

> 严格地说，经典的 regret 界历史上主要由 GP-UCB / Thompson Sampling 系列给出；
> EI 的原始动机是 **myopic（一步前瞻）最优**，其一致性与 no-regret 性质是较晚近才补
> 上的理论结果（依赖 RKHS、核正则性等假设）。实践中 EI 的地位更多来自极强的经验表现。

---

## 2. 数学定义

最大化情形下，改进量定义为

$$
I(x)=\max\big(f(x)-f_{\text{best}},\,0\big)
$$

EI 即其期望。高斯后验下有**闭式解**：

$$
Z=\frac{\mu-f_{\text{best}}}{\sigma},\qquad
\mathrm{EI}=\underbrace{(\mu-f_{\text{best}})\,\Phi(Z)}_{\text{利用}}+\underbrace{\sigma\,\phi(Z)}_{\text{探索}}
$$

对应 `gp_ml.py:480-488`（`diff = mu - f_best`、`z = diff/sigma`、`phi_z` = $\Phi(Z)$、
返回 `diff*phi_z + sigma*pdf_z`）。

下面说明这个闭式解**从哪里来**。

### 2.1 闭式解的推导

闭式解不是谁的独门定理，它来自**高斯分布的一个积分恒等式**。

设 $f\sim\mathcal N(\mu,\sigma^2)$，密度 $\frac1\sigma\phi\!\left(\frac{t-\mu}{\sigma}\right)$：

$$
EI=\int_{f_{best}}^{\infty}(t-f_{best})\,\frac1\sigma\phi\!\left(\frac{t-\mu}{\sigma}\right)dt
$$

换元 $z=\frac{t-\mu}{\sigma},\ t=\mu+\sigma z$，下限变成 $z=\frac{f_{best}-\mu}{\sigma}=-Z$：

$$
EI=\int_{-Z}^{\infty}(\mu+\sigma z-f_{best})\phi(z)\,dz
=(\mu-f_{best})\underbrace{\int_{-Z}^{\infty}\phi(z)\,dz}_{(A)}
+\sigma\underbrace{\int_{-Z}^{\infty}z\phi(z)\,dz}_{(B)}
$$

(A)、(B) 都能闭式算出，靠的是标准正态密度 $\phi$ 的两个性质：

| | 用到的性质 | 结果 |
|---|---|---|
| (A) | 对称性 $\Phi(-Z)=1-\Phi(Z)$ | $1-\Phi(-Z)=\Phi(Z)$ |
| (B) | $\phi'(z)=-z\phi(z)\ \Rightarrow\ \int z\phi\,dz=-\phi$，且 $\phi(-Z)=\phi(Z)$ | $\big[-\phi(z)\big]_{-Z}^{\infty}=\phi(Z)$ |

代入即得盒式结论。

**关键点**：被积函数是「线性项 $(t-f_{best})$ × 高斯密度」，而 $-z\phi(z)$ 恰好是 $\phi$ 的
导数，所以「线性 × 高斯」的积分永远能闭式写出——这就是闭式解唯一的技巧来源。
$\max(\cdot,0)$ 把积分区间**截断**在 $f_{best}$ 以上，因此本质上求的是**截断高斯的一阶矩**：
一般分布只能数值积分，高斯恰好能给出解析式。

### 2.2 独立佐证：与 Black–Scholes 同构

金融中欧式看涨期权的价格是 $E[\max(S-K,\,0)]$。把标的换成高斯，
$S\sim\mathcal N(\mu,\sigma^2)$、行权价 $K=f_{best}$，得到的就是上面同一个式子，
**逐项同构**。这不是巧合——两边都在算「截断（对数）正态的尾部一阶矩」。这可以算作对该
闭式解最有力的独立佐证。

### 2.3 文献出处

| 阶段 | 来源 |
|---|---|
| 想法提出（expected improvement + 期望形式） | **Mockus, Tiešis & Žilinskas (1978)**，*The Application of Bayesian Methods for Seeking the Extremum*，收录于 *Towards Global Optimisation 2* |
| 闭式解 + 高斯过程语境下的完整表述（被引用最多的 canonical reference） | **Jones, Schonlau & Welch (1998)**，*Efficient Global Optimization of Expensive Black-Box Functions*，J. Global Optimization **13**(4), 455–492，即 EGO |

此后所有 EI 论文与教材（如 Shahriari et al. 2016 的 BO 综述）都沿用该式。

### 2.4 边界：什么时候没有闭式解

这个式子的**唯一假设是后验高斯**。一旦破掉就只能 MC 近似：

- **批量 q-EI**：q 个点的最大改进涉及多维高斯的最大值，一般无闭式解 → 蒙特卡洛。
  这也是 §6.1 说 top-K 只是近似的原因。
- **后验非高斯**（未知方差的 Student-t、非高斯似然）→ 闭式解失效。
- **$\sigma\to 0$**：$Z$ 发散、两项趋于不定。代码 `gp_ml.py:481-482` 据此直接退化成
  `max(0, mu - f_best)`（纯贪婪），正确处理了极限 $EI\to\max(\mu-f_{best},0)$。

严格讲，这个闭式解源于「截断高斯一阶矩」这一古典积分结果，并非 1998 年才有的新数学；
EGO 的贡献是把它放进 **surrogate + 序列设计**的框架，并给出 DACE/Kriging 模型下的完整形式。

---

## 3. 两项分别在做什么

| 项 | 含义 |
|---|---|
| $(\mu-f_{\text{best}})\,\Phi(Z)$ | **利用**：预测均值比 incumbent 好多少 × 真的会改进的概率 |
| $\sigma\,\phi(Z)$ | **探索**：后验越不确定越值得去问一次；$\phi(Z)$ 在 $\mu$ 接近 $f_{\text{best}}$ 时最大 |

它流行的主要原因是：**不需要手工调权衡系数**（对比 GP-UCB 要设 $\kappa$ 序列、
PI 要设 $\xi$），探索/利用自动平衡。

---

## 4. 成立的前提，也正是它的软肋

**EI 的探索项直接吃 $\sigma$，所以 $\sigma$ 必须被校准。** 若某类结构（分布外、
低质量）被 GP 判成宽后验，$\sigma\phi(Z)$ 会盖过改进项，把明显差的晶体推到第一名
去做昂贵的 DFT。

这正是 `explore` 开关存在的动因：`explore=0` 时 EI 退回 `diff*phi_z`，即纯利用
（等价于**加权 PI**；注意不是纯 PI = $\Phi(Z)$）。实用上有效，但理论代价是失去探索
保证、可能早熟收敛——见 `gp_ml.py:469-488` 的 docstring。

其余前提：

- 后验需近似高斯（GPR 天然满足，故可用 $\Phi$、$\phi$ 闭式）。
- 核函数要正确指定（密度用 ARD Matérn $\nu=5/2$、能量用 RBF）。EI 的质量完全依赖
  $\mu,\sigma$ 的校准；$\sigma$ 过宽或过窄都会让探索项失去意义。

---

## 5. 代码里哪些处理是有理论依据的

| 处理 | 位置 | 依据 |
|---|---|---|
| $f_{\text{best}}$ 用**观测**值（`feature.csv` 的 DFT 真值）而非 GP 预测 | `gp_ml.py:515-527`、`590` | EI 定义里 incumbent 必须是**已完成观测**的最优值。用预测值会让所有 $Z$ 整体偏移，系统性偏向利用（把模型噪声当真值） |
| optType 1 在 $-E_{gp}$ 上做最大化（`sign`） | `gp_ml.py:499-501`、`581-582` | $\min\mathrm{EI}(f)=\max\mathrm{EI}(-f)$，公式可统一 |
| `residual > 10` 先过滤 | `gp_ml.py:597-599`、`625` | 代理本身不可信的候选不进采集，类似 trust-region / 数据质量门 |

---

## 6. 偏离教科书的地方（不是 bug，但值得知道）

### 6.1 用 top-K 而非 q-EI

EI 是**单点、myopic** 的采集函数。gpevo 先逐点算 EI、再排序取前 K
（`gp_ml.py:637-657`）。严格的批量版本应当取 **q-EI**（q 个点联合改进的期望），
或 Kriging Believer / constant liar 近似。

直接取 top-K 的已知毛病：候选容易**聚集在同一区域**（都贴近当前最优，EI 值都很高且
彼此相关），导致批次多样性差、重复计算相似结构。

### 6.2 `den` 是硬门，而约束 EI 用乘性可行概率

教科书的约束 EI（Constrained EI, cEI）形式为

$$
\mathrm{cEI}(x)=\mathrm{EI}(x)\times\prod_j P\big(c_j(x)\ \text{可行}\big)
$$

gpevo 这里是 0/1 硬筛（`gp_ml.py:614-618` 整代门、`664` 逐候选门），因此：

- 丢掉了「稍差一点但可行概率高」的信息；
- 当 `den` 高于实际可达密度时 $P_{feas}\equiv 0$，**整代被判死**。ct22 的
  `den = 1.8 > 1.7876`（该运行可达的最大预测密度）正是这一情形。

### 6.3 `u` 门

`u`（`gp_ml.py:658`）是在 top-K **内部**要求 $\sigma>u$，含义是「不要把 DFT 花在模型
已经算得很准的点上」，属不确定度地板滤器，与「避免重复采样已知区域」的思路一致。

---

## 7. 结论

理论依据是充分的——在昂贵黑箱 + 校准良好的高斯代理下，EI 是最稳的默认选择。但要让它
真正生效，**$\sigma$ 的校准是关键**（这也是 `explore` 开关存在的根本原因）。

若要进一步贴近教科书口径，有两处可改：

1. 用 **q-EI** 或 Kriging Believer 替代 top-K 排序，提升批量多样性；
2. 把 **`den` 硬门改成乘性可行概率**（cEI），避免整代被判死。
