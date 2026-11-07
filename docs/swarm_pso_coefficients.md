# swarm 算子的 PSO 系数：稳定性推导与推荐取值

> 姊妹篇：[`swarm_operator.md`](swarm_operator.md) 讲算子是什么、代码怎么落；
> 本文只回答「`swarmInertia` / `swarmCognitive` 凭什么取这个值、边界在哪」。
> 行号以提交 `db63d2f` 为基线，会随提交变陈旧，请按函数名 / 参数名检索。

---

## 0. 结论先行

推荐起点（保持 `swarmStep = 0.5` 不变）：

```
0.6   : swarmInertia
0.5   : swarmCognitive
```

- 保守（几乎不改变现有行为）：`0.3 / 0.25`
- 激进但仍稳定、不触发 `v_max` 截断：`0.7298 / 0.75`
- 不要把教科书经典的 `0.7298 / 1.49618` 直接抄进来：在本实现里它会让认知项
  顶到速度上限，PSO 动力学退化成定长随机步（§4）。
- 前提：`fracSwarm > 0`。这两个系数默认是 0，不写就完全不生效
  （`createORGDefault.py:112-113`）。
- 这组值是**理论推导的起点，不是实证结论**：仓库目前没有启用后的 A/B 数据，
  `docs/swarm_operator.md` §5.3 明确写了「有效性未经实证」。

---

## 1. 先把本实现的系数对上教科书 PSO

教科书（Shi & Eberhart 1998 的惯性权重形式）：

```
v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
x <- x + v
```

本实现 `SwarmMutation_310.py:216-218`：

```
v <- inertia*v_prev + cognitive*r1*d_p + (step*r2)*d_g
x <- x + v
```

其中 `d_p = min_image(pbest_frac @ L - x, L)`（`:212-213`），
`d_g = min_image(g, x, L, L_best)`（`:211`），`r1, r2 ~ U(0,1)`（`:214-215`）。

逐项对应：

| 教科书 | 本实现 | 取值 |
|---|---|---|
| w（惯性） | `swarmInertia` | 待定 |
| c1（认知） | `swarmCognitive` | 待定 |
| c2（社会） | **`swarmStep`** | 0.5（`createORGDefault.py:110`） |
| r1, r2 | `np.random.rand()` | U(0,1) |
| v_max 截断 | `0.5*norm(L, axis=1)`（`:221-222`） | 半根晶格矢（Å/代） |

**关键差异**：c2 不是自由参数，它恒等于 `swarmStep`。教科书的 c1 = c2 对称
配置，在这里意味着 `swarmCognitive ≈ swarmStep = 0.5`。这是整个取值推导的锚点。

另外取向也共用同一组系数：惯性 `slerp(I, q_v_prev, inertia)`（`:256-257`）、
认知 `t_p = cognitive*r1*...`（`:252-253`）、社会 `t_g`（`:241-242`）。所以下面
对 w / c1 的讨论同时约束质心与取向。

---

## 2. 稳定性：w 与 c1 的硬边界

### 2.1 化简

固定 pbest / gbest，把 r1, r2 吸进常数系数（最坏情形取 r1 = r2 = 1）：

```
phi1 = c1*r1,  phi2 = c2*r2,  phi = phi1 + phi2
pbar = (phi1*pbest + phi2*gbest)/phi
```

则

```
v_{t+1} = w*v_t + phi*(pbar - x_t)          (1)
x_{t+1} = x_t + v_{t+1}                     (2)
```

把 (2) 的 `v_t = x_t - x_{t-1}` 代入 (1) 消去 v：

```
x_{t+1} = x_t + w*(x_t - x_{t-1}) + phi*(pbar - x_t)
        = (1 + w - phi)*x_t - w*x_{t-1} + phi*pbar
```

令 `e_t = x_t - pbar`（pbar 是常数，平移不改变收敛性）：

```
e_{t+1} = (1 + w - phi)*e_t - w*e_{t-1}     (3)
```

### 2.2 特征方程与 Jury 判据

(3) 的特征方程：

```
lambda^2 - (1 + w - phi)*lambda + w = 0
```

写成 `lambda^2 + a1*lambda + a2`，即 `a1 = -(1+w-phi)`、`a2 = w`。
二阶首一多项式两根都在单位圆内（|lambda| < 1）的 Jury 条件是：

```
1 + a1 + a2 > 0
1 - a1 + a2 > 0
1 - a2    > 0
```

逐条代入（关键恒等式：`1 + w - phi` 展开后 w 与 -w 相消）：

```
1 - (1+w-phi) + w = phi                    > 0
1 + (1+w-phi) + w = 2 + 2w - phi          > 0   ->  phi < 2(1+w)
1 - w                                     > 0   ->  w < 1
```

再加 `a2 = w > 0`（否则速度每代反号），得到教科书 PSO 的**稳定三角形**：

```
0 < w < 1,    0 < phi < 2(1+w)
```

（Clerc & Kennedy 2002；Trelea 2003 对 (w, phi) 收敛域有系统分析。）

数值复核：取 w = 0.6，临界 phi = 3.2；伴随矩阵谱半径在 0.99*3.2 处
= 0.9051 < 1，在 1.01*3.2 处 = 1.0726 > 1，边界与推导一致。

**两处近似的说明**：(a) 上面把 r1, r2 冻结成常数，是标准的确定性 / 固定吸引子
分析；本实现每代重抽 U(0,1)，用最坏情形 r = 1 得到的上界是保守的（均值
phi_mean = (c1 + c2)/2 更小）。(b) `v_max` 截断（`:221-222`）本身就保证速度
有界，所以稳定三角形在这里不是「会不会炸」的安全约束，而是「比例动力学是否
还没被 clip 吃掉」的保真度约束。

### 2.3 换算到本实现

最坏情形 r1 = r2 = 1：

```
phi = swarmCognitive + swarmStep = c1 + 0.5
```

稳定条件 `c1 + 0.5 < 2(1+w)`：

| w | 允许的 c1 上界 |
|---|---|
| 0.3 | < 2.1 |
| 0.6 | < 2.7 |
| 0.7298 | < 2.96 |

**结论**：只要 w > 0，c1 在合理范围内（≤1）都远在稳定域内。所以 w / c1 不是
被稳定性卡住的，而是被探索-开发平衡和 `v_max` 截断卡住的；w 由此变成一个
纯粹的「动量记忆长度」旋钮。

---

## 3. 经典 0.7298 / 1.49618 是怎么来的（以及为何不能直接抄）

Clerc & Kennedy 2002 的收缩因子（constriction）：取 `phi = 4.1 > 4`（即
c1 = c2 = 2.05，故意让无收缩系统发散以换取快速收敛），再整体乘一个

```
chi = 2 / |2 - phi - sqrt(phi^2 - 4*phi)|
```

代入 phi = 4.1：`sqrt(16.81 - 16.4) = sqrt(0.41) = 0.6403`，
`|2 - 4.1 - 0.6403| = 2.7403`，`chi = 2/2.7403 = 0.72984`。

于是有效系数：

```
w_eff   = chi               = 0.7298
c1_eff  = c2_eff = chi*2.05 = 1.49618
phi_eff = chi*4.1           = 2.9924
```

检验稳定三角形：`2.9924 < 2(1+0.7298) = 3.4596`，成立 —— 收缩因子正是把发散
系统拉回稳定域内、并贴着「快速收敛」边界的那个点。

**为什么不能照抄到 gpevo**：这里的 c2 被 `swarmStep = 0.5` 钉死了，phi_eff
小得多；照抄 `c1 = 1.49618` 会让认知项的量级达到社会项的 3 倍（§4），算子
从「跟当代最优」变成「追自己的祖先」。

---

## 4. 量级核算：以 a=12 Å 的对角晶胞为例

`SwarmMutation_310.py:221` 的截断是 `v_max = 0.5*norm(L, axis=1)`，对
`L = diag([12,30,30])` 得 `v_max = [6, 15, 15]` Å。最小镜像折叠后
`|d_p|, |d_g| <= 0.5*|L_row| = 6` Å（x 方向）。取 r1 = r2 = 1（上界），
`swarmStep = 0.5`：

| c1 | 认知项上界 (c1*6 Å) | 社会项上界 (0.5*6 Å) | 是否顶到 v_max = 6 Å |
|---|---|---|---|
| 0.25 | 1.50 Å | 3.00 Å | 否 |
| **0.5** | **3.00 Å** | **3.00 Å** | **否** |
| 0.75 | 4.50 Å | 3.00 Å | 否 |
| 1.49618 | 8.98 Å | 3.00 Å | **是** |

c1 = 0.5 时认知与社会**等量**，合成速度约 3~4.5 Å，低于 6 Å 的截断，PSO 的
比例动力学完整保留；c1 = 1.49618 时单认知项就越过截断，`np.clip`（`:222`）
把速度削成定长，惯性 / 认知 / 社会的相对关系全丢。

这就是推荐 `c1 ≈ swarmStep` 的**量级理由**，与 §1 的对称性理由一致。

---

## 5. w 取多少：动量记忆长度

速度的齐次解 `v_t = w^t * v_0`，衰减时间常数

```
tau = -1/ln(w)   （单位：代）
```

| w | tau | 含义 |
|---|---|---|
| 0.3 | 0.83 | 几乎无记忆 |
| 0.5 | 1.44 | 短记忆 |
| **0.6** | **1.96** | 约 2 代 |
| 0.7298 | 3.17 | 约 3 代（经典值） |
| 0.9 | 9.49 | 长记忆 |

本实现的引导信号比标准 PSO 更脏：

- `g` 是**当代**最优 `ranking[0]`，不是 ever-best，且 `valid_ranking` 注释自述
  可能 stale（`variation_common.py:70-83`）。
- `pbest` 是谱系继承的祖先最优，晶胞可能已变（`pbest_frac` 按分数坐标存，
  `swarm_state.py:59,107`）；适应度用 AntiSeeds / 排序修正**之前**的原值刷新
  （`CalcFitness_310.py:101-106`，修正在 `EA_310.py:79-80` 才发生）。

信号越脏，动量越不该长。取 `tau ≈ 2` 代 → `w ≈ 0.6`；想更贴近经典收敛点可用
`0.7298`（tau ≈ 3.2）。`w > 0.8` 会把几代前的旧方向一直带下去，在 CSP 多漏斗
地形上不利。

---

## 6. 推荐值与调参步骤

```
# 起点（保持 swarmStep = 0.5）
0.6   : swarmInertia
0.5   : swarmCognitive
```

调参顺序：

1. 先只开 `swarmCognitive = 0.5`、`swarmInertia = 0`，确认「认知 + 社会」等量时
   最优适应度曲线不劣于纯社会项。
2. 再加 `swarmInertia = 0.6` 看是否加速收敛。
3. 多样性掉得太快就退到 `0.3 / 0.25`；想更强开发用 `0.7298 / 0.75`。
4. 硬约束：`0 < w < 1`，`c1 + 0.5 < 2(1+w)`；实用上还要保证
   `c1*0.5*|L_row| < v_max`（§4），否则截断会吃掉动力学。

**验收**：用 `fixRandSeed` 做配对随机流 A/B（见
`docs/swarm_operator.md` §6 第 6 项），对比 `0/0` 与推荐组的最优适应度曲线、
种群多样性指标。上面全部是推导值，别当成已验证的最优超参。

---

## 7. 参考文献

- Y. Shi, R. Eberhart, "A modified particle swarm optimizer",
  IEEE International Conference on Evolutionary Computation, 1998
  （引入惯性权重 w，常用 0.9 线性降到 0.4）。
- M. Clerc, J. Kennedy, "The particle swarm - explosion, stability, and
  convergence in a multidimensional complex space",
  IEEE Transactions on Evolutionary Computation, 6(1):58-73, 2002
  （收缩因子 chi、phi = 4.1、稳定域）。
- I. C. Trelea, "The particle swarm optimization algorithm: convergence
  analysis and parameter selection", Information Processing Letters,
  85(6):317-325, 2003（(w, phi) 平面收敛域的系统分析）。

> 本文只做参数口径推导；算子结构、晶胞对齐、MtypeLIST 断言、pbest 状态持久化
> 见 [`swarm_operator.md`](swarm_operator.md)。

