# 旋转算子（Rotation_310）默认行为说明

`Rotation_310` 对应 MATLAB 的 `Rotation_310.m`（分子刚体旋转算子），代码位于
`gpevo/FunctionFolder/USPEX/310/Rotation_310.py`。每个被选中分子会施加**两类旋转**，
并且默认还会做一次“呼吸晶格”（breathing lattice）。

---

## 1. 两类旋转

### 1.1 整体刚体旋转（principal-axis rigid rotation）

分子先减去质心，对三条主惯性轴分别做一次随机角度旋转，模拟 MATLAB 的
`SmartRotInertia`（`Rotation_310.py:128-137`）。

- 角度幅度由 `rotationAngleMax` 控制，默认 **`π/2`（90°）**
  （`Rotation_310.py:102`，`getattr(..., "rotationAngleMax", np.pi / 2)`）。
- 实际每轴角度 = `(rand-0.5) * 2 * angle_max * ref / b[loop]`，
  其中 `ref` 是最小非零主惯性矩、`b[loop]` 是当前轴的主惯性矩
  （`Rotation_310.py:136`）。即**最轻的轴可转满 `rotationAngleMax`，
  重轴按比例缩小**——这是 MATLAB 对拍修正后的行为。
- 之后再做一次小幅刚体平移：`±translationMax`，默认 **0.5**
  （`Rotation_310.py:102, 138`）。

### 1.2 基团旋转（flexible intramolecular dihedral twist）

在刚体旋转之后，若分子模板声明了柔性二面角，绕可旋转键把一侧取代基（基团）
转开，例如**硝基绕 C–N 键转动**（`Rotation_310.py:14-16, 140-148` 调用
`_twist_dihedrals`）。这是分子内部的“基团旋转”，功能等价于 MATLAB 的
Z-matrix 二面角编辑，但在笛卡尔空间完成以便复用 `Rotate_rigid_body`。

**默认旋转角度（`_twist_dihedrals`，`Rotation_310.py:71-77`）：**
不是固定一个角，而是**渐进式随机角**，一次尝试内按迭代阶段切换：

| 迭代阶段 | 步长 | 角度范围 |
|---|---|---|
| 前 1/3 迭代 | `π/2` | ±90° |
| 中 1/3 迭代 | `π/3` | ±60° |
| 后 1/3 迭代 | `π/6` | ±30° |

只有**当分子模板声明了柔性二面角**时才执行——即 `STDMOL[k].flex_dihedral`
非空。该字段由 `read_molecules.py` 根据 `optFlags[i,2]` 与分子 `format` 构建
（`AbinitCode/read_molecules.py:164` 附近）。若模板未声明，`flex` 为空，
基团旋转整段跳过，只做刚体旋转。**前提是模板按分子类型正确索引**——多拷贝共晶（如 `MtypeLIST=[1,1,2,2]`）下若用胞内位置查 `STDMOL`，真正带 flex 的分子会被跳过，见 §5。

每次扭转后都会检查分子内连通性（键集合是否保持不变），只有不破坏连接的扭转
才被接受；若所有尝试都失败，则保留仅做刚体旋转的结果
（`Rotation_310.py:79-95`）。

---

## 2. 呼吸晶格（breathing lattice）

`Rotation_310` 在旋转前默认对晶胞做一次 10–20% 的膨胀
（`breath_factor = 1.10 + rand()*0.10`，`Rotation_310.py:118-124`）：
只缩放晶胞与各分子质心，**不缩放分子内坐标**，以避免把每个分子内键拉长
10–20% 而被反应力场误判/打断。这是此前确认的 10–20% 范围。

---

## 3. 操作比例（fracRotMut）

旋转算子在每代子代中的占比由 `fracRotMut` 控制：

- 默认值 **`0.1`（约 10%）** —— `createORGDefault.py:117`
  （`ORG.fracRotMut = 0.1`）。
- 在 `createORG_EA.py:53-57` 中：**分子体系**取 INPUT 值或默认 `0.1`；
  **非分子体系**直接置 `0.0`（不启用旋转）。
- 该比例参与三种调度（静态 / AutoFrac / RL bandit）：
  `update_STUFF.py:22,33` 的 `FRACTIONS` 与 `RL_FRACTIONS` 都含 `fracRotMut`；
  RL 模式下 `fracRotMut` 是 bandit 的一个臂（`update_STUFF.py:150-161`）。

> 注意：**基团旋转没有独立的比例参数**。它只是 `Rotation_310` 内部的第二步，
> 随 `fracRotMut` 一起被选中后才执行；是否真正发生只取决于模板里有没有
> `flex_dihedral`。

---

## 4. 关键参数速查

| 参数 | 默认值 | 来源 | 含义 |
|---|---|---|---|
| `rotationAngleMax` | `π/2`（90°） | `Rotation_310.py:102` | 刚体旋转角上限（最轻轴） |
| `translationMax` | `0.5` | `Rotation_310.py:102` | 旋转后刚体平移上限 |
| 基团扭转角 | `±π/2 / ±π/3 / ±π/6` | `Rotation_310.py:71-77` | 渐进随机：±90°/±60°/±30° |
| 呼吸晶格 | `1.10–1.20` | `Rotation_310.py:118` | 晶胞膨胀 10–20% |
| `fracRotMut` | `0.1` | `createORGDefault.py:117` | 旋转算子占每代子代比例（分子体系） |

---

## 5. 模板按「分子类型」索引，不是胞内位置（group rotation 占比曾恒为 0 的根因）

`STDMOL` 与 `ORG.molRefBonds` 都是**按分子类型**建的：长度等于分子种类数
（`numMols` 的项数），第 `k` 项对应第 `k` 类分子模板。而 `child.MOLECULES`
是按**胞内位置**排列的，一个晶胞可以含同一类型的多个拷贝，例如
`MtypeLIST = [1, 1, 2, 2]`（2 个 TNT + 2 个 CL20）时，`STDMOL` 仍只有 2 项。

给某个分子找模板，必须用该分子的类型 `MtypeLIST[mi] - 1` 去索引（与 MATLAB
`STDMOL(k)` 一致，`variation_common.recompute_zmatrix()` 也是这么做的）：

```python
tidx = int(child.MtypeLIST[mi]) - 1
std  = std_mols[tidx]                 # flex_dihedral 在这里
rb   = ref_bonds[tidx]                # 参考键集也按类型
```

**历史 bug（本次修复前）**：这里曾写成 `std_mols[mi]` / `ref_bonds[mi]`
（用位置 `mi`），并用 `mi < len(std_mols)` 做保护。对 `[1,1,2,2]` 的晶胞：

| 位置 `mi` | 类型 | 旧代码取到的模板 | 结果 |
|---|---|---|---|
| 0 | TNT | `STDMOL[0]`（TNT, 无 flex） | 不扭转 |
| 1 | TNT | `STDMOL[1]`（CL20, 有 flex） | 错套模板，几何不匹配，扭转被拒 |
| 2 | CL20 | `mi<2` 为 False → 跳过 | **真正的 CL20，从未扭转** |
| 3 | CL20 | 同上，跳过 | 同上 |

于是在 `ct22/results3`（TNT+CL20 共晶）里，唯一声明了可旋转基团的 CL20
处在位置 2、3，永远进不了 `_twist_dihedrals`，`group_rotated` 恒为 False，
每代汇总的 `group rotation: 0 / N (0.0%)` 就是这个索引错误，而非扭转算法
或接受判据的问题（对 CL20 模板几何单独测试，扭转 300/300 成功）。

**修复**：按 `MtypeLIST` 类型索引（同时修好 `molRefBonds` 的同类错位）；
`MtypeLIST` 缺失时回退到按位置索引，兼容旧个体。回归测试见
`tests/test_rotation_310.py` 的
`test_group_rotation_reaches_molecules_past_the_template_count`；
真实弛豫几何验证：`ct22/results3` 里 8/8 个 CL20 分子扭转成功（50/50 每次）。

> 与 §1.2 对照：基团旋转是否发生，除了「模板声明了 `flex_dihedral`」这个前提，
> 还隐含「该分子的类型索引能正确取到那个模板」。多拷贝共晶是常态，不要用
> 胞内位置去查 `STDMOL`。

---

## 6. 打开基团旋转后的高失败率与算子级回退（2026-11-06）

### 现象

`ct22/results3` 打开 `flex_dihedral` 后，旋转算子大量打印
`failed to do Rotation, switch to Random`。用真实弛豫几何（`generation3`）
做受控对照，60 次 `Rotation_310` 调用：

| | 回退 Random | 带基团旋转的子代 |
|---|---|---|
| `flex_dihedral` 关闭 | 2/60（3%） | 0/60 |
| `flex_dihedral` 打开（修复前） | 43/60（**72%**） | 17/60 |
| `flex_dihedral` 打开（修复后） | **0/60（0%）** | 6/60 |

### 根因：两级判据不一致

`Rotation_310` 每个 attempt 先对所有分子做完「呼吸晶格 + 主轴刚体旋转 +
平移 + 基团扭转」，最后才做一次整胞 `valid_molecules(child, -0.2)`；
失败就丢弃整个子代，50 次全败则 `Random_310`。

而这两级检查用的是**不同的阈值**：

- `_twist_dihedrals` 的接受判据是 `intramol_bonds(cand) == ref`，即分子内
  **成键集合**不变，用的是共价半径阈值 `1.2*(r_i+r_j)`（C/N/O 约 1.64–1.82 Å）；
- 最终的 `valid_molecules` 走 `newMolCheck`，检查**分子间**最小距离
  `minDistMatrice - 0.2`（ct22 实测 1.55–1.70 Å）和分子内周期镜像。

一个转开的硝基完全可能「键集合没变」却插进相邻分子 → 扭转自认为成功，
整胞检查把它判死，于是整个子代被丢弃。

### 已有的回退和缺少的回退

- **分子级已有**：`_twist_dihedrals` 里某个 flex 在 `max_iter`（默认 200）
  内找不到合法角度时，`coords = original`，该分子只保留刚体旋转+平移
  （`Rotation_310.py` 的 `if not moved` 分支，与 MATLAB
  `SmartRotInertia` 的 `if ~goodRot` 对应）。
- **算子级没有**：一旦扭转成功但整胞 `newMolCheck` 失败，没有任何降级，
  直接 `continue` 重试；50 次全败 → `Random_310`（完全不同的随机结构）。

### 试过但不采用的做法

最初按「每个候选二面角立刻验分子间距离」实现（`clash_check`）：实测
**单个子代 40.9 秒**、碰撞拒绝率 65%，12 分钟跑不完 10 个子代。两个原因：
每个角度都要跑整胞 `newMolCheck` + 纯 Python `intramol_bonds`，200 角度
× 6 flex × 4 分子 × 50 attempt 直接爆炸；且该检查发生在分子循环**中途**，
其余分子尚未刚体旋转/平移，验的是半成品晶胞，语义本身就不对。该实现已回退。

### 采用的修复：子代定稿后验证 + 回退到刚体旋转

在 `Rotation_310.py` 的分子循环里，把「刚体旋转+平移后、扭转前」的
`(mol, MOLCOORS, MOLCENTER)` 存进 `rigid_geometry`；全部处理完后：

1. 先 `valid_molecules(child, -0.2)` 验一次；
2. 若失败且本轮 `group_rotated` 为真 → 把所有分子恢复到 `rigid_geometry`、
   `group_rotated = False`，再验一次；
3. 两次都失败才 `continue`（并最终回退 `Random_310`）。

每次 attempt 最多 2 次整胞检查（Cython `newMolCheck`，~1ms），查的是
**完整晶胞**；效果：回退 Random 72% → 0%，同时仍保留 6/60 的真实基团旋转
子代（那些恰好不撞的）。回退产生的子代是纯刚体旋转，不带
`[group rotation]` 标记，也不计入 `group rotation` 汇总计数——这是正确的，
它本来就没有基团旋转。

回归测试：`tests/test_rotation_310.py` 的
`test_rigid_rotation_fallback_when_twisted_cell_is_invalid`、
`test_no_fallback_when_twisted_cell_is_valid`、
`test_fallback_still_fails_retries_then_uses_random`。

> 与 MATLAB 对拍：MATLAB 的 `Rotation_310.m` 只有分子级
> `if ~goodRot` 回退，算子级仍是 50 次全败 → `Random_310`。这里新增的
> 第 2 次整胞检查是**gpevo 独有的行为改进**，会改变启用 `flex_dihedral`
> 时的子代来源（原本回退 Random 的那部分现在变成合法 rotmutation）。
> 关闭 `flex_dihedral` 时行为完全不变。

---

## 7. intramol_bonds 的向量化 / Cython 加速（2026-11-06）

### 热点

`sys/mol_bond_table.py` 的 `intramol_bonds` 原本是 O(n²) 双层 Python 循环，
每个原子对调用一次 `np.linalg.norm` 和两次 `_rcov()`（后者含 `G["ORG_STRUC"]`
查找 + `covalentRadius()` dict 查找）。36 原子分子单次 ~1.3–5 ms。真正的放大
点在 `Rotation_310._twist_dihedrals`：每个候选二面角都要重算一次键集合
（`intramol_bonds(cand, type_idxs)`，`Rotation_310.py:109`），而 `type_idxs`
在整个搜索里不变——半径被重复查询几十万次。

### 做法（Cython 主路径 + 向量化兜底）

- 新增 `sys/bondtable_cy.pyx`：typed memoryviews + `libc.math.sqrt` 的双层 C
  循环，返回 `(i,j)` 列表；接进 `setup.py`（第三个 ACTIVE 扩展，与
  `molcheck_cy` 同一套 `cythonize` 指令），生成的 `.c`/`.so` 加进
  `.gitignore`，`.pyx` 加进 `pyproject.toml` package-data。
- `mol_bond_table.py` 拆成三层：`_reference_bonds`（原始双循环，作 parity
  oracle）、`_vectorized_bonds`（numpy 广播，未编译时的兜底）、Cython 主路径；
  公共 `intramol_bonds(coords, type_idxs)` 签名与逐位语义不变。
- 新增 `radii_for(type_idxs)` / `bonds_from_radii(coords, radii)`，让
  `_twist_dihedrals` 在入口**预算一次半径数组**，后续所有候选角度复用；
  `_bond_graph` 也改为接收 radii。

### 实测（同轮可比，36 原子）

| 路径 | 耗时 | 加速 |
|---|---|---|
| 原始双循环 | 1.319 ms | 1× |
| 向量化（兜底） | 0.047 ms | 28× |
| `intramol_bonds` 公共入口（Cython） | 0.014 ms | 93× |
| `bonds_from_radii`（预算半径，扭转搜索实际用） | 0.004 ms | **347×** |

端到端（`ct22/results3` 真实弛豫几何，40 次 `Rotation_310`，固定种子）：
**mean 0.701 s → 0.069 s（10×）**，max 3.825 s → 0.185 s；上一节加的
算子级回退仍生效，回退 Random 0/40。

### 语义与对拍

- 判据仍是 `sqrt(dx²+dy²+dz²) < BOND_SCALE*(r_i+r_j)`，严格 `<`；
- `_rcov` 的越界 / `type_idx=0` 负索引回绕等既有语义全部保留（三条路径
  共用同一个 `radii_for`）；
- `tests/test_bondtable_cy.py` 用 30 组随机分子 + 20 组近阈值几何断言
  Cython / 向量化 / 原始双循环三者逐位一致，另覆盖空分子、非法 type、
  扩展缺失时的兜底。`tests/` 全量 251 passed。

> 部署：`~/ct22` 那类运行目录用的是 site-packages 非 editable 旧副本，
> 要吃到新内核仍需 `pip install -e .`；未编译 `.so` 时自动回退向量化版，
> 行为不变只是慢 `~6×`（相对 Cython 路径）。

---

## 8. 共用数值 kernel 的 Cython 化：NEW_coord2Zmatrix / Rotate_rigid_body（2026-11-06）

### 为什么不是编译 Rotation_310.py 本身

对真实 `ct22/results3` 数据做 cProfile（20 次 `Rotation_310`，累计 2.694 s）：

| 函数 | cumtime | 占比 |
|---|---|---|
| `_twist_dihedrals` | 0.909 s | 34% |
| `recompute_zmatrix` → `NEW_coord2Zmatrix` | 0.556 s | 21% |
| `Rotate_rigid_body`（6456 次） | 0.316 s | 12% |
| `copy_individual`（deepcopy） | 0.271 s | 10% |
| `valid_molecules` → `newMolCheck` | 0.254 s | 9%（已 Cython） |
| `PrincipleAxis` | 0.201 s | 7% |
| **`Rotation_310` 自身 tottime** | **0.103 s** | **4%** |

`Rotation_310.py` 自身只是调度逻辑，单独编成 Cython 几乎没收益。时间都在
它调用的**共用数值 kernel** 里。

### 做法

新增 `sys/rotation_kernels_cy.pyx`，两个 kernel（typed memoryviews + `nogil`
纯 C 循环，输出数组由 Python 侧分配，不需要 numpy 头文件）：

- `coord2zmatrix` 复刻 `NEW_coord2Zmatrix` 的逐原子标量循环：1-based
  `fmt` 索引钳到 atom 0、`ind==2` 的固定 `v3=(0,0,-1)`、acos 前的
  `[-1,1]` clip、`1e-12` 退化分母保护、`|tors+pi|<1e-4 → pi` 特例。
- `rotate` 复刻 Rodrigues 矩阵，注意是**行向量** `r @ M`（`@` 右侧矩阵，
  即列向量形式的转置）；零长度轴按 `n=1.0` 归一。

`NEW_coord2Zmatrix.py` / `Rotate_rigid_body.py` 保留原纯 numpy 实现作
`_python_coord2zmatrix` / `_python_rotate` 参考 oracle 与未编译时兜底；
公共入口签名不变，`fmt` 转成 `int32`、坐标转成 contiguous 再调 Cython。
`NEW_ZMATRIXCOORD` 等其余函数未动。接进 `setup.py`（第四个 ACTIVE 扩展）、
`.gitignore`、`pyproject.toml`。

两个函数都是跨算子共用：`NEW_coord2Zmatrix` 被 8 处调用
（`read_molecules.py`、`readMOL.py`、`GetOrientation.py`、
`Heredity_310.py`、`Mutation_310.py`、`SoftModeMutation_310.py`、
`variation_common.py` 等），`Rotate_rigid_body` 被
`GetOrientation.py` / `SwarmMutation_310.py` / `Rotation_310.py` 调用——
所以这轮收益**对所有 7 个算子都成立**，不止旋转。

### 实测（真实 CL20 36 原子几何）

| kernel | 纯 Python | Cython | 加速 | parity |
|---|---|---|---|---|
| `NEW_coord2Zmatrix` | 4.42 ms | 0.0085 ms | **523×** | 4.4e-16 |
| `Rotate_rigid_body` | 0.031 ms | 0.0053 ms | 5.8× | 1.8e-15 |

端到端（`ct22/results3`，40 次 `Rotation_310`，固定种子）：
**mean 0.069 → 0.044 s（再快 1.6×）**，max 0.185 → 0.131 s；回退 Random
仍 0/40。累计相对 §6 之前（0.701 s）快 **16×**。

`tests/test_rotation_kernels_cy.py` 的 55 个用例断言两条路径逐位一致，
覆盖随机分子、单/双原子、共线退化分母、`fmt` 含 0 钳位、零长度轴、
list / 非 contiguous 输入、刚体运动距离守恒、扩展缺失兜底。
`tests/` 全量 306 passed。

> 部署与前面几轮一致：`~/ct22` 那类运行目录用的是 site-packages 非 editable
> 旧副本，要吃到新 kernel 仍需 `pip install -e .`；未编译 `.so` 时自动
> 回退纯 numpy oracle，行为不变只是慢。

---

## 姊妹篇

- [`rl_bandit_tutorial.md`](rl_bandit_tutorial.md)：把 `rotationAngleMax` /
  `fracRotMut` 作为 RL 臂做单参数消融与 INPUT.txt 配置。
- [`algorithm_review.md`](algorithm_review.md) §7 / §8 / §9 / §10：全仓审查里
  记录的索引 bug（group rotation 占比恒 0）、72% 回退 Random 的修复，
  以及 `intramol_bonds`、共用数值 kernel 的两次 Cython 加速。
- [`strain_matrix.md`](strain_matrix.md)：旋转等算子完成后 `mol.ZMATRIX` 的
  刷新与 `Parents.strainMatrix` 的来历。
- [`swarm_operator.md`](swarm_operator.md)：另一个分子刚体算子（swarm）的原理，
  及其与 rotmutation 在作用单位 / 取向处理上的差别。
