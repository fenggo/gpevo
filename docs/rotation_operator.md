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
| 前 1/3 迭代 | `π` | ±90° |
| 中 1/3 迭代 | `π/3` | ±60° |
| 后 1/3 迭代 | `π/6` | ±30° |

只有**当分子模板声明了柔性二面角**时才执行——即 `STDMOL[k].flex_dihedral`
非空。该字段由 `read_molecules.py` 根据 `optFlags[i,2]` 与分子 `format` 构建
（`AbinitCode/read_molecules.py:164` 附近）。若模板未声明，`flex` 为空，
基团旋转整段跳过，只做刚体旋转。

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
| 基团扭转角 | `±π / ±π/3 / ±π/6` | `Rotation_310.py:71-77` | 渐进随机：±90°/±60°/±30° |
| 呼吸晶格 | `1.10–1.20` | `Rotation_310.py:118` | 晶胞膨胀 10–20% |
| `fracRotMut` | `0.1` | `createORGDefault.py:117` | 旋转算子占每代子代比例（分子体系） |

---

## 姊妹篇

- [`rl_bandit_tutorial.md`](rl_bandit_tutorial.md)：把 `rotationAngleMax` /
  `fracRotMut` 作为 RL 臂做单参数消融与 INPUT.txt 配置。
- [`strain_matrix.md`](strain_matrix.md)：旋转等算子完成后 `mol.ZMATRIX` 的
  刷新与 `Parents.strainMatrix` 的来历。
