# `strainMatrix` 是什么 —— 晶格应变与谱系字段

> 姊妹篇：[`rl_bandit_design.md`](rl_bandit_design.md)（算子配额怎么学）、
> [`molecule_damage_check.md`](molecule_damage_check.md)（结构合法性检查）。
> 本文回答一个具体问题：MATLAB USPEX 里 `Parents.strainMatrix` 干什么用，
> gpevo 不存它有没有影响；并记录 `mol.ZMATRIX` 一致性问题的处置结果。

## 1. 一句话结论

`strainMatrix` 是**晶格变异算子抽到的那次随机应变矩阵**，被塞进子代个体的
`Parents` 里当作谱系/审计元数据。它**不参与任何后续计算**：全树只有写入、
没有读取，也不进 `OUTPUT.txt` / `origin` 行。gpevo 不存它，对搜索行为和
输出文件都没有影响。

## 2. 它怎么算出来的

源头是 `FunctionFolder/USPEX/src/lattice_Mutation.m`（310/110 调用它）：

```
epsilons = randn(6,1) * mutationRate;     % 6 个独立正态应变分量
strainMatrix = zeros(3);
strainMatrix(1,1) = 1 + epsilons(1);      % 对角：三个轴向的拉伸/压缩
strainMatrix(2,2) = 1 + epsilons(2);
strainMatrix(3,3) = 1 + epsilons(3);
strainMatrix(1,2) = strainMatrix(2,1) = epsilons(4)/2;   % 非对角：剪切
strainMatrix(1,3) = strainMatrix(3,1) = epsilons(5)/2;
strainMatrix(2,3) = strainMatrix(3,2) = epsilons(6)/2;
new_Lattice = old_lattice * strainMatrix;
```

即一个对称的 3×3 应变矩阵 `S = I + E`：对角 `1+ε`（法应变），非对角
`ε/2`（工程剪应变的一半，对称放置）。右乘到晶格上得到新晶格。
`mutationRate` 是应变的标准差，MATLAB 默认 `0.25`；gpevo
`LatMutation_310.py` 里对应 `mutationRate`，默认 `0.08`（可在 INPUT.txt
覆盖），构造方式逐项一致。

### 注意：它不是最终生效的晶格变换

`lattice_Mutation.m` 在 `new_Lattice = old_lattice*strainMatrix` 之后还做了
两步，所以 `MUT_LAT ≠ old_lattice * strainMatrix`：

1. **体积恢复**：`ratio = det(old)/det(new)`，把晶格长度乘 `ratio^(1/3)`，
   让应变只改变形状、不改变密度（`lattice_Mutation.m:23-29`）。
2. **重包裹**：`LatMutation_310.m` 再调 `optLattice` 化简晶格，并把分子随
   重包裹位移整体平移。

所以 `strainMatrix` 记录的是**抽样时刻的原始应变**，不是父子晶格之间的实际
变换。想复现子代晶格，光有它不够。

## 3. 它存到哪、谁读它

`LatMutation_310.m` 成功后把三样东西写进 `info_parents`：

```
info_parents.parent       = 父代的 Number（字符串）
info_parents.enthalpy     = 父代弛豫焓（300/301 里还除以 sum(numIons)）
info_parents.strainMatrix = strainMatrix
OFF_STRUC.POPULATION(Ind_No).Parents = info_parents
```

`Parents` 与 `parent` / `enthalpy` 一起构成子代的**谱系记录**：谁生的、
父代多好、这次施加了什么变异。理论上 `strainMatrix` 可以用于事后分析
（哪种应变幅度更容易产出低焓结构）、复现某次变异、或在 heredity 里传递
应变偏好 —— 但 USPEX 里这些都没有实现。

实测证据（对 `/home/feng/uspex0925` 全树 grep `strainMatrix`）：

- **只有写入，没有读取**：出现的行全部是 `lattice_Mutation.m` 的定义/构造、
  各 `LatticeMutation_*.m` 的 `info_parents.strainMatrix = ...`、以及
  `Mutation_310.m` 里 `[MUT_LAT,strainMatrix] = lattice_Mutation(...)` 后
  把 `strainMatrix` 丢掉不用。没有任何 `xxx.Parents.strainMatrix` 被读取。
- **作者自己都禁用了**：`USPEX/300/LatticeMutation_300.m` 和
  `301/LatticeMutation_301.m` 里，`info_parents.strainMatrix=strainMatrix;`
  紧跟着一行 `info_parents.strainMatrix=0;` —— 存了又立刻清零。
- **不进输出文件**：`WriteIndividual.m` / `WriteOrigin` 只读
  `Parents.parent` 与 `Parents.enthalpy`（gpevo 的 `WriteOrigin.py:56-70`
  同样只取这两个字段）；`OUTPUT.txt` 的 `origin` 行只含父代 ID 与焓。

## 4. gpevo 现状与影响面

gpevo 的 `LatMutation_310.py` 里 `strain` 已经按同样公式算出来了（只是没
存）：

```python
eps = np.random.randn(6) * mutation_rate
strain = np.eye(3); ...
new_lat = lat @ strain
```

成功分支直接 `finalize_offspring(child, "LatMutate")`，没有写 `Parents`。
影响面分两层：

| 差异 | 影响 |
|---|---|
| 不存 `Parents.strainMatrix`（以及整个 `Parents`） | **无运行时影响**。零读者；不进 OUTPUT.txt / origin。只影响 `update_USPEX_INDIVIDUAL.py` 整体 deepcopy 进 `USPEX_STRUC.POPULATION` 的字典内容，即 `USPEX.pkl` 里少一个键 |
| `mol.ZMATRIX` | **已补齐**（见第 5 节）。此前不重算曾判为"纯簿记"，但真正实现 SoftModeMutation 后需修正：新加的 `Mutation_310.py` / `move_all_mol_mutation` 会读 `mol.ZMATRIX[0]` 作分子中心基准。补齐前不会出错，只因为这些读者的父代都来自 `POP.POPULATION`，是弛豫后经 `readMOL` 重建过的（ZMATRIX 新鲜）；安全性靠流程顺序保证，而非数据一致性，比较脆弱 |

## 5. 最终处置

- **`strainMatrix`**：**决定不补**。它是死字段，补了只让 `USPEX.pkl` 的字典
  更接近 MATLAB，不改变任何计算或输出。若将来做逐字段 pkl 对拍，可顺手补
  一行 `Parents.strainMatrix = strain`（`strain` 已在 `LatMutation_310.py`
  算出）。
- **`ZMATRIX`**：**已补，5 个算子一起补**。公共逻辑收敛到
  `variation_common.recompute_zmatrix(ind)`：按 `MtypeLIST` 取
  `STDMOL[mt-1].format`，逐分子执行
  `ZMATRIX = real(NEW_coord2Zmatrix(MOLCOORS, fmt))`；模板缺失时退回
  `MOLCOORS.copy()`，保证 `ZMATRIX[0]` 至少是最新的。

  调用点（都在成功分支、`finalize_offspring` 之前）：
  `Rotation_310` / `Permutation_310` / `Random_310` / `LatMutation_310` /
  `SwarmMutation_310`。另外三类算子本来就会写 ZMATRIX，无需改动：
  `Heredity_310`、`Mutation_310`（回退算子）、
  `SoftModeMutation_310`（经 `move_along_soft_mode`）。

  测试 `tests/test_zmatrix.py` 给每个父代塞全零的陈旧 ZMATRIX，跑完 5 个
  算子后断言后代的 ZMATRIX 与 `NEW_coord2Zmatrix(MOLCOORS)` 一致，并检查
  没有退化到 `Random_310` 回退分支（否则会掩盖漏调）。

## 6. 与 MATLAB 对拍时的定位

这个字段属于典型的「谱系元数据」而非算法状态。对拍时优先级应低于：
`optType` 覆盖、`CalcFitness` 数值、`newMolCheck` 阈值、软模/旋转的幅度与
轴向。它和 `Parents.mut_degree` / `mut_mode` / `mut_fre`（软模变异的谱系字段）
是同一类东西 —— 记下来供人看，程序不消费。
