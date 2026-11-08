# 是否把结构指纹并入 GP 的 10 维特征？

**结论：在这份真实 DFT 数据上没有帮助，不建议做。**

配套脚本：[`feature_ablation.py`](feature_ablation.py)（可复现）。

---

## 1. 问题

GP 的输入是 `uspexkit` 造的 10 维特征（`feature_mlp.csv`），输出 DFT 真值（`feature.csv`）——一个典型的 **Δ-learning** 设定：

| 列 | 含义 |
|---|---|
| `etot` | 廉价方法（ReaxFF）总能量 |
| `ebond / eang / etor / evdw / ecoul` | 能量分解项 |
| `ehb_cho / ehb_chn / ehb_chc` | 三类氢键项 |
| `density` | **密度代理**（廉价方法弛豫后的密度） |

每个弛豫完的个体**已经**带着 RDF 指纹（`FINGERPRINT`，见 `fingerprint_utils.py`），所以很自然会想把它拼到这 10 维后面。本实验就是量这件事。

> 注：这里说的指纹来自 gpevo 自己的 `FunctionFolder/sys/fingerprint_utils.py`（纯 numpy）。`uspexkit` 里那份同名的 `uspex_fast_core` 已按冗余删除；gpevo 自己也留了一份 `gpevo/FunctionFolder/sys/uspex_fast_core.pyx`，但**故意没有接进 `setup.py`**，因此从未编译、从未生效，实际计算一直走 `fingerprint_utils.py`。详见 `docs/fingerprint_method.md` §6 的状态说明。

## 2. 实验设计

- **数据**：`~/tnt22/data22`，216 个样本（TNT，固定组分），含 `feature_mlp.csv` / `feature.csv` / `structures.traj`。
- **模型**：**照抄生产核** `gpevo.uspexkit.core.load_gaussian_process`（`gpevo/uspexkit/core.py`）——ARD 长度尺度 + `n_restarts_optimizer=10`。用别的核会让比较失去意义（见第 5 节）。
- **降维**：指纹 216×4000（4 元素 → 16 个元素对 × 250 分箱），PCA 降到 3/5/8 维；**PCA 只在训练折内拟合**，不泄漏。
- **评估**：5 折 CV，看 MAE 与 R²。
- **对照**：把指纹**行随机打乱**再跑。若"提升"在打乱后依然存在，那它只是"多加了列"的数值假象。

```bash
python docs/feature_ablation.py --data ~/tnt22/data22            # 与生产一致（约 5 分钟）
python docs/feature_ablation.py --data ~/tnt22/data22 --restarts 0   # 快速版
```

## 3. 结果（R²）

| 输入 | density | energy |
|---|---|---|
| **基线 10 维** | **0.650**（MAE 0.0200） | **0.602**（MAE 0.2542） |
| + 指纹 PCA-3 | 0.637 | 0.591 |
| + 指纹 PCA-5 | 0.581 | 0.593 |
| + 指纹 PCA-8 | 0.651 | 0.550 |
| + **打乱的**指纹 PCA-5（对照） | 0.602 | 0.617 |

**没有任何提升**：density 持平或变差；energy 上**打乱指纹（0.617）反而比真指纹（0.593）更好**。对照直接说明这不是信号——指纹没有提供超出 10 维特征的增量信息。

## 4. 为什么没用

关键在于 10 维里已经有一个**密度代理**，它与 DFT 密度真值的相关性 **r = 0.707**：

```
10 维各列与 DFT 密度的 |r|:
[0.52 0.33 0.35 0.09 0.35 0.40 0.28 0.02 0.06 0.71]
                                              ^^^^ density 代理
```

密度本身就是"堆积几何的积分"，而 RDF 指纹编码的也主要是堆积几何——**两者信息高度冗余**。基线已能到 R² 0.65 / 0.60，剩下的残差更可能是 DFT 电子结构层面的东西，几何描述符都抓不到。

## 5. 一个差点被当成结论的假象（务必读完）

第一版实验跑出了非常唬人的数字：**R² 0.04 → 0.68**，看起来"指纹大幅提升精度"。它是**假的**，被两个证据戳穿：

1. **那版 baseline GP 本身是坏的**：用了各向同性（isotropic）长度尺度，超参优化后核塌成
   ```
   0.953**2 * Matern(length_scale=1e-05, nu=2.5) + WhiteKernel(noise_level=0.0915)
   ```
   长度尺度撞到下限、噪声吃掉全部信号 → 预测≈均值。同一份数据上线性回归就有 R²=0.55、单变量 GP 0.54——GP 不可能比线性还差，说明设置错了。
2. **打乱对照照样给 R² 0.61**（真指纹 0.68），说明"提升"只是多加了几个标准化列带来的副作用。

**教训**：
- 多维 GP 一定要用 **ARD**（逐维长度尺度）。生产代码 `gpevo/uspexkit/core.py` 的 `load_gaussian_process` 写的是 `length_scale = [0.1] * n`（列表），所以生产没问题；但任何自己搭的 GP 基线都要检查这一点。
- **任何"特征有效"的结论都必须配打乱对照**，否则无法区分"信号"和"多加了几列"。

## 6. 结论边界

只测了：TNT 一套体系、固定组分、216 样本；指纹参数 `rmax=10 / δ=0.04 / σ=0.015`（gpevo 默认）；PCA 一种降维方式。

以下情况结论**未必**成立：换体系（尤其同分异构竞争更激烈时）、换用 `atom_fing` 或 SOAP 而非 RDF、或显著不同的 `rmax`。但基于现有对照证据，不建议先在这条路上投入。

## 7. 那精度该往哪儿提

- **数据量是当前最大杠杆**。基线在这份 216 样本数据上 R² 0.65；而 tnt 那次运行的 `data_exp` 只有 **2 行**样本——那才是它 GP 完全不工作的原因，与特征设计无关。
- 想再往上走，更值得试的是**廉价方法侧的更好特征**（换 ReaxFF 参数化、加入力/应力信息），而非继续堆几何指纹。
