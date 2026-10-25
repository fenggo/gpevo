# USPEX 310 进化算法 + 高斯过程机器学习 (GPML) 主动学习

## 算法机理流程图

```mermaid
flowchart TD
    A["🚀 USPEX 启动<br/>初始化种群 (第1代)"] --> B["计算 Fitness<br/>optType=11: density-based"]
    B --> C{"generation > 1 ?"}
    C -->|"否 (第1代)"| Z["标准 EA 流程<br/>选择 → 变异/交叉 → 下一代"]
    C -->|"是 (第2代起)"| D["🔍 UQ 主动学习阶段"]

    subgraph UQ["UQ-based Active Learning (GPML)"]
        D --> E["读取 CalcFold1/gp.csv<br/>预训练的 GP 模型"]
        E --> F["扫描所有 valid 晶体<br/>fitness < 100000"]
        F --> G["遍历每个 valid 晶体<br/>从 gp.csv 查找其 GP 预测"]

        G --> H{"残差 (residual)<br/> > 10 ?"}
        H -->|"是"| I["⛔ 跳过 (GP 预测不可靠)"]
        H -->|"否"| J["计算 UCB 分数<br/>UCB = density_gp + 2.0 × σ"]

        J --> K["记录最高 UCB 的晶体<br/>(best_ucb_crystal)"]

        K --> L{"最高 UCB 晶体的<br/>σ > threshold ?"}
        L -->|"是"| M["📐 Step 1: uspexkit traj<br/>生成轨迹文件"]
        M --> N["⚛️ Step 2: uspexkit calc<br/>触发 DFT 计算"]
        L -->|"否"| O["跳过 DFT<br/>(GP 置信度足够)"]

        N --> P["📊 Step 3: uspexkit pred<br/>GP 预测所有 valid 晶体"]
        O --> P

        P --> Q["解析 density_predict.log<br/>读取 density_gp"]
        Q --> R["更新所有 valid 晶体 fitness<br/>fitness = -density_gp"]
    end

    R --> Z

    Z --> S{"满足收敛条件?"}
    S -->|"否"| B
    S -->|"是"| T["🏁 输出最优结构"]

    style A fill:#4a90d9,color:#fff
    style T fill:#2ecc71,color:#fff
    style UQ fill:#f8f9fa,stroke:#6c757d
    style M fill:#e74c3c,color:#fff
    style N fill:#e74c3c,color:#fff
    style P fill:#f39c12,color:#fff
    style I fill:#ecf0f1,color:#7f8c8d
```

## 核心公式

### UCB (Upper Confidence Bound) 采集函数

$$UCB = \hat{\mu}(x) + \kappa \cdot \hat{\sigma}(x)$$

其中：
- $\hat{\mu}(x)$ = GP 预测的密度 (`density_gp`，gp.csv col6)
- $\hat{\sigma}(x)$ = GP 预测的不确定性 (`uncertainty`，gp.csv col7)
- $\kappa = 2.0$ (exploration-exploitation 平衡系数)

### 关键决策逻辑

```
if UCB_crystal.σ > u_threshold:
    → 触发 DFT 计算 (探索高不确定性区域)
else:
    → 跳过 DFT (GP 已有足够置信度)
```

### 残差过滤

```
if residual > 10:
    → 该晶体不参与 UCB 竞争 (GP 预测质量太差)
```

## 数据流

```mermaid
flowchart LR
    subgraph 输入
        A1["INPUT.txt<br/>uspexkit gp --n=24 --data=... --u=0.04"]
        A2["CalcFold1/gp.csv<br/>预训练 GP 模型"]
        A3["results1/Individuals<br/>晶体结构列表"]
        A4["results1/gatheredPOSCARS<br/>POSCAR 集合"]
    end

    subgraph 处理
        B1["UCB 扫描<br/>选最高 UCB 晶体"]
        B2["uspexkit traj<br/>生成 Individuals.traj"]
        B3["uspexkit calc<br/>提交 DFT 作业"]
        B4["uspexkit pred<br/>GP 预测密度"]
    end

    subgraph 输出
        C1["fitness = -density_gp<br/>覆盖原始密度 fitness"]
        C2["density_predict.log<br/>预测结果"]
        C3["DFT 计算结果<br/>回填到 gp.csv"]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B2
    A4 --> B2
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> C1
    B4 --> C2
    B3 --> C3
```

## 索引体系

| 索引 | 范围 | 存储位置 | 用途 |
|------|------|---------|------|
| `bodyCount` | 全局累加 (如 268, 271...) | `POP_STRUC.POPULATION(i).Number` | gp.csv col2, calc/pred --ids, density_predict.log col1 |
| `valid_idx` | 当前代 1..N | `find(fitness < 100000)` | POP_STRUC 本地索引 |
| `Individuals` 行号 | 跨代累加 (1-based) | Individuals 文件 | 与 Individuals.traj 一致 |

## 参数说明

| 参数 | 来源 | 默认值 | 说明 |
|------|------|--------|------|
| `ncpu` | `--n=` | 8 | 并行计算核数 |
| `dat_dir` | `--data=` | `'data'` | GP 训练数据目录 |
| `u_threshold` | `--u=` | 0.04 | 不确定性阈值，超过则触发 DFT |
| `kappa` | 硬编码 | 2.0 | UCB 探索系数 |
| `residual_max` | 硬编码 | 10 | 残差过滤阈值 |
| `CalcFold` | 硬编码 | `CalcFold1` | GP 模型存储目录 |