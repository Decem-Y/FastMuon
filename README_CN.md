# FastMuon ⚡

> **复现与基准测试**: 将 [Turbo-Muon](https://hal.science/hal-05390446) ([flash-newton-schulz](https://github.com/thib-s/flash-newton-schulz)) 应用于 [Muon 优化器](https://kellerjordan.github.io/posts/muon/) 的学习笔记。

[English README](README.md)

## 这是什么？

本仓库是对 Turbo-Muon 核心技术的**学习性复现**，将 [原始 Muon](https://kellerjordan.github.io/posts/muon/) 代码和 [Turbo-Muon 改进](https://github.com/thib-s/flash-newton-schulz) 整合到单文件优化器中，并提供独立的基准测试。

**这不是原创研究贡献。** 核心算法（AOL 预条件化、动态多项式系数、Triton 内核）全部来自 Boissin 等人的 Turbo-Muon 论文。如果需要官方实现，请使用 **[flash-newton-schulz](https://github.com/thib-s/flash-newton-schulz)**。

### 本仓库提供的内容

- **单文件实现** (`fastmuon.py`)：原始 Muon + Turbo-Muon 合并为一个文件，方便复制粘贴
- **基准测试脚本** (`benchmark.py`)：从零复现速度和质量对比
- **小幅工程修复**：1D 参数安全处理、放宽参数组校验
- **双语文档**：中英文解释 Turbo-Muon 的工作原理

### Turbo-Muon vs 原始 Muon（论文摘要）

| 特性 | 原始 Muon | Turbo-Muon |
|---|---|---|
| 归一化方式 | Frobenius 范数 | AOL 预条件化 |
| NS 多项式系数 | 固定 `(3.4445, -4.7750, 2.0315)` | 逐迭代动态系数 |
| NS 迭代次数 | 5 | 4（质量相当或更优） |
| Triton 加速 | ❌ | ✅（自动回退至 PyTorch） |

## 基准测试复现

我们独立运行了基准测试，验证 Turbo-Muon 论文中的结论。

**测试环境：** PyTorch 2.10.0+cu128, Triton 3.6.0, NVIDIA RTX 5090 D

### Newton-Schulz 速度对比

| 矩阵尺寸 | 原始 (5 次迭代) | FastMuon PyTorch (4 次) | FastMuon Triton (4 次) |
|---|---|---|---|
| 1×512×512 | 0.20 ms | 0.18 ms (1.14×) | 0.31 ms |
| 1×1024×1024 | 0.42 ms | 0.35 ms (1.20×) | 0.32 ms (1.34×) |
| 1×2048×2048 | 1.64 ms | 1.33 ms (1.23×) | 0.86 ms (1.91×) |
| 1×4096×4096 | 11.91 ms | 9.61 ms (1.24×) | **5.48 ms (2.17×)** |
| 8×1024×1024 | 1.65 ms | 1.37 ms (1.21×) | 0.98 ms (1.70×) |
| 16×1024×1024 | 3.45 ms | 2.84 ms (1.21×) | **1.89 ms (1.82×)** |
| 1×2048×512 | 0.32 ms | 0.28 ms (1.15×) | 0.27 ms (1.17×) |
| 1×4096×1024 | 0.94 ms | 0.77 ms (1.22×) | **0.45 ms (2.10×)** |

> Triton 内核在大矩阵（≥1024×1024）上表现优异。对于小矩阵，纯 PyTorch 实现开销更低。

### 正交化质量（Polar Error ↓ 越小越好）

| 矩阵尺寸 | 原始 (5 次, Frob) | FastMuon (4 次, AOL) | FastMuon (5 次, AOL) |
|---|---|---|---|
| 1×512×512 | 0.3225 | **0.1309** | **0.0726** |
| 1×1024×1024 | 0.3481 | **0.1412** | **0.0777** |
| 1×2048×2048 | 0.3621 | **0.1517** | **0.0826** |
| 1×4096×4096 | 0.4015 | **0.1637** | **0.0882** |
| 8×1024×1024 | 0.3479 | **0.1405** | **0.0781** |

> AOL 预条件化仅用 4 次迭代即可实现比原始 5 次迭代**低约 2.5 倍的 polar error**，验证了论文的结论。

### 优化器 Step 计时

| 配置 | 原始 (5 次迭代) | FastMuon (4 次迭代) | 加速比 |
|---|---|---|---|
| 8 参数 @ 512×512 | 1.38 ms | 2.09 ms | 0.66× |
| 8 参数 @ 1024×1024 | 3.35 ms | 2.07 ms | **1.62×** |
| 4 参数 @ 2048×2048 | 6.66 ms | 3.49 ms | **1.91×** |

> 对于小矩阵（512×512），Triton 内核启动开销超过收益。对于 ≥1024，FastMuon 提供稳定的加速，最高达 **1.91×**。

## 安装

```bash
# 克隆仓库
git clone https://github.com/Decem-Y/FastMuon.git
cd FastMuon

# （可选）安装 Triton 加速内核以获得额外加速
pip install kernels
```

## 快速上手

### 单卡使用

```python
from fastmuon import SingleDeviceMuon

# 收集隐藏层权重参数（2D 或 4D 卷积）
hidden_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]

optimizer = SingleDeviceMuon(hidden_params, variant="turbo", lr=0.02, momentum=0.95)
```

### 配合 Adam（推荐用于整个网络）

```python
from fastmuon import SingleDeviceMuonWithAuxAdam

hidden_matrix_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
other_params = [p for n, p in model.named_parameters() if p.ndim < 2 or "embed" in n]

param_groups = [
    dict(params=hidden_matrix_params, use_muon=True, variant="turbo",
         lr=0.02, momentum=0.95, weight_decay=0.0),
    dict(params=other_params, use_muon=False,
         lr=3e-4, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
]
optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
```

### 分布式使用

```python
from fastmuon import Muon, MuonWithAuxAdam

# 分布式训练的即插即用替换
optimizer = Muon(hidden_params, variant="turbo", lr=0.02)
# 或配合 Adam
optimizer = MuonWithAuxAdam(param_groups)
```

### 底层 Newton-Schulz 接口

```python
from fastmuon import newton_schulz

# 对梯度矩阵进行正交化（自动选择 Triton 或 PyTorch）
G = torch.randn(16, 1024, 1024, device="cuda")
X = newton_schulz(G, iter=4, precondition=True)
```

## 变体选项

| 变体 | 预条件化 | NS 迭代次数 | 速度 | 推荐用途 |
|---|---|---|---|---|
| `turbo`（默认） | AOL | 4 | 最快 | ✅ 通用场景 |
| `standard` | Frobenius | 5 | 基线 | 兼容旧实验 |

## Turbo-Muon 工作原理（来自论文）

以下是对 [Boissin 等人, 2025](https://hal.science/hal-05390446) 论文技术的摘要。所有功劳归于原作者。

### 1. AOL 预条件化（替代 Frobenius 归一化）

原始方法使用 Frobenius 范数归一化：
```python
X = X / (X.norm() + eps)  # Frobenius（原始）
```

FastMuon 使用 AOL（Absolute-value One-norm Layer）通过 Gram 矩阵进行逐行缩放：
```python
A = X @ X.mT                                               # Gram 矩阵
s = torch.rsqrt(clamp_min(A.abs().sum(dim=-1), min=eps))   # AOL 缩放因子
X = X * s.unsqueeze(-1)                                    # 逐行缩放
A = A * s.unsqueeze(-1) * s.unsqueeze(-2)                  # 复用 Gram 矩阵
```

由于第一次 NS 迭代本就需要计算 Gram 矩阵，AOL 预条件化**几乎零额外开销**，同时为收敛提供了更好的起始点。

### 2. 动态多项式系数

每次 NS 迭代使用针对性调优的系数，而非固定系数：
```python
ns_consts = [
    (4.0848, -6.8946, 2.9270),  # 第 1 次：更激进
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),  # 第 5 次：更保守
][-iter:]
```

### 3. Triton 融合算子

安装 `kernels` 库后，FastMuon 自动使用 Triton CUDA 内核融合矩阵运算，减少 GPU 显存带宽开销。

## 运行测试

```bash
cd FastMuon
python benchmark.py
```

## API 参考

### 类

| 类 | 说明 | 分布式 |
|---|---|---|
| `Muon` | 核心 Muon 优化器 | ✅ |
| `SingleDeviceMuon` | 单 GPU Muon | ❌ |
| `MuonWithAuxAdam` | Muon + 非 Muon 参数用 AdamW | ✅ |
| `SingleDeviceMuonWithAuxAdam` | 单 GPU Muon + AdamW | ❌ |

### 函数

| 函数 | 说明 |
|---|---|
| `newton_schulz(G, iter, precondition, epsilon, dtype)` | NS 正交化（自动选择 Triton/PyTorch） |
| `muon_update(grad, momentum, beta, ns_steps, nesterov, precondition, epsilon)` | 完整 Muon 更新步骤 |
| `zeropower_via_newtonschulz5(G, steps)` | 旧接口（Frobenius 归一化） |

## 小幅工程改进

在忠实复现之外，本仓库添加了一些小的工程修复：

- 1D 参数（bias、gain）安全跳过，而非崩溃
- 移除了严格的 `assert set(group.keys())` 检查，允许参数组中传入额外参数
- `zeropower_via_newtonschulz5()` 保留作为旧接口
- 默认 `variant="turbo"` 以便无缝升级

## 参考文献与致谢

本仓库的核心算法来自以下工作。**请引用它们，而非本仓库。**

- **Turbo-Muon 论文**（本仓库的主要来源）: Boissin, T., Massena, T., Mamalet, F., & Serrurier, M. [Turbo-Muon: Accelerating Orthogonality-Based Optimization with Pre-Conditioning](https://hal.science/hal-05390446), 2025
- **Flash Newton-Schulz**（Turbo-Muon 官方实现）: [thib-s/flash-newton-schulz](https://github.com/thib-s/flash-newton-schulz)
- **Muon 优化器**: Jordan, K. [MomentUm Orthogonalized by Newton-schulz](https://kellerjordan.github.io/posts/muon/)
- **AOL 缩放**: Prach, B. & Lampert, C.H. [Almost-Orthogonal Layers for Efficient General-Purpose Lipschitz Networks](https://arxiv.org/abs/2208.03160), ECCV 2022
- **Triton 内核**: 源自 [microsoft/dion](https://github.com/microsoft/dion) 的 Newton-Schulz Triton 实现
- **批量 Muon**: @scottjmaddox 实现，@YouJiacheng 实践
- **五次多项式策略**: 由 @jxbz、@leloykun、@YouJiacheng 提出
- **kernels 库**: [kernels](https://pypi.org/project/kernels/) — Triton 内核加载库

## 引用

请引用原始 Turbo-Muon 论文，**而非本仓库**：

```bibtex
@unpublished{boissin:hal-05390446,
  TITLE = {{Turbo-Muon: Accelerating Orthogonality-Based Optimization with Pre-Conditioning}},
  AUTHOR = {Boissin, Thibaut and Massena, Thomas and Mamalet, Franck and Serrurier, Mathieu},
  URL = {https://hal.science/hal-05390446},
  NOTE = {working paper or preprint},
  YEAR = {2025},
  MONTH = Dec,
  KEYWORDS = {Muon optimizer ; Newton-Schulz ; Nano-GPT ; Orthogonal Matrix},
  PDF = {https://hal.science/hal-05390446v1/file/main.pdf},
  HAL_ID = {hal-05390446},
  HAL_VERSION = {v1},
}
```

## 许可证

MIT License
