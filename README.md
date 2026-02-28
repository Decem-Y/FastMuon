# FastMuon ⚡

[![GitHub visitors](https://komarev.com/ghpvc/?username=Decem-Y&repo=FastMuon&label=visitors&color=blue)](https://github.com/Decem-Y/FastMuon)
[![GitHub stars](https://img.shields.io/github/stars/Decem-Y/FastMuon?style=flat)](https://github.com/Decem-Y/FastMuon/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A reproduction and benchmark study** of [Turbo-Muon](https://hal.science/hal-05390446) ([flash-newton-schulz](https://github.com/thib-s/flash-newton-schulz)) applied to the [Muon optimizer](https://kellerjordan.github.io/posts/muon/).

[中文版 README](README_CN.md)

## What is this?

This repository is a **learning-oriented reimplementation** of Turbo-Muon's key techniques, consolidating the [original Muon](https://kellerjordan.github.io/posts/muon/) code and the [Turbo-Muon improvements](https://github.com/thib-s/flash-newton-schulz) into a single-file drop-in optimizer with independent benchmarks.

**This is NOT an original research contribution.** The core algorithms (AOL preconditioning, dynamic polynomial coefficients, Triton kernels) are entirely from the Turbo-Muon paper by Boissin et al. If you need the official implementation, please use **[flash-newton-schulz](https://github.com/thib-s/flash-newton-schulz)**.

### What this repo provides

- **Single-file implementation** (`fastmuon.py`): Original Muon + Turbo-Muon in one file, easy to copy-paste
- **Benchmark script** (`benchmark.py`): Reproduces speed & quality comparisons from scratch
- **Minor engineering fixes**: 1D parameter safety, relaxed param group validation
- **Bilingual documentation**: English + Chinese explanation of how Turbo-Muon works

### Turbo-Muon vs Original Muon (summary from the paper)

| Feature | Original Muon | Turbo-Muon |
|---|---|---|
| Normalization | Frobenius norm | AOL preconditioning |
| NS Coefficients | Fixed `(3.4445, -4.7750, 2.0315)` | Dynamic per-iteration coefficients |
| NS Iterations | 5 | 4 (equal or better quality) |
| Triton Kernels | ❌ | ✅ (auto-fallback to PyTorch) |

## Benchmark Reproduction

We independently ran benchmarks to verify the claims in the Turbo-Muon paper.

**Environment:** PyTorch 2.10.0+cu128, Triton 3.6.0, NVIDIA RTX 5090 D

### Newton-Schulz Speed

| Matrix Size | Original (5 iter) | FastMuon PyTorch (4 iter) | FastMuon Triton (4 iter) |
|---|---|---|---|
| 1×512×512 | 0.20 ms | 0.18 ms (1.14×) | 0.31 ms |
| 1×1024×1024 | 0.42 ms | 0.35 ms (1.20×) | 0.32 ms (1.34×) |
| 1×2048×2048 | 1.64 ms | 1.33 ms (1.23×) | 0.86 ms (1.91×) |
| 1×4096×4096 | 11.91 ms | 9.61 ms (1.24×) | **5.48 ms (2.17×)** |
| 8×1024×1024 | 1.65 ms | 1.37 ms (1.21×) | 0.98 ms (1.70×) |
| 16×1024×1024 | 3.45 ms | 2.84 ms (1.21×) | **1.89 ms (1.82×)** |
| 1×2048×512 | 0.32 ms | 0.28 ms (1.15×) | 0.27 ms (1.17×) |
| 1×4096×1024 | 0.94 ms | 0.77 ms (1.22×) | **0.45 ms (2.10×)** |

> Triton kernels shine on larger matrices (≥1024×1024). For small matrices, PyTorch implementation overhead is lower.

### Orthogonality Quality (Polar Error ↓)

| Matrix Size | Original (5 iter, Frob) | FastMuon (4 iter, AOL) | FastMuon (5 iter, AOL) |
|---|---|---|---|
| 1×512×512 | 0.3225 | **0.1309** | **0.0726** |
| 1×1024×1024 | 0.3481 | **0.1412** | **0.0777** |
| 1×2048×2048 | 0.3621 | **0.1517** | **0.0826** |
| 1×4096×4096 | 0.4015 | **0.1637** | **0.0882** |
| 8×1024×1024 | 0.3479 | **0.1405** | **0.0781** |

> FastMuon achieves **~2.5× lower polar error** with 4 iterations compared to the original's 5 iterations.

### Optimizer Step Timing

| Config | Original (5 iter) | FastMuon (4 iter) | Speedup |
|---|---|---|---|
| 8 params @ 512×512 | 1.38 ms | 2.09 ms | 0.66× |
| 8 params @ 1024×1024 | 3.35 ms | 2.07 ms | **1.62×** |
| 4 params @ 2048×2048 | 6.66 ms | 3.49 ms | **1.91×** |

> For small matrices (512×512), the Triton kernel launch overhead outweighs the benefit. For ≥1024, FastMuon provides consistent speedups up to **1.91×**.

## Installation

```bash
# Clone the repository
git clone https://github.com/Decem-Y/FastMuon.git
cd FastMuon

# (Optional) Install Triton-accelerated kernels for additional speedup
pip install kernels
```

## Quick Start

### Single-Device Usage

```python
from fastmuon import SingleDeviceMuon

# Collect hidden weight parameters (2D or 4D conv)
hidden_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]

optimizer = SingleDeviceMuon(hidden_params, variant="turbo", lr=0.02, momentum=0.95)
```

### With Auxiliary Adam (Recommended for full network)

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

### Distributed Usage

```python
from fastmuon import Muon, MuonWithAuxAdam

# Drop-in replacement for distributed training
optimizer = Muon(hidden_params, variant="turbo", lr=0.02)
# or with auxiliary Adam
optimizer = MuonWithAuxAdam(param_groups)
```

### Low-Level Newton-Schulz

```python
from fastmuon import newton_schulz

# Orthogonalize a gradient matrix (auto-selects Triton if available)
G = torch.randn(16, 1024, 1024, device="cuda")
X = newton_schulz(G, iter=4, precondition=True)
```

## Variant Options

| Variant | Preconditioning | NS Iterations | Speed | Recommended |
|---|---|---|---|---|
| `turbo` (default) | AOL | 4 | Fastest | ✅ General use |
| `standard` | Frobenius | 5 | Baseline | Legacy compatibility |

## How Turbo-Muon Works (from the paper)

The following is a summary of the techniques from [Boissin et al., 2025](https://hal.science/hal-05390446). All credit goes to the original authors.

### 1. AOL Preconditioning (replaces Frobenius normalization)

Instead of normalizing by the Frobenius norm:
```python
X = X / (X.norm() + eps)  # Frobenius (original)
```

FastMuon uses Absolute-value One-norm Layer (AOL) rescaling via the Gram matrix:
```python
A = X @ X.mT                                               # Gram matrix
s = torch.rsqrt(clamp_min(A.abs().sum(dim=-1), min=eps))   # AOL scaling
X = X * s.unsqueeze(-1)                                    # row-wise scaling
A = A * s.unsqueeze(-1) * s.unsqueeze(-2)                  # reuse Gram matrix
```

The Gram matrix is already needed for the first NS iteration, so AOL preconditioning has **near-zero overhead** while providing a much better starting point for convergence.

### 2. Dynamic Polynomial Coefficients

Each NS iteration uses tuned coefficients instead of fixed ones:
```python
ns_consts = [
    (4.0848, -6.8946, 2.9270),  # iter 1: more aggressive
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),  # iter 5: more conservative
][-iter:]
```

### 3. Triton-Fused Kernels

When the `kernels` library is installed, FastMuon automatically uses Triton CUDA kernels that fuse matrix operations, reducing GPU memory bandwidth overhead.

## Running the Benchmark

```bash
cd FastMuon
python benchmark.py
```

## API Reference

### Classes

| Class | Description | Distributed |
|---|---|---|
| `Muon` | Core Muon optimizer | ✅ |
| `SingleDeviceMuon` | Single-GPU Muon | ❌ |
| `MuonWithAuxAdam` | Muon + AdamW for non-Muon params | ✅ |
| `SingleDeviceMuonWithAuxAdam` | Single-GPU Muon + AdamW | ❌ |

### Functions

| Function | Description |
|---|---|
| `newton_schulz(G, iter, precondition, epsilon, dtype)` | NS orthogonalization (auto Triton/PyTorch) |
| `muon_update(grad, momentum, beta, ns_steps, nesterov, precondition, epsilon)` | Full Muon update step |
| `zeropower_via_newtonschulz5(G, steps)` | Legacy wrapper (Frobenius norm) |

## Minor Engineering Additions

Beyond the faithful reproduction, this repo adds a few small engineering fixes:

- 1D parameters (bias, gain) are safely skipped instead of crashing
- Removed strict `assert set(group.keys())` checks, allowing extra parameters in param groups
- `zeropower_via_newtonschulz5()` is preserved as a legacy-compatible wrapper
- Default `variant="turbo"` for seamless upgrade

## References & Acknowledgments

The core algorithms in this repository are from the following works. **Please cite them, not this repo.**

- **Turbo-Muon Paper** (the primary source for this repo): Boissin, T., Massena, T., Mamalet, F., & Serrurier, M. [Turbo-Muon: Accelerating Orthogonality-Based Optimization with Pre-Conditioning](https://hal.science/hal-05390446), 2025
- **Flash Newton-Schulz** (official Turbo-Muon implementation): [thib-s/flash-newton-schulz](https://github.com/thib-s/flash-newton-schulz)
- **Muon Optimizer**: Jordan, K. [MomentUm Orthogonalized by Newton-schulz](https://kellerjordan.github.io/posts/muon/)
- **AOL Rescaling**: Prach, B. & Lampert, C.H. [Almost-Orthogonal Layers for Efficient General-Purpose Lipschitz Networks](https://arxiv.org/abs/2208.03160), ECCV 2022
- **Triton Kernels**: Derived from [microsoft/dion](https://github.com/microsoft/dion) Newton-Schulz Triton implementation
- **Batched Muon**: @scottjmaddox, record implementation by @YouJiacheng
- **Quintic Polynomial Strategy**: Suggested by @jxbz, @leloykun, @YouJiacheng
- **kernels Library**: [kernels](https://pypi.org/project/kernels/) — Hub for loading Triton kernels

## Citation

Please cite the original Muon and Turbo-Muon papers, **not this repository**:

```bibtex
@misc{jordan2024muon,
  author       = {Keller Jordan and Yuchen Jin and Vlado Boza and You Jiacheng and
                  Franz Cesista and Laker Newhouse and Jeremy Bernstein},
  title        = {Muon: An optimizer for hidden layers in neural networks},
  year         = {2024},
  url          = {https://kellerjordan.github.io/posts/muon/}
}

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

## License

MIT License
