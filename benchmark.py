"""
FastMuon Benchmark: Compare original Muon vs FastMuon (Turbo) Newton-Schulz implementations.

Tests:
  1. Newton-Schulz speed comparison at various matrix sizes
  2. Orthogonality quality (polar error)
  3. End-to-end optimizer step timing
"""

import sys
import time
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Import original Muon Newton-Schulz (Frobenius norm, fixed coeffs, 5 iters)
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")
from muon import zeropower_via_newtonschulz5 as ns_original

# ---------------------------------------------------------------------------
# Import FastMuon Newton-Schulz (AOL precondition, dynamic coeffs, 4 iters)
# ---------------------------------------------------------------------------
from fastmuon import newton_schulz as ns_fast, _newton_schulz_pytorch as ns_fast_pytorch, _USE_TRITON_NS


def polar_error(X, G):
    """
    Compute the polar error: ||X^T X - I||_F / sqrt(min(m,n))
    Measures how close X is to an orthogonal matrix.
    """
    if X.ndim == 3:
        # Batched
        if X.size(-2) <= X.size(-1):
            gram = X @ X.mT
        else:
            gram = X.mT @ X
        I = torch.eye(gram.size(-1), device=gram.device, dtype=torch.float32).unsqueeze(0)
        err = (gram.float() - I).norm(dim=(-2, -1)) / (gram.size(-1) ** 0.5)
        return err.mean().item()
    else:
        if X.size(-2) <= X.size(-1):
            gram = X @ X.mT
        else:
            gram = X.mT @ X
        I = torch.eye(gram.size(-1), device=gram.device, dtype=torch.float32)
        return ((gram.float() - I).norm() / (gram.size(-1) ** 0.5)).item()


def benchmark_ns_speed(batch_size, m, n, num_warmup=10, num_iters=50):
    """Benchmark Newton-Schulz speed for a given matrix size."""
    G = torch.randn(batch_size, m, n, device="cuda", dtype=torch.float32)

    results = {}

    # --- Original: Frobenius norm, fixed coeffs, 5 iters ---
    for _ in range(num_warmup):
        _ = ns_original(G.clone(), steps=5)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        _ = ns_original(G.clone(), steps=5)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters * 1000  # ms
    results["Original (5 iter, Frobenius)"] = elapsed

    # --- FastMuon PyTorch: AOL precondition, dynamic coeffs, 4 iters ---
    for _ in range(num_warmup):
        _ = ns_fast_pytorch(G.clone(), iter=4, precondition=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        _ = ns_fast_pytorch(G.clone(), iter=4, precondition=True)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters * 1000
    results["FastMuon PyTorch (4 iter, AOL)"] = elapsed

    # --- FastMuon with Triton (if available) ---
    if _USE_TRITON_NS:
        for _ in range(num_warmup):
            _ = ns_fast(G.clone(), iter=4, precondition=True)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            _ = ns_fast(G.clone(), iter=4, precondition=True)
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iters * 1000
        results["FastMuon Triton (4 iter, AOL)"] = elapsed

    return results


def benchmark_polar_error(batch_size, m, n, num_trials=10):
    """Compare orthogonality quality (polar error) of different implementations."""
    errors = {"Original (5 iter, Frobenius)": [], "FastMuon (4 iter, AOL)": [], "FastMuon (5 iter, AOL)": []}

    for _ in range(num_trials):
        G = torch.randn(batch_size, m, n, device="cuda", dtype=torch.float32)

        X_orig = ns_original(G.clone(), steps=5)
        errors["Original (5 iter, Frobenius)"].append(polar_error(X_orig, G))

        X_fast4 = ns_fast_pytorch(G.clone(), iter=4, precondition=True)
        errors["FastMuon (4 iter, AOL)"].append(polar_error(X_fast4, G))

        X_fast5 = ns_fast_pytorch(G.clone(), iter=5, precondition=True)
        errors["FastMuon (5 iter, AOL)"].append(polar_error(X_fast5, G))

    return {k: (np.mean(v), np.std(v)) for k, v in errors.items()}


def benchmark_optimizer_step(batch_size=8, dim=1024, num_warmup=5, num_iters=30):
    """Benchmark a full optimizer step (momentum + NS) for both variants."""
    from fastmuon import muon_update as fast_muon_update
    from muon import muon_update as orig_muon_update

    results = {}

    # --- Original muon_update ---
    grads = [torch.randn(dim, dim, device="cuda") for _ in range(batch_size)]
    momentums = [torch.zeros_like(g) for g in grads]

    for _ in range(num_warmup):
        for g, m in zip(grads, momentums):
            orig_muon_update(g.clone(), m, beta=0.95, ns_steps=5, nesterov=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        for g, m in zip(grads, momentums):
            orig_muon_update(g.clone(), m, beta=0.95, ns_steps=5, nesterov=True)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters * 1000
    results["Original muon_update (5 iter)"] = elapsed

    # --- FastMuon muon_update (turbo) ---
    grads2 = [torch.randn(dim, dim, device="cuda") for _ in range(batch_size)]
    momentums2 = [torch.zeros_like(g) for g in grads2]

    for _ in range(num_warmup):
        for g, m in zip(grads2, momentums2):
            fast_muon_update(g.clone(), m, beta=0.95, ns_steps=4, nesterov=True, precondition=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        for g, m in zip(grads2, momentums2):
            fast_muon_update(g.clone(), m, beta=0.95, ns_steps=4, nesterov=True, precondition=True)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters * 1000
    results["FastMuon muon_update (4 iter, turbo)"] = elapsed

    return results


def print_table(headers, rows, title=""):
    """Pretty print a table."""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")

    col_widths = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-+-".join("-" * w for w in col_widths)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))
    print()


def main():
    print("=" * 80)
    print("  FastMuon Benchmark")
    print("=" * 80)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"  Triton NS available: {_USE_TRITON_NS}")
    print("=" * 80)

    # =========================================================================
    # Test 1: Newton-Schulz Speed Benchmark
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Test 1: Newton-Schulz Speed Benchmark")
    print("=" * 80)

    configs = [
        (1, 512, 512),
        (1, 1024, 1024),
        (1, 2048, 2048),
        (1, 4096, 4096),
        (8, 1024, 1024),
        (16, 1024, 1024),
        (1, 2048, 512),   # non-square
        (1, 4096, 1024),  # non-square
    ]

    speed_rows = []
    for batch, m, n in configs:
        label = f"{batch}x{m}x{n}"
        res = benchmark_ns_speed(batch, m, n)
        base_time = res["Original (5 iter, Frobenius)"]

        row = [label, f"{base_time:.2f} ms"]
        pt_time = res["FastMuon PyTorch (4 iter, AOL)"]
        row.append(f"{pt_time:.2f} ms ({base_time/pt_time:.2f}x)")

        if "FastMuon Triton (4 iter, AOL)" in res:
            tr_time = res["FastMuon Triton (4 iter, AOL)"]
            row.append(f"{tr_time:.2f} ms ({base_time/tr_time:.2f}x)")
        else:
            row.append("N/A")

        speed_rows.append(row)

    headers = ["Matrix Size", "Original (5it)", "FastMuon PT (4it)", "FastMuon Triton (4it)"]
    print_table(headers, speed_rows, "Speed Results (lower = better)")

    # =========================================================================
    # Test 2: Orthogonality Quality (Polar Error)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Test 2: Orthogonality Quality (Polar Error)")
    print("=" * 80)

    error_configs = [
        (1, 512, 512),
        (1, 1024, 1024),
        (1, 2048, 2048),
        (1, 4096, 4096),
        (8, 1024, 1024),
    ]

    error_rows = []
    for batch, m, n in error_configs:
        label = f"{batch}x{m}x{n}"
        errs = benchmark_polar_error(batch, m, n)
        row = [label]
        for key in ["Original (5 iter, Frobenius)", "FastMuon (4 iter, AOL)", "FastMuon (5 iter, AOL)"]:
            mean, std = errs[key]
            row.append(f"{mean:.6f} ± {std:.6f}")
        error_rows.append(row)

    headers = ["Matrix Size", "Original (5it, Frob)", "FastMuon (4it, AOL)", "FastMuon (5it, AOL)"]
    print_table(headers, error_rows, "Polar Error Results (lower = better)")

    # =========================================================================
    # Test 3: Optimizer Step Timing
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Test 3: Optimizer Step Timing (muon_update)")
    print("=" * 80)

    opt_configs = [(8, 512), (8, 1024), (4, 2048)]
    opt_rows = []
    for batch, dim in opt_configs:
        label = f"{batch} params @ {dim}x{dim}"
        res = benchmark_optimizer_step(batch_size=batch, dim=dim)
        base = res["Original muon_update (5 iter)"]
        fast = res["FastMuon muon_update (4 iter, turbo)"]
        opt_rows.append([label, f"{base:.2f} ms", f"{fast:.2f} ms", f"{base/fast:.2f}x"])

    headers = ["Config", "Original (5it)", "FastMuon (4it)", "Speedup"]
    print_table(headers, opt_rows, "Optimizer Step Results")

    # =========================================================================
    # Test 4: Numerical Correctness
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Test 4: Numerical Correctness Verification")
    print("=" * 80)

    G = torch.randn(4, 1024, 1024, device="cuda", dtype=torch.float32)

    X_orig = ns_original(G.clone(), steps=5)
    X_fast_pt = ns_fast_pytorch(G.clone(), iter=4, precondition=True)

    # Check shape matches
    assert X_orig.shape == X_fast_pt.shape, f"Shape mismatch: {X_orig.shape} vs {X_fast_pt.shape}"
    print(f"  Shape check: PASS ({X_orig.shape})")

    # Check outputs are finite
    assert torch.isfinite(X_orig).all(), "Original output has non-finite values"
    assert torch.isfinite(X_fast_pt).all(), "FastMuon output has non-finite values"
    print(f"  Finiteness check: PASS")

    # Both should produce near-orthogonal matrices
    # Note: Original Muon by design produces S' ~ Uniform(0.5, 1.5), so polar error is ~0.35
    # FastMuon with AOL preconditioning achieves much better orthogonality (~0.14 for 4 iters)
    err_orig = polar_error(X_orig, G)
    err_fast = polar_error(X_fast_pt, G)
    print(f"  Polar error - Original (5it): {err_orig:.6f}")
    print(f"  Polar error - FastMuon (4it): {err_fast:.6f}")
    assert err_orig < 0.5, f"Original polar error too large: {err_orig}"
    assert err_fast < 0.3, f"FastMuon polar error too large: {err_fast}"
    assert err_fast < err_orig, f"FastMuon should have lower polar error than Original"
    print(f"  Quality check: PASS (FastMuon {err_fast:.4f} < Original {err_orig:.4f})")

    if _USE_TRITON_NS:
        X_fast_triton = ns_fast(G.clone(), iter=4, precondition=True)
        err_triton = polar_error(X_fast_triton, G)
        # Triton and PyTorch should produce similar results
        diff = (X_fast_triton.float() - X_fast_pt.float()).abs().mean().item()
        print(f"  Triton vs PyTorch mean abs diff: {diff:.8f}")
        print(f"  Triton polar error: {err_triton:.6f}")
        assert diff < 0.01, f"Triton-PyTorch difference too large: {diff}"
        print(f"  Triton consistency: PASS")

    # =========================================================================
    # Test 5: 1D Parameter Safety
    # =========================================================================
    print("\n" + "=" * 80)
    print("  Test 5: 1D Parameter Safety (Bias/Gain handling)")
    print("=" * 80)

    from fastmuon import muon_update as fast_muon_update

    # 1D grad should not crash and should return the momentum-updated grad
    grad_1d = torch.randn(256, device="cuda")
    momentum_1d = torch.zeros_like(grad_1d)
    result = fast_muon_update(grad_1d, momentum_1d, beta=0.95, ns_steps=4, precondition=True)
    assert result.shape == grad_1d.shape, f"1D shape mismatch: {result.shape}"
    assert torch.isfinite(result).all(), "1D result has non-finite values"
    print(f"  1D param handling: PASS (shape={result.shape})")

    # 4D grad (conv filter) should work
    grad_4d = torch.randn(64, 32, 3, 3, device="cuda")
    momentum_4d = torch.zeros_like(grad_4d)
    result_4d = fast_muon_update(grad_4d, momentum_4d, beta=0.95, ns_steps=4, precondition=True)
    print(f"  4D conv param handling: PASS (shape={result_4d.shape})")

    print("\n" + "=" * 80)
    print("  ALL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
