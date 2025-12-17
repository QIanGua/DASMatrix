import os
import sys
import time

import numpy as np
from scipy import signal

# Add project root to path
sys.path.append(os.getcwd())

from DASMatrix.api.dasframe import DASFrame


def benchmark():
    print("=" * 60)
    print("🚀 DASMatrix Engine Benchmark: NumPy vs Hybrid Tensor Engine")
    print("=" * 60)

    # 1. Prepare Data
    # 2GB Memory Bandwidth Pressure is needed to see huge diff,
    # but let's start with ~400MB to be safe on laptop.
    # 50,000 samples * 2,000 channels * 4 bytes = 400 MB
    n_samples = 50000
    n_channels = 2000
    print(f"Generating Data: {n_samples} samples x {n_channels} channels (float32)")
    print(f"Data Size: {n_samples * n_channels * 4 / 1024 / 1024:.2f} MB")

    data = np.random.randn(n_samples, n_channels).astype(np.float32)
    # Make a copy for fair comparison
    data_numpy = data.copy()
    data_hybrid = data.copy()

    # --- Scenario A: Old Engine (NumPy/SciPy) ---
    print("\n[Scenario A] Running Standard NumPy/SciPy Stack...")
    start_time = time.time()

    # 1. Detrend (Time-consuming, allocates new array or modifies in place depending on impl)
    # scipy.signal.detrend returns new array by default
    step1 = signal.detrend(data_numpy, axis=0)

    # 2. Abs (Allocates new array)
    step2 = np.abs(step1)

    # 3. Scale (Allocates new array)
    result_numpy = step2 * 0.5

    end_time = time.time()
    numpy_duration = end_time - start_time
    print(f"✅ NumPy Finished in: {numpy_duration:.4f} s")

    # --- Scenario B: Hybrid Engine (Fused) ---
    print("\n[Scenario B] Running Hybrid Tensor Engine (Fused)...")

    # Warmup (Compilation overhead included in first run usually)
    # Let's measure cold start (including compilation) vs warm start

    df = DASFrame(data_hybrid, fs=1000)

    # Define Lazy Graph
    # Note: These map to the stubs we implemented in DASFrame
    lazy_op = (
        df.detrend(axis="time")  # "detrend" op
        .abs()  # "abs" op
        .scale(factor=0.5)  # "scale" op
    )

    # First Run (Cold Start - Includes JIT Compilation)
    start_time = time.time()
    result_hybrid_cold = lazy_op.collect()
    end_time = time.time()
    cold_duration = end_time - start_time
    print(f"⚠️  Cold Start (w/ JIT Compile): {cold_duration:.4f} s")

    # Second Run (Warm Start - Pure Execution)
    start_time = time.time()
    result_hybrid_warm = lazy_op.collect()
    end_time = time.time()
    warm_duration = end_time - start_time
    print(f"✅ Hybrid Warm Run:             {warm_duration:.4f} s")

    # --- Comparison ---
    print("\n" + "-" * 60)
    print("📊 Performance Comparison Results")
    print("-" * 60)
    print(f"NumPy Time:       {numpy_duration:.4f} s")
    print(f"Hybrid Warm Time: {warm_duration:.4f} s")
    speedup = numpy_duration / warm_duration
    print(f"🚀 Speedup Factor: {speedup:.2f}x")

    # --- Correctness Check ---
    print("-" * 60)
    print(
        "Validation: Checking if Hybrid Engine result matches standard SciPy result..."
    )

    # Check shape
    assert result_hybrid_warm.shape == result_numpy.shape
    assert result_hybrid_warm.dtype == result_numpy.dtype

    # Check values (allow some float tolerance due to float32 and different algos)
    # SciPy uses lstsq, we used dot-product analytic solution. Should be very close.
    is_close = np.allclose(result_hybrid_warm, result_numpy, rtol=1e-3, atol=1e-3)

    if is_close:
        print("✅ Correctness Verified! Results match SciPy within tolerance.")
    else:
        print("❌ Corectness Mismatch!")
        # Debug info
        diff = np.abs(result_hybrid_warm - result_numpy)
        print(f"Max Diff: {np.max(diff)}")
        print(f"Mean Diff: {np.mean(diff)}")


if __name__ == "__main__":
    benchmark()
