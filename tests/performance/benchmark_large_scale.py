import os
import time

import dask.array as da
import psutil

from DASMatrix.api.dasframe import DASFrame


def print_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1024**2:.2f} MB")


def benchmark_lazy_execution():
    print("=== Scalability Benchmark ===")

    # 1. Create a large synthetic dataset (lazy)
    # Simulate: 10,000 channels, 100,000 time samples (float32)
    # Size: 10,000 * 100,000 * 4 bytes = 4 GB
    # To stress test, let's go bigger or check chunks

    n_channels = 1000
    n_time = 500000
    fs = 1000.0

    print(f"Generating synthetic data: {n_channels} channels x {n_time} samples")
    print(f"Estimated Size: {n_channels * n_time * 8 / 1024**3:.2f} GB (float64)")

    # Chunking: 1000 samples in time, full channels
    chunk_t = 10000
    chunks = (chunk_t, n_channels)

    # random.random is lazy in dask
    dask_data = da.random.random((n_time, n_channels), chunks=chunks)

    # 2. Initialize DASFrame
    print_memory()
    start_init = time.time()
    df = DASFrame(dask_data, fs=fs, dx=1.0)
    print(f"Initialization Time: {time.time() - start_init:.4f} s")
    print_memory()

    # 3. Build Computation Graph (Lazy)
    print("\nBuilding Lazy Pipeline...")
    start_build = time.time()

    # Pipeline: Detrend -> Bandpass -> Normalize
    # This just builds the dask graph
    processed = df.detrend(axis="time").bandpass(1, 100).normalize()

    print(f"Graph Build Time: {time.time() - start_build:.4f} s")
    print_memory()

    # 4. Execute (Compute)
    # We won't collect everything to memory because RAM may be limited
    # Let's compute a reduction to prove it streams
    print("\nExecuting (Reduction: Mean)...")

    start_exec = time.time()
    result = processed.data.mean().compute()

    print(f"Execution Time: {time.time() - start_exec:.4f} s")
    print(f"Result: {result}")
    print_memory()

    # 5. Check actual underlying graph size/complexity if needed
    # dask_data.dask


if __name__ == "__main__":
    benchmark_lazy_execution()
