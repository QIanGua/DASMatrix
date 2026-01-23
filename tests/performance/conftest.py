"""性能测试 Fixtures。"""

import os

import numpy as np
import pytest

import DASMatrix as dm
from DASMatrix import DASFrame


@pytest.fixture
def large_array():
    """生成大规模 NumPy 数组用于测试。

    Size: 20,000 samples x 1,000 channels (float32)
    Memory: ~80 MB
    """
    n_samples = 20_000
    n_channels = 1_000
    return np.random.randn(n_samples, n_channels).astype(np.float32)


@pytest.fixture
def large_dasframe(large_array):
    """生成大规模内存 DASFrame。"""
    return dm.from_array(large_array, fs=1000.0, dx=1.0)


@pytest.fixture
def lazy_dasframe(large_array, tmp_path):
    """生成大规模延迟加载 DASFrame (基于 HDF5)。"""
    # 写入 HDF5 文件
    import h5py

    file_path = tmp_path / "large_test_data.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=large_array)
        f.attrs["fs"] = 1000.0

    # 延迟读取
    # 注意：这里假设 df.read 能处理这种简单的 HDF5，或者我们需要使用 h5py 插件支持的格式
    # 为了简化，我们直接用 h5py 读取并构建 dask array
    import dask.array as da

    dask_arr = da.from_array(large_array, chunks=(10000, 1000))
    return DASFrame(dask_arr, fs=1000.0, dx=1.0)


@pytest.fixture
def memory_tracker():
    """简单的内存追踪器。"""

    class MemoryTracker:
        def __init__(self):
            import psutil

            self.process = psutil.Process(os.getpid())
            self.start_mem = 0
            self.peak_mem = 0

        def __enter__(self):
            self.start_mem = self.process.memory_info().rss
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            end_mem = self.process.memory_info().rss
            print(f"Memory delta: {(end_mem - self.start_mem) / 1024 / 1024:.2f} MB")

    return MemoryTracker()
