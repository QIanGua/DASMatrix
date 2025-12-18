# DASFrame

DASFrame 是 DASMatrix 的核心数据处理类，基于 **Xarray** 和 **Dask** 构建。

## 概述

DASFrame 采用 **Xarray** 作为数据容器，利用 **Dask** 实现延迟计算（Lazy Evaluation）和外存处理（Out-of-Core Processing）。

### 主要特性

1. **延迟计算**: 信号处理操作（如滤波、去趋势）返回构建好的计算图，不会立即消耗内存。
2. **外存处理**: 支持处理超过内存大小的超大规模 DAS 数据集（TB 级）。
3. **多维标签**: 保留时间和距离维度的物理坐标，防止元数据丢失。

```python
from DASMatrix import df

# 1. 初始化 (支持 numpy array, dask array, xarray DataArray)
frame = df(data, fs=1000)

# 2. 链式操作 (构建 Dask 计算图)
processed = (
    frame
    .detrend(axis="time")
    .bandpass(1, 100)
    .normalize()
)

# 3. 触发计算
# collect() 会触发 dask.compute() 并返回 numpy 数组
result = processed.collect()
```

---

## API 参考

::: DASMatrix.api.dasframe.DASFrame
