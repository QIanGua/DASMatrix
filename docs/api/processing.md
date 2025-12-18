# 信号处理

DASMatrix 的信号处理基于 **Xarray** 和 **Dask** 生态系统，实现了高性能的并行计算。

## 处理引擎

新的架构不再依赖自定义的 `HybridEngine`，而是直接利用 Dask 的调度器进行计算图优化和执行。

- **Dask Array**: 数据被分块（Chunks），支持并行处理。
- **Xarray**: 提供维度感知（Dimension-aware）的操作，如 `apply_ufunc`，自动处理广播和坐标对齐。
- **Scipy/Numpy**: 底层数值计算仍使用标准的 Scipy/Numpy 函数，通过 `map_blocks` 并行应用。

### 优势

- **无需手动算子融合**: Dask 自动优化计算图。
- **零内存拷贝**: 链式操作在计算图中传递引用。
- **透明扩展**: 代码无需修改即可在多核 CPU 或集群上运行。

---

## API 参考

### Signal Operations (Lazy)

所有 DASFrame 的信号处理方法（如 `bandpass`, `fft`）均在 `DASMatrix.api.dasframe.DASFrame` 中定义。

### Legacy Processors

某些特定算法（如 FK 滤波的核心实现）可能仍使用独立的 Processor 类，但在 DASFrame 中被封装。

::: DASMatrix.processing.das_processor.DASProcessor
