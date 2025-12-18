# 计算图 (Legacy)

> [!WARNING]
> 从 v0.2.0 开始，DASMatrix 已迁移至 **Xarray/Dask** 架构。自定义的 `ComputationGraph` 已被废弃，仅作为内部参考保留。新的延迟计算机制由 Dask 原生支持。

DASMatrix 早期版本使用此计算图实现延迟计算。

## 概述 (Legacy)

(此部分仅作为历史参考)

计算图是 DASFrame 延迟计算的核心机制... (已被 Dask Graph 取代)

---

## API 参考 (Deprecated)

### ComputationGraph

::: DASMatrix.core.computation_graph.ComputationGraph
