# 计算图

DASMatrix 使用计算图（Computation Graph）实现延迟计算，支持算子融合优化。

## 概述

计算图是 DASFrame 延迟计算的核心机制：

1. 用户调用方法（如 `.bandpass()`）时，不立即执行
2. 系统构建操作节点并加入计算图
3. 调用 `.collect()` 时，优化器融合可合并的操作
4. 执行引擎按优化后的计划执行

```text
用户操作 → 构建计算图 → 优化（算子融合） → 执行
```

---

## API 参考

### ComputationGraph

::: DASMatrix.core.computation_graph.ComputationGraph

### Node

::: DASMatrix.core.computation_graph.Node

### SourceNode

::: DASMatrix.core.computation_graph.SourceNode

### OperationNode

::: DASMatrix.core.computation_graph.OperationNode

### FusionNode

::: DASMatrix.core.computation_graph.FusionNode

### NodeDomain

::: DASMatrix.core.computation_graph.NodeDomain
