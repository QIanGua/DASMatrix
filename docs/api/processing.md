# 信号处理

DASMatrix 提供了丰富的信号处理功能，主要通过 `DASFrame` 链式 API 和底层引擎实现。

## 处理引擎

DASMatrix 使用混合执行引擎（HybridEngine），支持：

- **NumPy/SciPy 后端** - 通用兼容性
- **Numba JIT 后端** - 高性能数值计算
- **算子融合优化** - 减少内存开销

---

## API 参考

### HybridEngine

::: DASMatrix.processing.engine.HybridEngine

### DASProcessor

::: DASMatrix.processing.das_processor.DASProcessor
