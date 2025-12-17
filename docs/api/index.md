# API 文档

DASMatrix 的 API 按功能模块组织，主要包括以下部分：

## 模块概览

### 核心 API

- **[DASFrame](dasframe.md)** - 核心数据处理类，提供链式 API 进行信号处理

### 数据获取

- **[数据读取](reader.md)** - 支持 DAT、HDF5 等格式的数据读取

### 信号处理

- **[信号处理](processing.md)** - 滤波、变换、统计等信号处理功能

### 可视化

- **[可视化](visualization.md)** - 科学级数据可视化（波形图、频谱图、瀑布图等）

### 内部实现

- **[计算图](computation_graph.md)** - 延迟计算图实现

---

## 快速导入

```python
# 推荐方式：使用便捷函数
from DASMatrix import df

# 完整导入
from DASMatrix.api import DASFrame
from DASMatrix.acquisition import DASReader, DataType
from DASMatrix.config import SamplingConfig, VisualizationConfig
from DASMatrix.visualization import DASVisualizer
```
