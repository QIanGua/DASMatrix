# API 文档

DASMatrix 的 API 按功能模块组织。
本页面提供了自动生成的 API 参考文档，始终与代码保持同步。

## 核心模块

- **[DASFrame](../reference/DASMatrix/api/dasframe.md)** - 核心数据处理类，基于 Xarray/Dask
- **[数据读取](../reference/DASMatrix/acquisition/index.md)** - 支持多种格式的数据读取
- **[信号处理](../reference/DASMatrix/processing/index.md)** - 丰富的信号处理算法库
- **[可视化](../reference/DASMatrix/visualization/index.md)** - 科学级数据可视化工具

## 浏览完整 API

您可以从左侧导航栏或以下链接浏览完整的模块参考：

- [DASMatrix 核心](../reference/DASMatrix/index.md)
- [API 接口](../reference/DASMatrix/api/index.md)
- [配置管理](../reference/DASMatrix/config/index.md)

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
