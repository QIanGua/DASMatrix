# DASMatrix

**分布式声学传感数据处理与分析框架**

---

## 简介

DASMatrix 是一个专为分布式声学传感（DAS）数据处理和分析设计的高性能 Python 库。该框架提供了一整套工具，用于读取、处理、分析和可视化 DAS 数据，适用于地球物理学、结构健康监测、安防监控等领域的研究和应用。

## 核心特性

- 🚀 **高效数据读取**：支持 DAT、HDF5 等多种数据格式
- 📊 **专业信号处理**：提供频谱分析、滤波、积分等多种信号处理功能
- 🎨 **科学级可视化**：包含时域波形图、频谱图、时频图、瀑布图等
- ⚡ **高性能设计**：关键算法采用向量化、Numba JIT 编译优化
- 🔗 **链式 API**：流畅的 DASFrame API，支持延迟计算

## 快速安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/DASMatrix.git
cd DASMatrix

# 使用 uv 安装 (推荐)
uv sync

# 或使用 pip
pip install -e .
```

## 快速示例

```python
from DASMatrix import df

# 创建 DASFrame 并链式处理
result = (
    df(data, fs=1000)
    .detrend()
    .bandpass(1, 100)
    .normalize()
)

# 可视化
result.plot_heatmap(title="Processed DAS Data")
```

## 下一步

- [快速开始](quickstart.md) - 详细的入门教程
- [API 文档](api/index.md) - 完整的 API 参考
- [贡献指南](contributing.md) - 如何参与项目开发
