# DASMatrix

**分布式声学传感数据处理与分析框架**

---

## 简介

DASMatrix 是一个专为分布式声学传感（DAS）数据处理和分析设计的高性能 Python 库。
该框架提供了一整套工具，用于读取、处理、分析和可视化 DAS 数据，
适用于地球物理学、结构健康监测、安防监控等领域的研究和应用。

## 核心特性

- 🚀 **高效数据读取**：支持 12+ 种数据格式（DAT、HDF5、PRODML、Silixa、Febus、Terra15、APSensing、ZARR、NetCDF、SEG-Y、MiniSEED、TDMS）
- 📊 **专业信号处理**：提供滤波、频率分析、FK 滤波、自动增益控制等
- 🎨 **科学级可视化**：符合 Nature/Science 发表标准的波形图、瀑布图、FK 谱图
- ⚡ **高性能设计**：基于 Xarray 和 Dask 构建，支持 TB 级超大数据的延迟计算
- 🔗 **链式 API**：提供直观、流畅的 API，大大简化了处理流程

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
