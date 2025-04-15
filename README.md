# DASMatrix

<div align="center">
  <img src="./docs/assets/logo.png" alt="DASMatrix Logo" width="200"/>
  <h3>分布式声学传感数据处理与分析框架</h3>
</div>

## 项目简介

DASMatrix 是一个专为分布式声学传感（DAS）数据处理和分析设计的高性能 Python 库。该框架提供了一整套工具，用于读取、处理、分析和可视化 DAS 数据，适用于地球物理学、结构健康监测、安防监控等领域的研究和应用。

### 核心特性

- **高效数据读取**：支持 DAT、HDF5 等多种数据格式
- **专业信号处理**：提供频谱分析、滤波、积分等多种信号处理功能
- **科学级可视化**：包含时域波形图、频谱图、时频图、瀑布图等多种可视化方式
- **高性能设计**：关键算法采用向量化和并行计算优化
- **易用的接口**：提供直观的 API，便于快速上手和集成

## 安装指南

### 环境要求

- Python 3.8+
- NumPy 1.20+
- SciPy 1.7+
- Matplotlib 3.4+
- h5py 3.6+

### 安装步骤

```bash
# 从 GitHub 克隆仓库
git clone https://github.com/yourusername/DASMatrix.git
cd DASMatrix

# 安装依赖
pip install -r requirements.txt

# 安装库
pip install -e .
```

## 基本使用

### 数据读取示例

```python
from DASMatrix.acquisition import DASReader, DataType
from DASMatrix.config import SamplingConfig

# 配置采样参数
sampling_config = SamplingConfig(
    fs=10000,      # 采样频率 10kHz
    channels=512,  # 512 个通道
    wn=5.0,        # 5Hz 高通滤波
    byte_order="big"
)

# 创建数据读取器
reader = DASReader(sampling_config, data_type=DataType.DAT)

# 读取数据
raw_data = reader.ReadRawData("path/to/data.dat")
```

### 数据处理示例

```python
from DASMatrix.processing import DASProcessor

# 创建处理器
processor = DASProcessor(sampling_config)

# 处理数据
diff_data = processor.ProcessDifferential(raw_data)        # 差分数据处理
int_data = processor.IntegrateData(raw_data)               # 积分数据处理
spectrum = processor.ComputeSpectrum(diff_data, channel_index=100)  # 计算频谱
peaks = processor.FindPeakFrequencies(spectrum, n_peaks=3)          # 查找峰值频率
```

### 可视化示例

```python
from DASMatrix.visualization import DASVisualizer
import matplotlib.pyplot as plt

# 创建可视化器
visualizer = DASVisualizer(
    output_path="./output",
    sampling_frequency=sampling_config.fs
)

# 时域波形图
visualizer.WaveformPlot(
    diff_data[:, 100],          # 第100通道的时域数据
    time_range=(0, 10),         # 显示0-10秒
    amplitude_range=(-0.5, 0.5),# 幅值范围
    title="Waveform Plot",
    file_name="waveform_ch100"
)

# 频谱图
visualizer.SpectrumPlot(
    diff_data[:, 100],          # 第100通道的时域数据
    title="Spectrum Plot",
    db_range=(-80, 0),          # dB范围
    file_name="spectrum_ch100"
)

# 时频图
visualizer.SpectrogramPlot(
    diff_data[:, 100],          # 第100通道的时域数据
    freq_range=(0, 500),        # 频率范围
    time_range=(0, 10),         # 时间范围
    cmap="inferno",             # 颜色映射
    file_name="spectrogram_ch100"
)

# 瀑布图(时间-通道)
visualizer.WaterfallPlot(
    diff_data,                  # 二维数据(时间 x 通道)
    title="Waterfall Plot",
    colorbar_label="Amplitude",
    value_range=(-0.5, 0.5),    # 幅值范围
    file_name="waterfall"
)

plt.show()  # 显示图形(如果在脚本中运行)
```

## 项目结构

```
DASMatrix/
├── acquisition/           # 数据获取模块
│   ├── das_reader.py      # DAS数据读取类
├── config/                # 配置模块
│   ├── sampling_config.py # 采样配置
│   ├── visualization_config.py  # 可视化配置
├── processing/            # 数据处理模块
│   ├── das_processor.py   # DAS数据处理类
│   ├── numba_filters.py   # Numba优化滤波器
├── visualization/         # 可视化模块
│   ├── das_visualizer.py  # DAS可视化类
├── utils/                 # 工具函数
```

## 主要模块说明

### 数据获取模块 `DASMatrix.acquisition`

该模块负责从不同数据源和格式读取 DAS 原始数据：
- `DASReader`：主要读取器类，支持多种数据格式
- `DataType`：数据类型枚举（DAT、HDF5 等）

### 数据处理模块 `DASMatrix.processing`

该模块提供各种 DAS 数据处理和分析功能：
- `DASProcessor`：主要处理器类，实现各种信号处理算法
- 支持滤波、积分、频谱分析、峰值检测等功能

### 可视化模块 `DASMatrix.visualization`

该模块提供科学级的数据可视化能力：
- `DASVisualizer`：主要可视化器类，封装各种绘图功能
- 支持时域波形图、频谱图、时频图、瀑布图等多种可视化

### 配置模块 `DASMatrix.config`

该模块提供配置类和参数设置：
- `SamplingConfig`：采样和数据格式相关配置
- `VisualizationConfig`：可视化样式和参数配置

## 贡献

欢迎贡献代码、提出问题或建议。请通过 GitHub Issues 和 Pull Requests 参与项目开发。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
