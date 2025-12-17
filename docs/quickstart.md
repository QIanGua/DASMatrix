# 快速开始

本指南将帮助您快速上手 DASMatrix。

## 安装

### 环境要求

- Python 3.9+
- NumPy 1.20+
- SciPy 1.10+
- Matplotlib 3.7+

### 使用 uv 安装（推荐）

```bash
git clone https://github.com/yourusername/DASMatrix.git
cd DASMatrix
uv sync
```

### 使用 pip 安装

```bash
pip install -e .
```

---

## 基本使用

### 1. 数据读取

```python
from DASMatrix.acquisition import DASReader, DataType
from DASMatrix.config import SamplingConfig

# 配置采样参数
config = SamplingConfig(
    fs=10000,      # 采样频率 10kHz
    channels=512,  # 512 个通道
    byte_order="big"
)

# 创建读取器并读取数据
reader = DASReader(config, data_type=DataType.DAT)
data = reader.ReadRawData("path/to/data.dat")

# 读取 SEGY 文件
segy_reader = DASReader(config, data_type=DataType.SEGY)
segy_data = segy_reader.ReadRawData("path/to/data.sgy")
```

### 2. 使用 DASFrame 进行链式处理

```python
from DASMatrix import df

# 创建 DASFrame
frame = df(data, fs=10000)

# 链式信号处理
processed = (
    frame
    .detrend()              # 去趋势
    .demean()               # 去均值
    .bandpass(1, 500)       # 1-500Hz 带通滤波
    .fk_filter(v_min=1500)  # FK 速度滤波 (>1500m/s)
    .normalize()            # 归一化
)

# 获取处理结果
result = processed.collect()
```

### 3. 可视化

```python
# 时间序列图
processed.plot_ts(ch=0, title="Channel 0")

# 热图/瀑布图
processed.plot_heatmap(title="DAS Waterfall")

# 频谱图
processed.plot_spec(ch=0, title="Spectrogram")

# FK 谱图
processed.plot_fk(dx=1.0, v_lines=[1500, 3000], title="FK Spectrum")
```

---

## 常用操作

### 滤波器

| 方法 | 说明 |
|------|------|
| `bandpass(low, high)` | 带通滤波 |
| `lowpass(cutoff)` | 低通滤波 |
| `highpass(cutoff)` | 高通滤波 |
| `notch(freq)` | 陷波滤波 |
| `median_filter(k)` | 中值滤波 |

### 变换

| 方法 | 说明 |
|------|------|
| `fft()` | 快速傅里叶变换 |
| `stft()` | 短时傅里叶变换 |
| `hilbert()` | 希尔伯特变换 |
| `envelope()` | 包络提取 |

### 统计

| 方法 | 说明 |
|------|------|
| `mean(axis)` | 均值 |
| `std(axis)` | 标准差 |
| `max(axis)` | 最大值 |
| `min(axis)` | 最小值 |
| `rms()` | 均方根 |

---

## 下一步

- 查看 [API 文档](api/index.md) 了解完整功能
- 阅读 [贡献指南](contributing.md) 参与项目开发
