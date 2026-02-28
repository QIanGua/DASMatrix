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

# 创建读取器并读取数据 (支持延迟加载)
# 返回 dask.array (lazy) 或 numpy.ndarray
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

# 热图/瀑布图 (支持切片查看)
processed.plot_heatmap(channels=slice(400, 700), title="Waterfall (Ch 400-700)")

# 频谱图
processed.plot_spec(ch=0, title="Spectrogram")

# FK 谱图
processed.plot_fk(dx=1.0, v_lines=[1500, 3000], title="FK Spectrum")

# 集成剖面图 (Profile Plots)
processed.plot_rms(
    x_axis="channel", channels=slice(40,100), title="RMS Profile"
)
processed.plot_mean(
    x_axis="channel", channels=slice(40,100), title="Mean Profile"
)
processed.plot_std(
    x_axis="channel", channels=slice(40,100), title="Std Profile"
)

> [!NOTE]
> 所有的绘图方法现在都支持**绝对坐标**。即使你对数据进行了 `.slice()` 切片，
> 图表的轴（时间、点位）也会正确显示原始的绝对位置，而不再从 0 开始。
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
| `stft()` | 短时傅里叶变换 (现代 ShortTimeFFT API) |
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
| `plot_rms()` | 绘制 RMS 剖面图 |
| `plot_mean()` | 绘制均值剖面图 |
| `plot_std()` | 绘制标准差剖面图 |

### AI 与检测

| 方法 | 说明 |
|------|------|
| `threshold_detect()` | 基于能量阈值的事件检测 |
| `template_match()` | 信号模板匹配 (NCC) |
| `predict(model)` | **[NEW]** 应用 AI 模型 (Torch/ONNX) |
| `sta_lta(n_sta, n_lta)` | 标准 STA/LTA 能量比检测 (Recursive) |

---


### 4. 极致性能：延迟计算与保护

DASMatrix 采用 "Lazy-by-Default" 策略：
- **链式算子**：`.bandpass().detrend()` 等操作不会立即计算，而是构建计算图。
- **自动分块**：后台自动并行化处理，支持处理大于内存的数据集。
- **绘图降采样**：
  ```python
  # 即使有 1 亿个采样点，该方法也会自动降采样至 2000 点进行快速显示
  processed.plot_heatmap(max_samples=2000)
  ```


### 5. 智能分析：模板匹配 (Template Matching)

针对特定信号（如微震、车辆行驶波形），DASMatrix 提供高性能的归一化互相关 (NCC) 匹配：

```python
# 定义或提取一个 1D 模板信号
template_1d = ... 
# 在所有通道上并行寻找该信号
matches = frame.template_match(template_1d).collect()

# 也可以进行 2D 时空模板匹配（寻找特定速度传播的波）
template_2d = ... # (time x distance)
matches_2d = frame.template_match(template_2d).collect()
```

### 6. 工业级多文件管理 (DASSpool)

对于海量小文件项目，使用 Spool 进行统一管理：

```python
# 扫描目录并开启持久化索引缓存
spool = dm.spool("data/*.h5", cache_path=".cache").update()

# 毫秒级元数据筛选
subset = spool.select(iu_model="QuantX", ProjectName="DeepWell")

# 虚拟合并为连续 Frame
long_frame = subset.to_frame()
```


### 7. 物理单位管理 (Units & SI Normalization)

DASMatrix 集成了 Pint 单位系统，支持物理量纲的自动转换与归一化：

```python
# 将原始读数（如相角 rad）转换为标准 SI 单位（应变 strain）
# 该方法会自动从元数据（标距、激光波长等）计算比例系数
si_frame = frame.to_standard_units()

# 显式单位换算
# 将单位从 m/s 转换为 nm/s
nanometer_frame = si_frame.convert_units("nm/s")
print(f"当前单位: {nanometer_frame.get_unit()}")
```

### 8. 有状态流式处理 (Atoms & Streaming)

使用 Atoms 框架进行分块处理，确保块边缘的数学连续性（如 IIR 滤波状态）：

```python
from DASMatrix.processing.atoms import Sequential, SosFilt, Partial

# 定义有状态处理流水线
pipeline = Sequential([
    Partial("demean"),
    SosFilt(sos_coeffs),  # 自动维护滤波器内部状态
    Partial("abs")
])

# 实时流处理循环
for chunk in stream_source:
# 每一块的处理都会继承上一块的滤波器状态，避免边缘跳变
    processed_chunk = pipeline(chunk)
```

### 9. AI 推理与 Agent 框架 (NEW)

DASMatrix 集成了深度学习推理能力，支持直接对 DASBlock/DASFrame 应用预训练模型：

```python
from DASMatrix.ml.model import TorchModel, ONNXModel
from DASMatrix.ml.pipeline import InferencePipeline

# 1. 初始化模型后端 (支持 CPU/CUDA)
model = ONNXModel("event_classifier.onnx", device="cpu")

# 2. 构建推理流水线 (包含预处理逻辑)
def my_preprocess(x):
    return (x - x.mean()) / x.std()

pipeline = InferencePipeline(model, preprocess_fn=my_preprocess)

# 3. 在 DASFrame 中应用
# 返回分类概率、置信度或标记结果
predictions = frame.predict(pipeline)

# 4. Agent 协作
# 让 Agent 寻找并解释信号中的异常模式
tools = dm.agent.DASAgentTools()
insight = tools.run_inference(data_id="...", model_path="...")
```

---

## 下一步

- 查看 [API 文档](api/index.md) 了解完整功能
- 阅读 [贡献指南](contributing.md) 参与项目开发
