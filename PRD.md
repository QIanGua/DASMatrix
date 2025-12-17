# 项目需求文档

设计一个完整的高性能分布式光纤声波传感(DAS)数据处理框架库，采用正交分解架构，确保各层职责明确、耦合度低，同时兼顾高性能和可扩展性。目标是创建一个"像 data.table 一样简单而强大"的 Python 包，专用于处理2D时空DAS数据（time × channel）。

基于 Python 进行开发，使用 uv 管理依赖，以 Polars/Arrow 为底层计算引擎，辅以 Numba/GPU 加速关键算子。采用模块化设计以支持良好的可维护性和可扩展性。

## 1. 框架设计核心理念

1. **性能优先**：针对DAS大数据量特点优化核心算法和数据结构，支持GB～TB级数据的毫秒级操作
2. **简单易用**：提供类似data.table的极简DSL语法，一行链式表达可完成筛选/变换/聚合操作
3. **可扩展性**：保持接口一致性和松耦合设计，支持插拔式算法、流式或分布式处理
4. **渐进式复杂度**：支持从简单到复杂的使用方式，满足不同用户需求
5. **易部署**：纯Python主体，可选C/CUDA扩展，支持简单pip安装

## 2. 核心诉求与技术落点

| 诉求 | 解释 | 技术落点 |
|---|---|---|
| 极简语法 | 一行链式表达完成筛选/变换/聚合 | DASFrame对象 + DSL |
| 高性能 | GB～TB 级数据毫秒级操作 | 零拷贝列式存储、SIMD/Numba/Polars |
| 扩展性 | 插拔式算法、流式或分布式 | Pipeline + Lazy 计算图 |
| 易部署 | `pip install dasmatrix` | 纯 Python + 可选 C/CUDA 扩展 |

## 3. 专为DAS优化的核心算子与语法设计

### 3.1 DAS核心信号处理算子

| 操作族 | 常用算子 | 典型参数 | 结果维度 | 典型用途 |
|---|---|---|---|---|
| 时域预处理 | `detrend`、`normalize`、`demean`、`rms` | window、axis | 与输入一致 | 消除直流/幅值归一 |
| 滤波 | `lowpass`、`highpass`、`bandpass`、`notch`、`median`、`adaptive` | freq/Hz、order、window | 与输入一致 | 降噪、去趋势 |
| 衍生/包络 | `diff`、`cumsum`、`envelope`、`hilbert_env` | axis、order | 与输入一致 | 边缘检测、相移 |
| 变换 | `fft`、`ifft`、`wavelet`、`cwt`、`hilbert` | nfft、scale、wavelet | freq/domain × channel | 频域分析 |
| 时–频 | `stft`、`istft`、`spectrogram` | nperseg、noverlap | time × freq × ch | 事件定位/分类 |
| 统计聚合 | `stack`、`mean`、`max`、`std` | win_len、axis | 与输入或降维 | 提升信噪比 |
| 空域 | `spatial_smooth`、`beamforming` | kernel、velocity | time × ch 或 time | 波场成像 |
| 检测 | `threshold_detect`、`template_match`、`ml_detect` | threshold、model | events list | 事件粗检 |
| 可视化 | `plot_ts`、`plot_heatmap`、`plot_spec` | cmap、dB、ch_range | 图像 | 结果浏览 |

### 3.2 链式 DSL 语法

```python
from dasmatrix import df

D = df.read("fiber.dat")             # ➊ 读取
out = (D
       .detrend(axis='time')         # ➋ 去趋势
       .bandpass(low=10, high=200)   # ➌ 带通滤波
       .stft(nperseg=1024)           # ➍ 时频变换
       .threshold_detect(db=-30)     # ➎ 阈值检测
       .plot_spec(cmap="inferno"))   # ➏ 可视化
```

*语义解释*：每个算子返回新的"惰性" `DASFrame`；真正计算发生在 `plot_*` / `to_file()` / `collect()` 之类 **sink** 操作时。

### 3.3 迷你脚本式 DQL

对于批处理或CLI，可使用极简脚本：

```dql
READ fiber.dat
DETREND AXIS=time
BANDPASS 10 200
STFT 1024
THRESHOLD_DETECT -30
PLOT_SPEC CMAP=inferno
```

引擎逐行解析生成同一计算图，与Python DSL互通。

### 3.4 核心 API 草图 (`DASFrame`)

```python
class DASFrame:
    # --- 基础操作 ---
    def __init__(self, arrow_table, graph=None): ...
    def slice(self, t=slice(None), x=slice(None)): ...  # 时间和通道切片

    # --- 时域 ---
    def detrend(self, axis='time'): ...
    def normalize(self, method="zscore"): ...
    def bandpass(self, low, high, order=4, fs=None, design='butter'): ...
    def lowpass(self, cutoff, order=4, fs=None): ...
    def highpass(self, cutoff, order=4, fs=None): ...
    def notch(self, freq, Q=30, fs=None): ...
    def median_filter(self, k=5, axis='time'): ...
    def demean(self, axis='time'): ...

    # --- 变换 ---
    def fft(self, n=None, axis='time'): ...
    def ifft(self, axis='time'): ...
    def wavelet(self, wavelet='morl', scales=None): ...
    def cwt(self, wavelet='morl', scales=None): ...
    def stft(self, nperseg, noverlap=None, window='hann'): ...
    def hilbert(self, axis='time'): ...

    # --- 衍生 ---
    def envelope(self, method='hilbert', axis='time'): ...
    def diff(self, order=1, axis='time'): ...
    def cumsum(self, axis='time'): ...
    def hilbert_env(self, axis='time'): ...

    # --- 空域 ---
    def spatial_smooth(self, kernel=5): ...
    def beamforming(self, velocity, angle=None): ...

    # --- 统计 ---
    def rms(self, window, axis='time'): ...
    def stack(self, window, axis='time'): ...
    def mean(self, axis=None): ...
    def max(self, axis=None): ...
    def min(self, axis=None): ...
    def std(self, axis=None): ...

    # --- 检测 ---
    def threshold_detect(self, threshold=None, db=None): ...
    def template_match(self, template): ...
    def ml_detect(self, model, **kwargs): ...

    # --- Sink (触发计算) ---
    def plot_ts(self, ch=None): ...
    def plot_heatmap(self, t_range=None, ch_range=None): ...
    def plot_spec(self, cmap='viridis'): ...
    def to_h5(self, path): ...
    def to_parquet(self, path): ...
    def collect(self): ...  # 返回具体数据
    def run_forever(self): ... # 用于流处理
```

所有高阶函数都接受 `lazy=True/False` 参数，默认为 `True`（惰性），允许用户显式触发计算。

### 3.5 实时流处理示例

```python
stream = df.stream("tcp://0.0.0.0:9000", chunk=2048)

(stream
 .highpass(20)
 .envelope()
 .rms(window=256)
 .threshold_detect(db=-25)
 .sink(lambda events: alert(events)) # 自定义处理逻辑
 .run_forever())      # 启动非阻塞流处理
```

### 3.6 设计总结

1. 明确 **信号处理** 领域核心算子，封装为 `DASFrame` 原生方法；
2. 链式 Lazy 语法保证"一行完成业务"，避免显式的 `mutate/agg` 调用；
3. 结合 Arrow/Polars 列式存储 + Numba/CUDA，确保 TB 级数据高效；
4. `plot_*` / `to_*` / `collect` / `sink` 等方法打断惰性，是计算图真正落地执行点；
5. Python DSL 与文本 DQL 共用同一后端，易于脚本化与 GUI 集成。

## 4. 改进后的架构

```text
DASMatrix/
│
├── src/                           # 源代码目录
│   ├── __init__.py                # 包初始化文件
│   │
│   ├── api/                       # 高级API层
│   │   ├── __init__.py
│   │   ├── dasframe.py            # DASFrame核心对象（新增）
│   │   ├── dsl.py                 # 迷你DSL解释器（新增）
│   │   ├── df.py                  # df函数式API入口（新增）
│   │   ├── q.py                   # q快捷查询API（新增）
│   │   ├── easy.py                # 简易API入口
│   │   ├── pipeline_builder.py    # 流水线构建器 
│   │   └── batch_processor.py     # 批处理API
│   │
│   ├── core/                      # 核心模块
│   │   ├── __init__.py
│   │   ├── engine.py              # 处理引擎
│   │   ├── worker.py              # 工作线程
│   │   ├── scheduler.py           # 任务调度器
│   │   ├── performance_monitor.py # 性能监控器
│   │   ├── computation_graph.py   # 计算图实现（新增）
│   │   ├── graph_nodes.py         # 计算图节点类型（新增）
│   │   ├── backend_registry.py    # 计算后端注册表（新增）
│   │   └── distributed/           # 分布式基础设施
│   │       ├── __init__.py
│   │       ├── interface.py       # 分布式接口
│   │       ├── node_manager.py    # 节点管理器
│   │       ├── task_dispatcher.py # 任务分发器
│   │       └── ray_backend.py     # Ray分布式后端（新增）
│   │
│   ├── config/                    # 配置层
│   │   ├── __init__.py
│   │   ├── sampling_config.py     # 采样配置
│   │   ├── processing_config.py   # 处理配置
│   │   ├── analysis_config.py     # 分析配置
│   │   ├── visualization_config.py # 可视化配置
│   │   ├── storage_config.py      # 存储配置
│   │   ├── das_config.py          # 总配置类
│   │   ├── config_manager.py      # 配置管理器
│   │   └── auto_config.py         # 自动配置
│   │
│   ├── acquisition/               # 数据采集层
│   │   ├── __init__.py
│   │   ├── interfaces/            # 接口定义
│   │   │   ├── __init__.py
│   │   │   └── data_reader.py     # 数据读取器接口
│   │   ├── readers/               # 具体读取器实现
│   │   │   ├── __init__.py
│   │   │   ├── dat_reader.py      # DAT文件读取器
│   │   │   ├── h5_reader.py       # HDF5文件读取器
│   │   │   ├── parquet_reader.py  # Parquet文件读取器（新增）
│   │   │   └── stream_reader.py   # 实时流数据读取器
│   │   ├── sources/               # 数据源实现
│   │   │   ├── __init__.py
│   │   │   ├── file_source.py     # 文件数据源
│   │   │   ├── tcp_source.py      # TCP数据源
│   │   │   └── udp_source.py      # UDP数据源
│   │   ├── das_reader.py          # 统一数据读取接口
│   │   ├── chunked_reader.py      # 分块读取器
│   │   └── arrow_adapter.py       # Arrow/Polars适配器（新增）
│   │
│   ├── processing/                # 数据处理层
│   │   ├── __init__.py
│   │   ├── interfaces/            # 接口定义
│   │   │   ├── __init__.py
│   │   │   ├── filter.py          # 滤波器接口
│   │   │   ├── backend.py         # 计算后端接口（新增）
│   │   │   └── feature_extractor.py # 特征提取器接口
│   │   ├── filters/               # 滤波器实现
│   │   │   ├── __init__.py
│   │   │   ├── high_pass_filter.py # 高通滤波器
│   │   │   ├── low_pass_filter.py  # 低通滤波器
│   │   │   ├── band_pass_filter.py # 带通滤波器
│   │   │   ├── median_filter.py    # 中值滤波器
│   │   │   └── adaptive_filter.py  # 自适应滤波器
│   │   ├── transforms/            # 信号变换
│   │   │   ├── __init__.py
│   │   │   ├── fft_transform.py    # 快速傅里叶变换
│   │   │   ├── wavelet_transform.py # 小波变换
│   │   │   └── hilbert_transform.py # 希尔伯特变换
│   │   ├── feature_extractors/    # 特征提取器
│   │   │   ├── __init__.py
│   │   │   ├── statistical_features.py # 统计特征
│   │   │   ├── spectral_features.py # 频谱特征
│   │   │   └── wavelet_features.py # 小波特征
│   │   ├── processors/            # 数据处理器
│   │   │   ├── __init__.py
│   │   │   ├── base_processor.py  # 基础处理器
│   │   │   └── enhanced_processor.py # 增强处理器
│   │   ├── backends/              # 计算后端实现（新增）
│   │   │   ├── __init__.py
│   │   │   ├── python_backend.py  # 纯Python后端
│   │   │   ├── polars_backend.py  # Polars后端
│   │   │   ├── numba_backend.py   # Numba JIT后端
│   │   │   └── cuda_backend.py    # CUDA GPU后端
│   │   ├── optimized/             # 优化实现
│   │   │   ├── __init__.py
│   │   │   ├── numba_filters.py   # Numba优化滤波器 
│   │   │   └── vectorized_ops.py  # 向量化操作
│   │   ├── pipeline.py            # 处理管道
│   │   ├── parallel_processor.py  # 并行处理器
│   │   ├── lazy_processor.py      # 延迟计算处理器
│   │   └── stream_processor.py    # 流处理器
│   │
│   ├── analysis/                  # 分析层
│   │   ├── __init__.py
│   │   ├── interfaces/            # 接口定义
│   │   │   ├── __init__.py
│   │   │   └── analyzer.py        # 分析器接口
│   │   ├── detectors/             # 事件检测器
│   │   │   ├── __init__.py
│   │   │   ├── threshold_detector.py # 阈值检测器
│   │   │   ├── pattern_detector.py # 模式检测器
│   │   │   └── ml_detector.py     # 机器学习检测器
│   │   ├── classifiers/           # 事件分类器
│   │   │   ├── __init__.py
│   │   │   ├── svm_classifier.py  # SVM分类器
│   │   │   ├── rf_classifier.py   # 随机森林分类器
│   │   │   └── nn_classifier.py   # 神经网络分类器
│   │   ├── frequency/             # 频率分析
│   │   │   ├── __init__.py
│   │   │   ├── spectrum_analyzer.py # 频谱分析器
│   │   │   └── spectrogram_analyzer.py # 时频谱分析器
│   │   └── statistics/            # 统计分析
│   │       ├── __init__.py
│   │       ├── descriptive_stats.py # 描述性统计
│   │       └── correlation_analyzer.py # 相关性分析
│   │
│   ├── visualization/             # 可视化层
│   │   ├── __init__.py
│   │   ├── interfaces/            # 接口定义
│   │   │   ├── __init__.py
│   │   │   └── visualizer.py      # 可视化器接口
│   │   ├── plotters/              # 绘图器
│   │   │   ├── __init__.py
│   │   │   ├── time_series_plotter.py # 时间序列绘图器
│   │   │   ├── spectrogram_plotter.py # 时频谱绘图器
│   │   │   ├── heatmap_plotter.py # 热图绘图器
│   │   │   └── event_plotter.py   # 事件绘图器
│   │   ├── dashboards/            # 仪表板
│   │   │   ├── __init__.py
│   │   │   ├── monitoring_dashboard.py # 监控仪表板
│   │   │   └── analysis_dashboard.py # 分析仪表板
│   │   ├── exporters/             # 导出器
│   │   │   ├── __init__.py
│   │   │   ├── image_exporter.py  # 图像导出器
│   │   │   └── report_exporter.py # 报告导出器
│   │   ├── interactive/          # 交互式可视化
│   │   │   ├── __init__.py
│   │   │   └── plotly_plots.py   # Plotly交互式图表
│   │   └── das_visualizer.py      # 统一可视化接口
│   │
│   ├── storage/                   # 存储层
│   │   ├── __init__.py
│   │   ├── interfaces/            # 接口定义
│   │   │   ├── __init__.py
│   │   │   └── storage_backend.py # 存储后端接口
│   │   ├── backends/              # 存储后端实现
│   │   │   ├── __init__.py
│   │   │   ├── file_backend.py    # 文件存储后端
│   │   │   ├── hdf5_backend.py    # HDF5存储后端
│   │   │   ├── parquet_backend.py # Parquet存储后端（新增）
│   │   │   └── db_backend.py      # 数据库存储后端
│   │   ├── memory_management/     # 内存管理
│   │   │   ├── __init__.py
│   │   │   ├── memory_mapped.py   # 内存映射工具
│   │   │   ├── cache_manager.py   # 缓存管理器
│   │   │   └── auto_spill.py      # 自动溢出管理（新增）
│   │   ├── result_storage.py      # 结果存储管理器
│   │   └── metadata_manager.py    # 元数据管理器
│   │
│   ├── applications/              # 应用层
│   │   ├── __init__.py
│   │   ├── base_application.py    # 应用基类
│   │   ├── pipeline_monitoring/   # 管道监测应用
│   │   │   ├── __init__.py
│   │   │   ├── leak_detector.py   # 泄漏检测器
│   │   │   └── flow_analyzer.py   # 流量分析器
│   │   ├── perimeter_security/    # 周界安全应用
│   │   │   ├── __init__.py
│   │   │   ├── intrusion_detector.py # 入侵检测器
│   │   │   └── activity_classifier.py # 活动分类器
│   │   ├── seismic_monitoring/    # 地震监测应用
│   │   │   ├── __init__.py
│   │   │   ├── earthquake_detector.py # 地震检测器
│   │   │   └── tremor_analyzer.py # 震动分析器
│   │   └── traffic_monitoring/    # 交通监测应用
│   │       ├── __init__.py
│   │       ├── vehicle_detector.py # 车辆检测器
│   │       └── traffic_analyzer.py # 交通分析器
│   │
│   ├── common/                    # 公共组件
│   │   ├── __init__.py
│   │   ├── types.py               # 类型定义
│   │   ├── constants.py           # 常量定义
│   │   ├── exceptions.py          # 异常定义
│   │   ├── enums.py               # 枚举定义
│   │   └── data_structures.py     # 优化数据结构
│   │
│   └── utils/                     # 工具函数
│       ├── __init__.py
│       ├── logger.py              # 日志工具
│       ├── profiler.py            # 性能分析工具
│       ├── validators.py          # 验证工具
│       ├── converters.py          # 转换工具
│       ├── math_utils.py          # 数学工具
│       ├── io_utils.py            # 输入输出工具
│       └── performance_utils.py   # 性能优化工具
│
├── tests/                         # 测试目录
│   ├── __init__.py
│   ├── unit/                      # 单元测试
│   │   ├── __init__.py
│   │   ├── test_config/           # 配置测试
│   │   ├── test_acquisition/      # 数据采集测试
│   │   ├── test_processing/       # 数据处理测试
│   │   ├── test_analysis/         # 分析测试
│   │   ├── test_visualization/    # 可视化测试
│   │   ├── test_storage/          # 存储测试
│   │   ├── test_applications/     # 应用测试
│   │   └── test_api/              # API测试
│   │
│   ├── integration/               # 集成测试
│   │   ├── __init__.py
│   │   ├── test_pipeline.py       # 管道测试
│   │   ├── test_processing_chain.py # 处理链测试
│   │   └── test_full_flow.py      # 全流程测试
│   │
│   ├── performance/               # 性能测试
│   │   ├── __init__.py
│   │   ├── test_reader_perf.py    # 读取器性能测试
│   │   ├── test_filter_perf.py    # 滤波器性能测试
│   │   ├── test_parallelism.py    # 并行性能测试
│   │   ├── test_memory_usage.py   # 内存使用测试
│   │   ├── test_backend_compare.py # 不同后端性能对比（新增）
│   │   └── test_scaling.py        # 扩展性测试（新增）
│   │
│   └── data/                      # 测试数据
│       ├── raw/                   # 原始测试数据
│       ├── processed/             # 处理后的测试数据
│       └── expected/              # 预期结果数据
│
├── configs/                       # 配置文件目录
│   ├── default.yaml               # 默认配置
│   ├── logging.yaml               # 日志配置
│   ├── sampling/                  # 采样配置
│   │   ├── high_frequency.yaml    # 高频采样配置
│   │   └── low_frequency.yaml     # 低频采样配置
│   ├── processing/                # 处理配置
│   │   ├── noise_reduction.yaml   # 降噪配置
│   │   └── feature_extraction.yaml # 特征提取配置
│   ├── applications/              # 应用配置
│   │   ├── pipeline_monitoring.yaml # 管道监测配置
│   │   ├── perimeter_security.yaml # 周界安全配置
│   │   └── seismic_monitoring.yaml # 地震监测配置
│   └── optimized/                 # 性能优化配置
│       ├── small_data.yaml        # 小数据集优化
│       └── large_data.yaml        # 大数据集优化
│
├── examples/                      # 示例目录
│   ├── __init__.py
│   ├── tutorials/                 # 教程
│   │   ├── __init__.py
│   │   ├── quick_start.py         # 快速入门
│   │   ├── working_with_data.py   # 数据处理教程
│   │   ├── dsl_syntax.py          # DSL语法教程（新增）
│   │   └── custom_pipeline.py     # 自定义管道教程
│   ├── basic/                     # 基础示例
│   │   ├── read_data.py           # 数据读取示例
│   │   ├── basic_processing.py    # 基础处理示例
│   │   └── simple_visualization.py # 简单可视化示例
│   ├── advanced/                  # 高级示例
│   │   ├── custom_pipeline.py     # 自定义管道示例
│   │   ├── parallel_processing.py # 并行处理示例
│   │   ├── custom_detector.py     # 自定义检测器示例
│   │   ├── memory_optimization.py # 内存优化示例
│   │   └── backend_selection.py   # 后端选择示例（新增）
│   └── applications/              # 应用示例
│       ├── pipeline_leak_detection.py # 管道泄漏检测示例
│       ├── perimeter_intrusion.py    # 周界入侵检测示例
│       └── earthquake_monitoring.py  # 地震监测示例
│
├── docs/                          # 文档目录
│   ├── api/                       # API文档
│   │   ├── config.md              # 配置API
│   │   ├── acquisition.md         # 数据采集API
│   │   ├── processing.md          # 数据处理API
│   │   ├── analysis.md            # 分析API
│   │   ├── visualization.md       # 可视化API
│   │   ├── storage.md             # 存储API
│   │   ├── applications.md        # 应用API
│   │   ├── high_level_api.md      # 高级API文档
│   │   └── dasframe_dsl.md        # DASFrame与DSL文档（新增）
│   ├── user_guide/                # 用户指南
│   │   ├── getting_started.md     # 入门指南
│   │   ├── installation.md        # 安装指南
│   │   ├── configuration.md       # 配置指南
│   │   ├── processing_pipeline.md # 处理管道指南
│   │   ├── custom_extensions.md   # 自定义扩展指南
│   │   ├── performance_tips.md    # 性能优化技巧
│   │   └── datatable_comparison.md # 与data.table对比（新增）
│   ├── developer_guide/           # 开发者指南
│   │   ├── architecture.md        # 架构概述
│   │   ├── contributing.md        # 贡献指南
│   │   ├── coding_standards.md    # 编码标准
│   │   ├── testing.md             # 测试指南
│   │   ├── optimization_guide.md  # 优化指南
│   │   └── backend_development.md # 后端开发指南（新增）
│   ├── examples/                  # 示例文档
│   │   ├── basic_examples.md      # 基础示例文档
│   │   └── advanced_examples.md   # 高级示例文档
│   ├── performance/               # 性能文档
│   │   ├── optimization_guide.md  # 优化指南
│   │   ├── benchmarks.md          # 性能基准
│   │   ├── memory_management.md   # 内存管理指南
│   │   └── scaling_strategies.md  # 扩展策略指南（新增）
│   └── tutorials/                 # 教程文档
│       ├── quickstart.md          # 快速入门教程
│       ├── common_tasks.md        # 常见任务教程
│       └── dsl_guide.md           # DSL语法指南（新增）
│
├── tools/                         # 工具和脚本目录
│   ├── setup/                     # 安装脚本
│   │   ├── install_dependencies.sh # 依赖安装脚本
│   │   └── setup_environment.py   # 环境设置脚本
│   ├── dev/                       # 开发工具
│   │   ├── generate_stubs.py      # 生成存根文件
│   │   ├── code_formatter.py      # 代码格式化工具
│   │   └── doc_generator.py       # 文档生成工具
│   └── analysis/                  # 分析工具
│       ├── data_converter.py      # 数据转换工具
│       ├── benchmark_runner.py    # 基准测试运行器
│       └── profile_analyzer.py    # 性能分析器
│
├── build/                         # 构建和部署配置
│   ├── docker/                    # Docker配置
│   │   ├── Dockerfile             # 主Dockerfile
│   │   ├── docker-compose.yml     # Docker Compose配置
│   │   └── .dockerignore          # Docker忽略文件
│   ├── ci/                        # CI/CD配置
│   │   ├── .github/               # GitHub Actions配置
│   │   └── jenkins/               # Jenkins配置
│   └── deployment/                # 部署配置
│       ├── kubernetes/            # Kubernetes配置
│       └── cloud/                 # 云部署配置
│
├── .gitignore                     # Git忽略文件
├── README.md                      # 项目说明文档
├── CHANGELOG.md                   # 变更日志
├── LICENSE                        # 许可证文件
├── setup.py                       # 安装脚本
├── requirements.txt               # 依赖项列表
└── pyproject.toml                 # Python项目配置
```

## 5. 新增和改进的关键部分

### 5.1 DASFrame核心列式对象

新增`api/dasframe.py`模块，提供类似data.table的列式对象抽象：

- **零拷贝列存储**：基于Arrow/Polars实现高效内存表示
- **链式操作API**：filter/mutate/agg/slice等基本操作，每次返回新DASFrame
- **DSL语法支持**：支持表达式评估和字符串查询解析
- **自动内存管理**：基于数据规模自动选择内存/mmap/分块策略

### 5.2 计算图与多后端架构

- **计算图抽象**：所有DASFrame操作构建Lazy计算图
- **多后端注册系统**：支持Python-Native/Polars/Numba/CUDA多种后端
- **后端自动选择**：根据操作类型和数据规模自动选择最优后端
- **算子注册机制**：允许第三方扩展注册自定义算子和后端

### 5.3 高性能引擎与优化

- **Polars/Arrow集成**：利用现代列存引擎和SIMD指令集加速
- **Numba JIT编译**：对性能关键路径应用即时编译
- **内存分级管理**：<2GB内存处理、>2GB自动mmap、>20GB分块+外排
- **向量化与并行化**：自动向量化和线程级并行处理

### 5.4 高级DSL与快捷入口

- **df函数式API**：类似Pandas的函数式操作接口
- **q迷你DSL**：快捷字符串查询解析器，支持.dql脚本
- **语法解析器**：将字符串表达式转换为高效计算节点

### 5.5 分布式计算支持

- **Ray Dataset集成**：支持分布式数据处理
- **Polars Streaming**：支持流式大数据处理
- **分片与调度策略**：自动数据分片和任务调度

## 6. 性能基线要求与实现

### 6.1 性能目标

| 场景 | 数据规模 | 目标 | 技术路标 |
|---|---|---|---|
| 单机内存 | 128 M samples × 1024 ch ≈ 1 TB | 滤波 + FFT < 10 s | Polars Lazy → ThreadPool → SIMD |
| 流式实时 | 50 kSps × 2 k ch | 延迟 ≤ 100 ms | mmap ring buffer + Numba JIT |
| 分布式 | 10 TB 历史回放 | 1 h 内完成全链 | Ray Dataset + Parquet 分片 |

### 6.2 关键算子实现策略

| 算子类型 | 技术实现 |
|---|---|
| FIR/IIR 滤波 | SciPy SOS → Numba vectorized → FFT-convolution（大窗口）|
| FFT/STFT | 1) `pyFFTW` 2) `cupy.fft` 3) Polars `expr.fft()`（批量列）|
| Wavelet | `pywt` + 线程池 / GPU optional |
| Beamforming | Numba JIT + SIMD；>1 TB 场景切分 + Ray Map |
| 实时处理 | ring-buffer + lock-free 写，消费端基于 `stream_processor` |

## 7. 实现策略与里程碑

### 7.1 渐进式开发路线

- **M0** (1个月)：DASFrame + 读写 + filter/mutate/agg - 单机Arrow
- **M1** (2个月)：Lazy计算图 + Polars后端 - 性能≥5× pandas
- **M2** (3个月)：FFT/Wavelet/检测算子 + 可视化Heatmap - 与现有pipeline对齐
- **M3** (4-5月)：流式 / mmap / GPU支持 - 实时能力
- **M4** (6月)：分布式prototype (Ray) - 水平扩展

### 7.2 性能优先级策略

- 识别性能瓶颈组件，优先优化数据读取和处理部分
- 使用内存映射和延迟计算降低内存占用
- 向量化操作和并行处理提升计算速度

### 7.3 接口一致性策略

- 保持各层接口风格一致
- 高级API复用底层组件，确保行为一致性
- 用户引导路径：DSL查询 → DASFrame信号处理API → 流水线构建 → 高级定制

### 7.4 测试与评估策略

#### 7.4.1 测试矩阵

| 分类 | 数据集 | 断言 |
|---|---|---|
| Accuracy | 合成正弦 + 已知 FFT | 峰值位置误差 < 1 bin |
| Throughput | 1 GB 文件 + `bandpass` | > 100 MB/s 单核 |
| Latency | 2 k ch × 50 kSps 在线 | end-to-end < 80 ms |

#### 7.4.2 评估方法

- 添加基准测试比较不同后端性能
- 持续监控内存使用情况
- 不同规模数据集的扩展性测试

## 8. 文档与示例规范

- README顶部放置5行极简示例（读→筛→绘→保存）
- `examples/tutorials/quick_start.py` 使用 `df` 与 `q` 两套写法对照
- `docs/high_level_api.md` 用 "data.table vs DASFrame" 对比表
- `docs/dasframe_dsl.md` 详细解释核心算子与用法

这一架构保留了原始设计的所有优点，同时通过DASFrame列式对象、多后端计算图、分级内存管理和DSL语法增强了易用性和性能，为用户提供了类似data.table的极简体验，并为未来的分布式扩展做好了准备。
