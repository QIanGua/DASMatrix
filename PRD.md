# 项目需求文档

设计一个完整的高性能分布式光纤声波传感(DAS)数据处理框架库，采用正交分解架构，确保各层职责明确、耦合度低，同时兼顾高性能和可扩展性。目标是创建一个"像 data.table 一样简单而强大"的 Python 包，专用于处理2D时空DAS数据（time × channel）。

基于 Python 进行开发，使用 uv 管理依赖。核心数据结构 `DASFrame` 基于 **Xarray + Dask** 实现延迟计算与分块处理，Polars 用于元数据管理，辅以 Numba/GPU 加速关键算子。文档使用 MkDocs + mkdocstrings 自动生成。采用模块化设计以支持良好的可维护性和可扩展性。

## 1. 框架设计核心理念

1. **性能优先 (Performance First)**：不仅仅是"快"，而是追求**硬件极限**。通过零拷贝 (Zero-Copy) 和算子融合 (Kernel Fusion) 最小化内存带宽占用。
2. **张量原生 (Tensor-Native)**：认识到 DAS 数据本质是时空矩阵 (Tensor)，而非简单的表格。核心计算采用 SIMD/GPU 加速的矩阵运算。
3. **极简易用 (Simplicity)**：保持 `data.table` 般的极简 DSL，但在底层自动将 Python 代码编译为高性能机器码。
4. **混合引擎 (Hybrid Engine)**：元数据用 Polars 管理，信号矩阵用 Numba/CuPy 处理，取长补短。
5. **渐进式复杂度**：从单机内存处理无缝扩展到 Zarr/HDF5 核外计算 (Out-of-Core) 和分布式集群。

## 2. 核心诉求与技术落点

| 诉求 | 解释 | 技术落点 |
|---|---|---|
| 极简语法 | 一行链式表达完成筛选/变换/聚合 | DASFrame 对象 + Lazy DSL |
| **极致性能** | 饱和内存带宽，利用多核/GPU | **Polars (Meta) + Numba/CuPy (Signal) + Kernel Fusion** |
| 存储优化 | TB 级数据的秒级访问与切片 | **Zarr (Chunked Store) + Parquet (Metadata)** |
| 扩展性 | 插拔式算法、流式或分布式 | Execution Planner + JIT Compilation |
| 易部署 | `pip install dasmatrix` | 核心纯 Python，可选 CUDA 扩展包 |

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

## 4. 当前项目结构

> **注**: ✅ 已实现 | 🚧 部分实现 | 📋 计划中

```text
DASMatrix/                         # 项目根目录
│
├── DASMatrix/                     # 源代码目录 ✅
│   ├── __init__.py
│   │
│   ├── api/                       # 高级API层 🚧
│   │   ├── __init__.py
│   │   ├── dasframe.py            # DASFrame核心对象 ✅ (Xarray/Dask后端)
│   │   └── df.py                  # df函数式API入口 ✅
│   │   # 📋 计划: dsl.py, q.py, easy.py, pipeline_builder.py
│   │
│   ├── core/                      # 核心模块 🚧
│   │   ├── __init__.py
│   │   └── computation_graph.py   # 计算图实现 ✅
│   │   # 📋 计划: engine.py, scheduler.py, distributed/
│   │
│   ├── config/                    # 配置层 🚧
│   │   ├── __init__.py
│   │   ├── sampling_config.py     # 采样配置 ✅
│   │   └── visualization_config.py # 可视化配置 ✅ (Nature/Science风格)
│   │   # 📋 计划: processing_config.py, storage_config.py
│   │
│   ├── acquisition/               # 数据采集层 🚧
│   │   ├── __init__.py
│   │   └── das_reader.py          # 统一数据读取接口 ✅ (支持DAT/H5/Zarr)
│   │   # 📋 计划: stream_reader.py, sources/
│   │
│   ├── processing/                # 数据处理层 ✅
│   │   ├── __init__.py
│   │   ├── das_processor.py       # DAS信号处理器 ✅
│   │   ├── engine.py              # 混合执行引擎 ✅
│   │   ├── planner/               # 执行计划器 ✅
│   │   │   ├── __init__.py
│   │   │   └── optimizer.py       # 计算图优化器 ✅
│   │   ├── kernels/               # 低级计算内核 🚧
│   │   │   ├── cpu/               # Numba/SIMD 内核
│   │   │   └── gpu/               # CUDA/CuPy 内核
│   │   └── backends/              # 计算后端实现 ✅
│   │       ├── __init__.py
│   │       ├── numba_backend.py   # 高性能CPU后端 ✅
│   │       └── polars_backend.py  # Polars后端 ✅
│   │
│   └── visualization/             # 可视化层 ✅
│       ├── __init__.py
│       ├── das_visualizer.py      # 统一可视化接口 ✅ (含多种图表类)
│       └── styles.py              # Nature/Science出版级样式 ✅
│
├── tests/                         # 测试目录 ✅
│   ├── unit/                      # 单元测试 ✅
│   │   ├── test_dasframe.py
│   │   ├── test_computation_graph.py
│   │   ├── test_hybrid_engine.py
│   │   ├── test_visualization.py
│   │   └── ...
│   └── performance/               # 性能测试 🚧
│
├── examples/                      # 示例目录 ✅
│   ├── quick_start.py             # 快速入门 ✅
│   ├── visualization_demo.py      # 可视化演示 ✅
│   ├── fk_filter_demo.py          # F-K滤波演示 ✅
│   ├── realtime_monitoring.py     # 实时监控示例 ✅
│   └── benchmark_engine.py        # 性能基准测试 ✅
│
├── docs/                          # 文档目录 ✅ (MkDocs + mkdocstrings)
│   ├── api/                       # API文档 (自动生成)
│   ├── index.md                   # 首页
│   ├── quickstart.md              # 快速入门
│   └── contributing.md            # 贡献指南
│
├── nb/                            # Jupyter笔记本 ✅
│   └── DASMatrix_Tutorial.ipynb   # 教程笔记本 ✅
│
├── scripts/                       # 脚本目录 ✅
│
├── .github/                       # GitHub配置 ✅
│   └── workflows/                 # CI/CD工作流 ✅
│
├── mkdocs.yml                     # MkDocs配置 ✅
├── justfile                       # Just命令配置 ✅
├── pyproject.toml                 # 项目配置 ✅
└── README.md                      # 项目说明 ✅
```

### 4.1 后续规划模块 (📋 未实现)

| 模块 | 用途 | 优先级 |
|---|---|---|
| `analysis/` | 事件检测器、分类器、频谱分析 | 高 |
| `storage/` | Zarr/Parquet存储后端、内存管理 | 中 |
| `applications/` | 行业应用（管道监测、地震监测等） | 低 |
| `common/` | 类型定义、异常、常量 | 中 |
| `utils/` | 日志、性能分析、验证工具 | 中 |
| `core/distributed/` | Ray分布式后端 | 低 |

## 5. 已实现的关键技术

### 5.1 Xarray/Dask 延迟计算引擎

`DASFrame` 核心数据结构基于 **Xarray + Dask** 实现：

- **xr.DataArray**: 维度标签（time, channel）+ 元数据管理
- **Dask Array**: 分块延迟计算，自动并行化
- **链式 API**: `.bandpass().detrend().normalize()` 链式调用，构建惰性计算图
- **Sink 操作**: `.collect()`, `.plot_*()` 触发实际计算

### 5.2 混合计算后端

- **Numba Backend**: CPU 高性能计算，JIT 编译加速
- **Polars Backend**: 元数据查询与聚合
- **未来扩展**: CuPy/GPU 后端（计划中）

### 5.3 JIT Compilation & Kernel Fusion (JIT 编译与算子融合)

为了避免 Python 循环和 NumPy 产生的大量中间内存分配（Memory Traffic 是高性能计算的最大瓶颈），我们引入 **Execution Planner**：

1. **Lazy Graph**: 所有 DSL 操作（如 `.detrend().filter().abs()`）首先构建逻辑图。
2. **Fusion**: 编译器识别出可以融合的操作序列。例如 `detrend + filter + abs` 可以被融合为一个 Loop。
3. **Codegen**: 使用 **Numba** 将融合后的 Loop 动态编译为机器码 (LLVM)，一次遍历内存完成所有计算。
4. **Parallel**: 自动将编译后的 Kernel 调度到多核 CPU 或 GPU 上执行。

### 5.4 Zarr + Parquet 存储架构 (规划中)

- **Zarr**: 用于存储主要的 Signal Matrix。支持 N 维分块 (Chunking) 和多种压缩算法 (Blosc, LZ4)。支持并发读写，非常适合 DAS 这种 Time x Channel 的大矩阵。
- **Parquet**: 用于存储 Metadata Frame。列式存储，查询极快。

### 5.5 分布式与核外计算 (规划中)

对于超大内存的数据集，引擎自动切换到 **Chunked Mode**：

- 利用 `dask` 或自定义调度器，按 Chunk 流式处理 Zarr 数据。
- 保证内存占用恒定，不随数据量增加而 OOM。

## 6. 性能基线与优化策略

### 6.1 性能目标 (硬件级极限)

| 场景 | 数据规模 | 目标 | 衡量标准 |
|---|---|---|---|
| **带宽饱和** | 内存处理 | **> 20 GB/s** (DDR4/5 带宽的 60%+) | Kernel Fusion 减少 90% 内存读写 |
| **实时延迟** | 50 kSps × 2 k ch | < 50 ms (End-to-End) | 零重分配 (Zero-Allocation) Ring Buffer |
| **交互式可视** | 1 TB 数据集 | < 1 s (Heatmap Render) | 多级分辨率索引 (Multi-scale Indexing) |

### 6.2 关键性能优化技术

#### 6.2.1 算子融合 (Operator Fusion)

传统 NumPy 写法：

```python
x = x - np.mean(x)  # Allocation 1
x = signal.filtfilt(b, a, x) # Allocation 2 (huge)
x = np.abs(x) # Allocation 3
```

DASMatrix Numba 融合写法 (自动生成)：

```python
@numba.jit(nopython=True, parallel=True)
def fused_kernel(x, out):
    # 单次循环完成所有操作，寄存器内计算，无中间内存写入
    for i in prange(x.shape[0]):
        val = x[i] - mean
        val = filter_step(val)
        out[i] = abs(val) 
```

#### 6.2.2 显式 SIMD 指令

利用 Numba 的 `vectorize` 和 `guvectorize`，显式通过 LLVM 使用 AVX-2 / AVX-512 指令集处理 FFT 和滤波。

#### 6.2.3 零拷贝视图

在切片、重塑等操作中，严格保证不复制数据，仅传递指针和步长 (Strides)。

#### 6.2.4 GPU 加速 (CuPy)

对于 FFT、2D 卷积等计算密集型任务，自动调度数据到 GPU 显存（如果可用），利用 CUDA Core 的海量并行能力。

## 7. 实现策略与里程碑

### 7.1 开发进度 (截至 2024-12)

- **M0** ✅ DASFrame (Xarray/Dask后端) + das_reader + 链式滤波
- **M1** ✅ 计算图 + Numba/Polars后端 + 优化器
- **M2** ✅ FFT/STFT/F-K滤波 + Nature/Science级可视化
- **M3** 🚧 实时监控示例 (部分实现)
- **M4** 📋 分布式 Ray 后端 (计划中)

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

## 8. 文档与示例

### 8.1 已完成

- ✅ MkDocs + mkdocstrings 自动生成 API 文档
- ✅ `examples/quick_start.py` 快速入门示例
- ✅ `examples/visualization_demo.py` 可视化演示
- ✅ `examples/fk_filter_demo.py` F-K 滤波演示
- ✅ `nb/DASMatrix_Tutorial.ipynb` 完整教程笔记本
- ✅ GitHub Actions CI/CD 工作流

### 8.2 待完善

- 📋 详细的 DSL 语法文档
- 📋 性能基准测试报告

这一架构保留了原始设计的所有优点，同时通过DASFrame列式对象、多后端计算图、分级内存管理和DSL语法增强了易用性和性能，为用户提供了类似data.table的极简体验，并为未来的分布式扩展做好了准备。
