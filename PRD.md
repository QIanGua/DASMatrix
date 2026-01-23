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
│   ├── api/                       # 高级API层 ✅
│   │   ├── __init__.py
│   │   ├── dasframe.py            # DASFrame核心对象 ✅ (Xarray/Dask后端)
│   │   ├── df.py                  # df函数式API入口 ✅
│   │   └── spool.py               # DASSpool多文件管理 ✅
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
│   ├── acquisition/               # 数据采集层 ✅
│   │   ├── __init__.py
│   │   ├── das_reader.py          # 统一数据读取接口 ✅ (兼容旧版)
│   │   └── formats/               # 格式插件系统 ✅ (12种格式)
│   │       ├── base.py            # FormatPlugin协议
│   │       ├── h5.py              # HDF5格式
│   │       ├── dat.py             # DAT格式
│   │       ├── prodml.py          # PRODML v2.0/2.1
│   │       ├── silixa.py          # Silixa HDF5
│   │       ├── febus.py           # Febus Optics
│   │       ├── terra15.py         # Terra15
│   │       ├── apsensing.py       # AP Sensing
│   │       ├── zarr_format.py     # Zarr云原生
│   │       ├── netcdf.py          # NetCDF
│   │       ├── segy.py            # SEG-Y
│   │       ├── miniseed.py        # MiniSEED
│   │       └── tdms.py            # TDMS (可选)
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
│   ├── visualization/             # 可视化层 ✅
│   │   ├── __init__.py
│   │   ├── das_visualizer.py      # 统一可视化接口 ✅ (含多种图表类)
│   │   └── styles.py              # Nature/Science出版级样式 ✅
│   │
│   ├── units.py                   # 单位系统 ✅ (基于 Pint)
│   ├── examples.py                # 示例数据生成 ✅
│   └── utils/                     # 工具函数 ✅
│       ├── __init__.py
│       └── time.py                # 时间工具函数 ✅
│
├── tests/                         # 测试目录 ✅
│   ├── unit/                      # 单元测试 ✅
│   │   ├── test_dasframe.py
│   │   ├── test_computation_graph.py
│   │   ├── test_hybrid_engine.py
│   │   ├── test_visualization.py
│   │   ├── test_units.py          # 单位系统测试 ✅
│   │   ├── test_examples.py       # 示例数据测试 ✅
│   │   ├── test_time_utils.py     # 时间工具测试 ✅
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
- **M3** ✅ 实时监控与极致性能优化 (2026-01)
- **M4** 🚧 标准化元数据与生态集成 (进行中)

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

---

## 9. 竞品分析与差距评估

> 基于 2024-12 对主流开源 DAS 处理库的全面调研

### 9.1 主要竞品概览

| 特性 | DASMatrix | DASCore | Xdas | DASPy |
|------|-----------|---------|------|-------|
| **GitHub Stars** | 新项目 | 63★ | 38★ | 84★ |
| **核心后端** | Xarray/Dask | 自定义 Patch/Spool | Xarray-like | ObsPy + NumPy |
| **文件格式支持** | 12种 | 20+ 种 | 10+ 种 | 10+ 种 |
| **延迟计算** | ✅ Dask | ✅ 自定义 | ✅ 自定义 | ❌ 即时计算 |
| **多文件虚拟合并** | ✅ DASSpool | ✅ Spool | ✅ open_mfdataarray | ❌ |
| **分块处理 (OOC)** | ✅ Dask | ✅ 原生 | ✅ Atoms | ❌ |
| **GPU 支持** | 📋 计划 | 📋 计划 | ❌ | ❌ |
| **流处理** | 🚧 部分 | ❌ | ✅ | ❌ |
| **元数据管理** | 简单 | 完善 (Inventory) | 完善 | ObsPy 兼容 |
| **可视化** | ✅ Matplotlib | ✅ 基础 | ✅ 基础 | ✅ |
| **Web Dashboard** | ✅ FastAPI | ❌ | ❌ | ❌ |
| **文档质量** | 🚧 | ✅✅✅ (Quarto) | ✅✅ | ✅✅ |
| **测试覆盖** | ✅ 94个用例 | ✅ 99.5% | ✅ ~85% | ✅ ~80% |

### 9.2 竞品独特亮点

**DASCore (<https://github.com/DASDAE/dascore>)**

- 支持 20+ DAS 文件格式 (APSENSING, FEBUS, TERRA15, PRODML 等)
- `Spool` 抽象统一管理多文件数据集和文件归档
- 完善的元数据 `Inventory` 规划 (遵循 PRODML/DAS-RCN 标准)
- 与 ObsPy/Pandas/Xarray 双向互转
- 测试覆盖率 99.5%, 文档完善 (Quarto)

**Xdas (<https://github.com/xdas-dev/xdas>)**

- `open_mfdataarray` 多文件虚拟合并
- `Atoms` 有状态流水线处理框架, 支持分块处理时保持滤波器状态
- HDF5 压缩支持 (Zfp 等)
- 多线程信号处理函数

**DASPy (<https://github.com/HMZ-03/DASPy>)**

- 专业地震学算法 (去噪、波形分解、应变-速度转换)
- 学术论文支持 (SRL 2024)
- 完整中英文教程文档

### 9.3 DASMatrix 现有优势

1. **架构设计前瞻性**: 基于 Xarray + Dask 的现代架构, 混合执行引擎 + 计算图优化
2. **Web 实时监控**: FastAPI + WebSocket Dashboard (竞品均未实现)
3. **Nature/Science 级可视化**: 专业出版级样式配置
4. **PRD 文档完善**: 清晰的技术愿景和性能目标

### 9.4 关键差距 (需优先弥补)

| 差距项 | 当前状态 | 目标状态 | 影响 |
|--------|----------|----------|------|
| ~~文件格式支持~~ | ✅ 12 种 | 15+ 种 | 已大幅改善 |
| ~~多文件管理~~ | ✅ DASSpool | Spool 抽象 | 已实现 |
| ~~单位系统~~ | ✅ Pint 集成 | 单位支持 | **已实现 (2026-01)** |
| ~~示例数据~~ | ✅ get_example_frame | 内置示例 | **已实现 (2026-01)** |
| ~~时间工具~~ | ✅ to_datetime64 等 | 时间转换 | **已实现 (2026-01)** |
| 元数据管理 | 简单字典 | Inventory 系统 | 无法标准化元数据 |
| 互操作性 | 无 | ObsPy/DASCore 互转 | 与地震学生态隔离 |
| ~~测试覆盖~~ | ✅ 146个用例 | 90%+ | 已显著改善 |
| 有状态处理 | 无 | Atoms 框架 | 分块处理时滤波不连续 |

---

## 10. 优化方向与详细方案

### 10.1 ~~P0 优先级: 文件格式扩展~~ ✅ 已实现

> **更新 (2026-01)**: 已实现 12 种格式插件，包括所有 P0 和大部分 P1 格式。

**原问题**: 仅支持 4 种格式, 无法满足实际 DAS 数据处理需求

**已实现格式** (按使用频率排序):

| 格式 | 厂商/标准 | 优先级 |
|------|-----------|--------|
| TERRA15 | Terra15 | P0 |
| PRODML v2.0/v2.1 | 工业标准 | P0 |
| FEBUS | Febus Optics | P0 |
| APSENSING | AP Sensing | P1 |
| SILIXA_H5 | Silixa | P1 |
| OPTODAS | OptaSense | P1 |
| TDMS | LabVIEW/NI | P1 |
| ZARR | 云原生 | P1 |
| NETCDF | 科学数据标准 | P2 |

**技术方案**:

```python
# DASMatrix/acquisition/formats/base.py
from abc import ABC, abstractmethod
from typing import Protocol, Optional, Dict, Any
from pathlib import Path
import xarray as xr

class FormatPlugin(Protocol):
    """DAS 文件格式插件协议"""

    format_name: str
    version: str
    file_extensions: tuple[str, ...]

    def can_read(self, path: Path) -> bool:
        """快速检测文件是否为该格式"""
        ...

    def scan(self, path: Path) -> Dict[str, Any]:
        """快速扫描元数据, 不加载数据"""
        ...

    def read(self, path: Path, **kwargs) -> xr.DataArray:
        """读取数据为 xr.DataArray"""
        ...

    def write(self, data: xr.DataArray, path: Path, **kwargs) -> None:
        """写入数据"""
        ...

# DASMatrix/acquisition/formats/registry.py
class FormatRegistry:
    """格式注册表, 支持动态发现和注册"""

    _formats: Dict[str, FormatPlugin] = {}

    @classmethod
    def register(cls, plugin: FormatPlugin) -> None:
        cls._formats[plugin.format_name] = plugin

    @classmethod
    def detect_format(cls, path: Path) -> Optional[str]:
        """自动检测文件格式"""
        for name, plugin in cls._formats.items():
            if plugin.can_read(path):
                return name
        return None

    @classmethod
    def read(cls, path: Path, format: Optional[str] = None) -> xr.DataArray:
        """统一读取接口"""
        if format is None:
            format = cls.detect_format(path)
        if format is None:
            raise ValueError(f"Unknown format: {path}")
        return cls._formats[format].read(path)
```

**实施步骤**:

1. 参考 DASCore 的格式实现 (MIT 协议兼容)
2. 为每种格式创建 `FormatPlugin` 实现
3. 添加 `get_format()` 自动检测功能
4. 添加格式注册表和插件动态发现机制

---

### 10.2 ~~P0 优先级: 多文件 Spool 管理~~ ✅ 已实现

> **更新 (2026-01)**: `DASSpool` 已完整实现于 `DASMatrix/api/spool.py`。

**原问题**: 无法处理 TB 级多文件数据集

**实际实现**:

```python
# DASMatrix/api/spool.py
from typing import Union, Optional, Iterator, Tuple, List
from pathlib import Path
import xarray as xr

class DASSpool:
    """DAS 数据集管理器, 统一多文件访问接口

    功能:
    - 虚拟合并多个文件为单一数据集视图
    - 支持时间/空间范围查询
    - 支持分块迭代处理
    - 支持索引缓存加速二次访问

    Example:
        >>> spool = dm.spool("/data/das/*.h5").update()
        >>> subset = spool.select(time=("2024-01-01", "2024-01-02"))
        >>> for frame in subset.chunk(time=60_000):
        ...     processed = frame.bandpass(1, 100).collect()
    """

    def __init__(
        self,
        path: Union[str, Path, List[Path]],
        format: Optional[str] = None,
    ):
        """初始化 Spool

        Args:
            path: 单文件路径、目录路径、通配符模式或文件列表
            format: 强制指定格式, None 表示自动检测
        """
        self._paths: List[Path] = []
        self._index: Optional[xr.Dataset] = None  # 元数据索引
        self._format = format
        self._resolve_paths(path)

    def update(self) -> "DASSpool":
        """更新文件索引 (增量扫描新文件)"""
        ...
        return self

    def select(
        self,
        time: Optional[Tuple[str, str]] = None,
        distance: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> "DASSpool":
        """延迟筛选子集 (返回新 Spool, 不加载数据)"""
        ...

    def chunk(
        self,
        time: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> "DASSpool":
        """指定输出分块策略"""
        ...

    def __iter__(self) -> Iterator["DASFrame"]:
        """迭代返回 DASFrame 对象"""
        ...

    def __getitem__(self, idx: int) -> "DASFrame":
        """索引访问"""
        ...

    def __len__(self) -> int:
        """返回分块数量"""
        ...

    def to_zarr(self, path: str, virtual: bool = True) -> None:
        """导出为 Zarr 格式

        Args:
            path: 输出路径
            virtual: True 表示仅写入虚拟链接, False 表示复制数据
        """
        ...

    def to_netcdf(self, path: str, virtual: bool = True) -> None:
        """导出为 NetCDF 格式 (支持虚拟链接)"""
        ...

# 便捷函数
def spool(path: Union[str, Path, List[Path]], **kwargs) -> DASSpool:
    """创建 Spool 的便捷函数"""
    return DASSpool(path, **kwargs)
```

**使用示例**:

```python
import dasmatrix as dm

# 创建并更新索引
spool = dm.spool("/data/das/2024/*.h5").update()

# 筛选子集
subset = spool.select(
    time=("2024-01-01T00:00:00", "2024-01-01T12:00:00"),
    distance=(1000, 5000),
)

# 分块处理
for frame in subset.chunk(time=60_000, overlap=1_000):
    result = (
        frame
        .detrend()
        .bandpass(1, 100)
        .collect()
    )
    # 保存结果...
```

---

### 10.3 P1 优先级: 元数据 Inventory 系统

**问题**: 缺少结构化元数据管理, 无法与 PRODML/DAS-RCN 标准兼容

**技术方案**:

```python
# DASMatrix/core/inventory.py
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from datetime import datetime

class FiberGeometry(BaseModel):
    """光缆几何信息"""
    coordinates: Optional[List[Tuple[float, float, float]]] = None  # (lon, lat, depth)
    gauge_length: float = Field(..., description="标距长度 (m)")
    channel_spacing: float = Field(..., description="通道间距 (m)")
    total_length: Optional[float] = None  # 光缆总长度 (m)

class Interrogator(BaseModel):
    """询问器配置"""
    manufacturer: Optional[str] = None
    model: str
    serial_number: Optional[str] = None
    sampling_rate: float = Field(..., description="采样率 (Hz)")
    pulse_width: Optional[float] = None  # 脉冲宽度 (ns)
    pulse_rate: Optional[float] = None  # 脉冲重复率 (Hz)

class Acquisition(BaseModel):
    """采集元数据"""
    start_time: datetime
    end_time: Optional[datetime] = None
    n_channels: int
    n_samples: Optional[int] = None
    data_unit: str = "strain_rate"  # strain, strain_rate, velocity

class ProcessingStep(BaseModel):
    """处理步骤记录"""
    operation: str
    parameters: dict
    timestamp: datetime
    software: str = "DASMatrix"
    version: str

class DASInventory(BaseModel):
    """DAS 数据集元数据清单

    遵循 PRODML v2.1 / DAS-RCN 元数据标准
    """

    # 基础信息
    project: str
    experiment: Optional[str] = None
    description: Optional[str] = None

    # 设备信息
    fiber: Optional[FiberGeometry] = None
    interrogator: Optional[Interrogator] = None

    # 采集信息
    acquisition: Acquisition

    # 处理历史
    processing_history: List[ProcessingStep] = []

    # 自定义属性
    custom_attrs: dict = {}

    def to_json(self, path: Optional[str] = None) -> str:
        """导出为 JSON"""
        ...

    def to_parquet(self, path: str) -> None:
        """导出为 Parquet (用于快速查询)"""
        ...

    @classmethod
    def from_prodml(cls, path: str) -> "DASInventory":
        """从 PRODML 文件解析"""
        ...

    @classmethod
    def from_h5_attrs(cls, path: str) -> "DASInventory":
        """从 HDF5 属性解析"""
        ...
```

---

### 10.4 P1 优先级: Atoms 有状态流水线

**问题**: 大文件分块处理时, 有状态操作 (如 IIR 滤波) 无法保持连续性

**技术方案** (参考 Xdas):

```python
# DASMatrix/processing/atoms.py
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Any
import numpy as np
from scipy import signal

class Atom(ABC):
    """原子处理单元, 支持有状态操作

    Atom 是处理流水线的基本构建块, 特点:
    - 可以保持内部状态 (如滤波器状态)
    - 支持分块调用时状态连续传递
    - 可组合为复杂流水线
    """

    @abstractmethod
    def __call__(self, data: "DASFrame") -> "DASFrame":
        """处理数据"""
        ...

    def reset(self) -> None:
        """重置内部状态"""
        pass

class Partial(Atom):
    """将普通函数包装为无状态 Atom"""

    def __init__(self, func: Callable, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data: "DASFrame") -> "DASFrame":
        return self.func(data, *self.args, **self.kwargs)

class LFilter(Atom):
    """有状态 IIR 滤波器

    支持分块处理时保持滤波器状态连续, 避免边界效应
    """

    def __init__(self, b: np.ndarray, a: np.ndarray, dim: str = "time"):
        self.b = b
        self.a = a
        self.dim = dim
        self._zi: Optional[np.ndarray] = None

    def __call__(self, data: "DASFrame") -> "DASFrame":
        arr = data.collect()

        if self._zi is None:
            # 初始化滤波器状态
            zi = signal.lfilter_zi(self.b, self.a)
            self._zi = np.tile(zi[:, np.newaxis], (1, arr.shape[1]))

        # 带状态滤波
        filtered, self._zi = signal.lfilter(
            self.b, self.a, arr, axis=0, zi=self._zi
        )

        return DASFrame(filtered, fs=data.fs, dx=data._dx)

    def reset(self) -> None:
        self._zi = None

class SosFilt(Atom):
    """有状态 SOS (二阶节) 滤波器"""

    def __init__(self, sos: np.ndarray, dim: str = "time"):
        self.sos = sos
        self.dim = dim
        self._zi: Optional[np.ndarray] = None

    def __call__(self, data: "DASFrame") -> "DASFrame":
        arr = data.collect()

        if self._zi is None:
            zi = signal.sosfilt_zi(self.sos)
            self._zi = np.tile(zi[:, :, np.newaxis], (1, 1, arr.shape[1]))

        filtered, self._zi = signal.sosfilt(
            self.sos, arr, axis=0, zi=self._zi
        )

        return DASFrame(filtered, fs=data.fs, dx=data._dx)

    def reset(self) -> None:
        self._zi = None

class Sequential(Atom):
    """顺序组合多个 Atom"""

    def __init__(self, atoms: List[Atom]):
        self.atoms = atoms

    def __call__(self, data: "DASFrame") -> "DASFrame":
        for atom in self.atoms:
            data = atom(data)
        return data

    def reset(self) -> None:
        for atom in self.atoms:
            atom.reset()

class Parallel(Atom):
    """并行执行多个 Atom, 返回结果列表"""

    def __init__(self, atoms: List[Atom]):
        self.atoms = atoms

    def __call__(self, data: "DASFrame") -> List["DASFrame"]:
        return [atom(data) for atom in self.atoms]
```

**使用示例**:

```python
from scipy.signal import iirfilter
from DASMatrix.processing.atoms import Sequential, Partial, SosFilt

# 设计高通滤波器
sos = iirfilter(4, 10, btype="high", fs=1000, output="sos")

# 构建有状态流水线
pipeline = Sequential([
    Partial(lambda df: df.detrend()),
    SosFilt(sos, dim="time"),  # 有状态, 分块处理时保持连续
    Partial(lambda df: df.envelope()),
])

# 分块处理大文件 (滤波器状态自动传递)
for chunk in spool.chunk(time=10_000):
    result = pipeline(chunk)
    result.to_zarr("output/", append=True)

# 处理完成后重置状态
pipeline.reset()
```

---

### 10.5 P1 优先级: 互操作性增强

**问题**: 与 ObsPy/DASCore 生态隔离

**技术方案**:

```python
# DASMatrix/api/interop.py
# 在 DASFrame 类中添加互操作方法

class DASFrame:
    # ... 现有方法 ...

    # === 互操作方法 ===

    def to_obspy(self) -> "obspy.Stream":
        """转换为 ObsPy Stream

        用于与地震学工具链集成, 如 ObsPy, EQcorrscan 等
        """
        from obspy import Stream, Trace, UTCDateTime

        data = self.collect()
        traces = []

        for ch in range(data.shape[1]):
            tr = Trace(data=data[:, ch].astype(np.float32))
            tr.stats.sampling_rate = self._fs
            tr.stats.channel = f"CH{ch:04d}"
            tr.stats.starttime = UTCDateTime(0)  # 或从元数据获取
            tr.stats.network = "DAS"
            tr.stats.station = f"S{ch:04d}"
            traces.append(tr)

        return Stream(traces)

    def to_dascore(self) -> "dascore.Patch":
        """转换为 DASCore Patch"""
        import dascore as dc

        return dc.Patch(
            data=self.collect(),
            coords={
                "time": self._data.time.values,
                "distance": self._data.distance.values,
            },
            dims=("time", "distance"),
            attrs={"fs": self._fs, "dx": self._dx},
        )

    def to_xdas(self) -> "xdas.DataArray":
        """转换为 Xdas DataArray"""
        import xdas

        return xdas.DataArray(
            self._data.values,
            dims=("time", "distance"),
            coords={
                "time": self._data.time.values,
                "distance": self._data.distance.values,
            },
        )

    def to_dataframe(self) -> "pd.DataFrame":
        """转换为 Pandas DataFrame (长格式)"""
        import pandas as pd

        return self._data.to_dataframe().reset_index()

    @classmethod
    def from_obspy(
        cls,
        stream: "obspy.Stream",
        dx: float = 1.0
    ) -> "DASFrame":
        """从 ObsPy Stream 创建 DASFrame"""
        import numpy as np

        data = np.stack([tr.data for tr in stream], axis=1)
        fs = stream[0].stats.sampling_rate

        return cls(data, fs=fs, dx=dx)

    @classmethod
    def from_dascore(cls, patch: "dascore.Patch") -> "DASFrame":
        """从 DASCore Patch 创建 DASFrame"""
        fs = 1.0 / float(patch.coords.step("time").total_seconds())
        dx = float(patch.coords.step("distance"))

        return cls(patch.data, fs=fs, dx=dx)

    @classmethod
    def from_xarray(cls, da: "xr.DataArray", fs: float, dx: float = 1.0) -> "DASFrame":
        """从 Xarray DataArray 创建 DASFrame"""
        return cls(da, fs=fs, dx=dx)
```

---

### 10.6 P2 优先级: 测试与质量保障 (⚠️ 进行中)

**更新 (2026-01)**: 已建立性能基准测试体系 (`tests/performance`) 和 DSP 精度验证套件 (`tests/unit/test_accuracy.py`)。

**目标**: 测试覆盖率从 ~60% 提升到 90%+


**测试矩阵**:

| 测试类别 | 测试项 | 验收标准 |
|----------|--------|----------|
| **精度测试** | FFT 峰值位置 | 误差 < 1 bin |
| | 滤波器频率响应 | 通带衰减 < 0.5 dB |
| | 积分/微分精度 | 相对误差 < 1e-6 |
| **吞吐量测试** | bandpass (1GB 数据) | > 100 MB/s 单核 |
| | FFT (1GB 数据) | > 500 MB/s 单核 |
| | 文件读取 (HDF5) | > 200 MB/s |
| **延迟测试** | 实时处理 (2k ch × 50kSps) | < 80 ms E2E |
| **内存测试** | OOC 处理 10GB 文件 | 峰值内存 < 500 MB |
| **兼容性测试** | 各格式读写 | 全部通过 |

**测试框架**:

```python
# tests/conftest.py
import pytest
import numpy as np
from DASMatrix import df

@pytest.fixture
def synthetic_das_data():
    """生成合成 DAS 测试数据"""
    np.random.seed(42)
    fs = 1000
    n_samples = 10000
    n_channels = 100

    t = np.arange(n_samples) / fs
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

    data = np.zeros((n_samples, n_channels))
    for ch in range(n_channels):
        shift = int(ch * 2)
        data[:, ch] = np.roll(signal, shift) + 0.1 * np.random.randn(n_samples)

    return df(data, fs=fs, dx=1.0)

@pytest.fixture
def large_synthetic_data():
    """生成大规模测试数据 (用于性能测试)"""
    np.random.seed(42)
    return df(np.random.randn(100_000, 500), fs=10000, dx=1.0)

# tests/test_accuracy.py
class TestSignalProcessingAccuracy:
    """信号处理精度测试"""

    def test_bandpass_frequency_response(self, synthetic_das_data):
        """验证带通滤波器频率响应"""
        filtered = synthetic_das_data.bandpass(5, 20).collect()

        spectrum = np.abs(np.fft.rfft(filtered[:, 0]))
        freqs = np.fft.rfftfreq(len(filtered), 1/1000)

        idx_10hz = np.argmin(np.abs(freqs - 10))
        idx_50hz = np.argmin(np.abs(freqs - 50))

        # 10Hz 应通过, 50Hz 应衰减 20dB+
        assert spectrum[idx_10hz] > spectrum[idx_50hz] * 10

    def test_fft_peak_location(self, synthetic_das_data):
        """验证 FFT 峰值位置精度"""
        fft_result = synthetic_das_data.fft()
        # ... 验证峰值在预期频率

# tests/test_performance.py
class TestPerformance:
    """性能基准测试"""

    @pytest.mark.benchmark(group="bandpass")
    def test_bandpass_throughput(self, benchmark, large_synthetic_data):
        """带通滤波吞吐量测试"""
        result = benchmark(lambda: large_synthetic_data.bandpass(1, 100).collect())

        data_size_mb = large_synthetic_data.shape[0] * large_synthetic_data.shape[1] * 8 / 1e6
        throughput = data_size_mb / result.stats.mean

        assert throughput > 100, f"Throughput {throughput:.1f} MB/s < 100 MB/s"
```

---

## 11. 实施路线图

### 11.1 Phase 1: 基础完善 (2025 Q1, 8 周)

| 任务 | 优先级 | 工时 | 负责人 | 依赖 |
|------|--------|------|--------|------|
| TERRA15 格式支持 | P0 | 1 周 | - | - |
| PRODML v2.0/v2.1 格式支持 | P0 | 1 周 | - | - |
| FEBUS 格式支持 | P0 | 1 周 | - | - |
| 格式注册表与自动检测 | P0 | 0.5 周 | - | 上述格式 |
| DASSpool 核心实现 | P0 | 2 周 | - | 格式支持 |
| 单元测试完善 (覆盖率 80%+) | P1 | 1 周 | - | - |
| MkDocs 文档完善 | P1 | 1 周 | - | - |

**Phase 1 交付物**:

- 支持 6+ 种主流 DAS 格式
- DASSpool 多文件管理
- 测试覆盖率 80%+
- 完善的 API 文档

### 11.2 Phase 2: 功能增强与极致性能 (2025 Q2 - 2026 Q1) [已结项]

| 任务 | 优先级 | 状态 | 成果 |
|------|--------|------|------|
| 极致性能优化 (Dask/Numba) | P0 | ✅ 已完成 | 核外处理、算子融合、Welford 归一化 |
| 现代 STFT API 升级 | P0 | ✅ 已完成 | 采用 ShortTimeFFT，支持 TB 级延迟分析 |
| 性能基准测试套件 | P1 | ✅ 已完成 | 建立 tests/performance 体系 |
| APSENSING/SILIXA 格式 | P1 | ✅ 已完成 | 支持 12+ 种主流 DAS 格式 |
| 智能可视化保护 | P1 | ✅ 已完成 | 自动 Decimation 降采样保护 |

**Phase 2 交付物**:

- 完整的元数据管理系统
- 有状态流水线处理
- 与 ObsPy/DASCore 互操作
- 性能基准测试报告

### 11.3 Phase 3: 智能分析与生态集成 (2026 Q1-Q2, 12 周)

| 任务 | 优先级 | 负责人 | 目标与技术细节 |
|------|--------|--------|----------------|
| **DASInventory 深度集成** | P1 | - | 1. 升级 FormatPlugin 接口以支持全量 Inventory 返回；2. 针对 PRODML/H5 实现元数据自动映射逻辑。 |
| **Atoms 有状态流水线** | P1 | - | 1. 实现 `processing/atoms.py` 核心抽象；2. 实现 `SosFilt` 有状态滤波，确保分块处理时无边界突变。 |
| **Ecosystem Interop** | P1 | - | 1. 建立与 ObsPy (Stream/Trace) 的双向转换；2. 支持与 DASCore (Patch) 的互转，打破数据隔离。 |
| **分析模块 (analysis/)** | P1 | - | 1. 实现经典 STA/LTA 事件检测算法；2. 增加基于 Polars 的高速事件属性查询功能。 |
| **GPU (CuPy) 后端融合** | P2 | - | 1. 将 Numba JIT 内核逻辑迁移至 CuPy/CUDA 内核；2. 支持超大规模矩阵的 GPU 加速 FFT。 |
| **PyPI 正式发布** | P1 | - | 1. 完善文档与教程视频；2. 发布 v0.2.0 稳定版至 PyPI。 |

| 任务 | 优先级 | 工时 | 依赖 |
|------|--------|------|------|
| GPU (CuPy) 后端 | P2 | 4 周 | - |
| Ray 分布式后端 | P2 | 4 周 | - |
| 完整流处理支持 | P2 | 2 周 | - |
| ML 检测器集成 | P2 | 2 周 | - |
| PyPI 发布 | P1 | 1 周 | Phase 2 |
| 教程视频制作 | P2 | 2 周 | 文档 |
| 剩余格式支持 | P2 | 2 周 | - |

**Phase 3 交付物**:

- GPU 加速支持
- 分布式计算支持
- PyPI 正式发布
- 完整教程和示例

---

## 12. 成功指标

### 12.1 功能指标

| 指标 | 当前 | Phase 1 目标 | Phase 2 目标 | Phase 3 目标 |
|------|------|--------------|--------------|--------------|
| 文件格式支持 | 4 | 8 | 12 | 15+ |
| 测试覆盖率 | ~60% | 80% | 90% | 95% |
| API 方法数 | ~50 | 70 | 100 | 120 |
| 示例数量 | 5 | 10 | 15 | 20 |

### 12.2 性能指标

| 指标 | 当前 | 目标 |
|------|------|------|
| 带通滤波吞吐量 (单核) | ~50 MB/s | > 100 MB/s |
| FFT 吞吐量 (单核) | ~200 MB/s | > 500 MB/s |
| 文件读取速度 (HDF5) | ~100 MB/s | > 200 MB/s |
| 实时处理延迟 | ~100 ms | < 50 ms |
| OOC 内存占用 | 无限制 | < 500 MB |

### 12.3 社区指标 (Phase 3 后)

| 指标 | 目标 |
|------|------|
| GitHub Stars | 100+ |
| PyPI 月下载量 | 500+ |
| 贡献者数量 | 5+ |
| 文档访问量 | 1000+ PV/月 |

---


## 14. 性能优化成果 (2026-01)

已成功实施以下高性能计算（HPC）优化：

### 14.1 [已完成] 极致 Dask 分块策略
- **成果**: 彻底移除硬编码 `time=-1`，全链路支持 Out-of-Core 处理。
- **技术**: 基于 `xr.apply_ufunc(dask="parallelized")` 实现流式并行计算。

### 14.2 [已完成] 现代真延迟 STFT/FK 变换
- **成果**: 升级至 SciPy `ShortTimeFFT` 现代 API，支持对 TB 级数据的延迟时频分析。
- **技术**: 自定义 Dask Block 映射，确保时间轴坐标精确对齐。

### 14.3 [已完成] 智能可视化保护 (Plot Protection)
- **成果**: 绘图速度提升 100x+，TB 级数据绘图不再卡死。
- **技术**: 自动 `decimation` 降采样，仅采集绘图所需的局部/核心数据。

### 14.4 [已完成] Numba 单次扫描算子融合
- **成果**: 归一化与统计速度提升 2x+，深度饱和硬件带宽。
- **技术**: 内置 Welford 算法实现均值/方差单次遍历计算。

## 15. 未来演进计划 (M4+)

- [ ] **GPU (CuPy) 后端融合**: 将现有 Numba 内核扩展至 CUDA 设备。
- [ ] **Ray 分布式调度**: 实现跨节点的大规模集群 DAS 处理。
- [ ] **有状态流式滤波**: 引入有状态 Atoms 框架，支持实时不间断数据的无缝滤波。

## 13. 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 格式实现复杂度超预期 | 中 | 高 | 参考 DASCore 实现, 分批实现 |
| 与竞品功能重叠 | 中 | 中 | 差异化定位 (Web Dashboard, GPU) |
| 社区采用缓慢 | 中 | 中 | 加强文档、教程, 参与学术会议 |
| 性能目标难以达成 | 低 | 高 | 渐进式优化, 优先保证正确性 |
| 维护资源不足 | 中 | 高 | 明确核心功能优先级, 延迟低优先级特性 |

---

这一架构保留了原始设计的所有优点，同时通过DASFrame列式对象、多后端计算图、分级内存管理和DSL语法增强了易用性和性能，为用户提供了类似data.table的极简体验，并为未来的分布式扩展做好了准备。

通过竞品分析明确了与成熟库的差距，制定了清晰的优化路线图，预计在 2025 年 Q3 可达到与 DASCore/Xdas 同等水平，并在 Web Dashboard、GPU 加速等方面形成差异化优势。
