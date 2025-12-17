from typing import Any, Optional, Union, List, cast
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from ..core.computation_graph import (
    ComputationGraph,
    Node,
    NodeDomain,
    OperationNode,
)
from ..processing.engine import HybridEngine


class DASFrame:
    """DAS 数据处理核心类，提供流畅的链式 API。

    DASFrame 采用延迟计算（Lazy Evaluation）模式，所有信号处理操作不会立即执行，
    而是构建计算图，直到调用 `collect()` 或可视化方法时才真正计算。这种设计允许
    引擎进行算子融合等优化，提高计算效率。

    内部使用混合执行引擎（HybridEngine），协调 NumPy/SciPy 后端和 Numba JIT 后端。

    Attributes:
        _fs: 采样频率 (Hz)
        _metadata: 用户自定义的元数据字典
        _source_data: 原始数据引用
        _graph: 计算图实例
        _node: 当前指向的计算图节点
        _engine: 混合执行引擎实例

    Example:
        >>> import numpy as np
        >>> from DASMatrix import df
        >>> data = np.random.randn(1000, 10)  # 1000 样本, 10 通道
        >>> result = (
        ...     df(data, fs=1000)
        ...     .detrend()
        ...     .bandpass(1, 100)
        ...     .normalize()
        ...     .collect()
        ... )
    """

    def __init__(
        self,
        data: Any,
        fs: float,
        graph: Optional[ComputationGraph] = None,
        node: Optional[Node] = None,
        **metadata: Any,
    ) -> None:
        """初始化 DASFrame 实例。

        Args:
            data: 输入数据，通常为形状 (n_samples, n_channels) 的 NumPy 数组
            fs: 采样频率 (Hz)
            graph: 计算图实例，用于链式操作传递（内部使用）
            node: 当前指向的计算图节点（内部使用）
            **metadata: 用户自定义的元数据，如 channel_names、units 等
        """
        self._fs = fs
        self._metadata = metadata
        self._source_data = data

        self._engine = HybridEngine()

        if graph is None:
            self._graph = ComputationGraph.leaf(data)
            self._node = self._graph.root
        else:
            self._graph = graph
            self._node = node

    @property
    def _data(self) -> np.ndarray:
        """获取原始数据（用于兼容性和测试）。"""
        if self._source_data is not None:
            return self._source_data
        # 如果没有源数据，执行计算获取
        return self.collect()

    def _apply_signal_op(self, name: str, **kwargs) -> "DASFrame":
        """应用信号域操作 (Signal Domain Op)。"""
        if self._node is None:
            raise ValueError("Current node is None")
        new_node = OperationNode(
            operation=name,
            inputs=[self._node],
            args=(),  # Assuming args should be an empty tuple by default
            kwargs=kwargs,
            domain=NodeDomain.SIGNAL,
            name=name,
        )

        return DASFrame(
            data=self._source_data,  # 保持源数据引用
            fs=self._fs,
            graph=self._graph,  # 共享图引用
            node=new_node,  # 指向新头部
            **self._metadata,
        )

    # --- 基础操作 ---
    def collect(self) -> np.ndarray:
        """触发延迟计算并返回结果数据。

        这是延迟计算的终止方法。调用此方法时，计算图中的所有操作将被优化
        并执行，返回最终的 NumPy 数组。

        Returns:
            np.ndarray: 计算后的数据数组，形状与输入相同或根据操作变化

        Example:
            >>> frame = df(data, fs=1000).bandpass(1, 100)
            >>> result = frame.collect()  # 此时才真正执行滤波
        """
        exec_graph = ComputationGraph(self._node)
        return self._engine.compute(exec_graph)

    def slice(self, t: slice = slice(None), x: slice = slice(None)) -> "DASFrame":
        """对数据进行切片操作。

        Args:
            t: 时间维度的切片 (行)
            x: 通道维度的切片 (列)

        Returns:
            DASFrame: 切片后的新 DASFrame
        """
        return self._apply_signal_op("slice", t=t, x=x)

    # --- 时域 (Signal Domain) ---
    def detrend(self, axis: str = "time") -> "DASFrame":
        """去趋势，移除信号的线性趋势成分。

        使用最小二乘法拟合并减去线性趋势，常用于预处理步骤。

        Args:
            axis: 去趋势的方向，'time' 表示沿时间轴（默认）

        Returns:
            DASFrame: 去趋势后的新 DASFrame 实例
        """
        return self._apply_signal_op("detrend", axis=axis)

    def demean(self, axis: str = "time") -> "DASFrame":
        """去均值，移除信号的直流分量。

        沿指定轴计算均值并减去，使信号零均值化。

        Args:
            axis: 去均值的方向，'time' 表示沿时间轴（默认）

        Returns:
            DASFrame: 去均值后的新 DASFrame 实例
        """
        return self._apply_signal_op("demean", axis=axis)

    def abs(self) -> "DASFrame":
        """计算信号的绝对值。

        对每个采样点取绝对值，常用于包络分析或能量计算。

        Returns:
            DASFrame: 绝对值后的新 DASFrame 实例
        """
        return self._apply_signal_op("abs")

    def scale(self, factor: float = 1.0) -> "DASFrame":
        """按指定因子缩放信号幅度。

        将所有采样值乘以缩放因子。

        Args:
            factor: 缩放因子，默认为 1.0（不缩放）

        Returns:
            DASFrame: 缩放后的新 DASFrame 实例
        """
        return self._apply_signal_op("scale", factor=factor)

    def normalize(self, method: str = "minmax") -> "DASFrame":
        """归一化。

        Args:
            method: 归一化方法
                - 'minmax': 归一化到 [-1, 1]
                - 'zscore': Z-score 标准化 (mean=0, std=1)

        Returns:
            DASFrame: 归一化后的新 DASFrame
        """
        return self._apply_signal_op("normalize", method=method)

    # --- 滤波器 ---
    def bandpass(
        self,
        low: float,
        high: float,
        order: int = 4,
        fs: Optional[float] = None,
        design: str = "butter",
    ) -> "DASFrame":
        """带通滤波器，保留指定频率范围内的成分。

        使用 Butterworth 滤波器设计，通过 SciPy 的零相位滤波（filtfilt）实现，
        避免相位失真。

        Args:
            low: 低截止频率 (Hz)
            high: 高截止频率 (Hz)
            order: 滤波器阶数，默认为 4
            fs: 采样频率 (Hz)，默认使用初始化时的采样频率
            design: 滤波器设计类型，目前仅支持 'butter'

        Returns:
            DASFrame: 滤波后的新 DASFrame 实例

        Raises:
            ValueError: 当 low >= high 或频率超过奈奎斯特频率时

        Example:
            >>> filtered = df(data, fs=1000).bandpass(1, 100)
        """
        return self._apply_signal_op(
            "bandpass", low=low, high=high, order=order, fs=fs or self._fs
        )

    def lowpass(self, cutoff: float, order: int = 4) -> "DASFrame":
        """低通滤波器，移除高于截止频率的成分。

        使用 Butterworth 滤波器设计，通过零相位滤波实现。

        Args:
            cutoff: 截止频率 (Hz)
            order: 滤波器阶数，默认为 4

        Returns:
            DASFrame: 滤波后的新 DASFrame 实例
        """
        return self._apply_signal_op("lowpass", cutoff=cutoff, order=order, fs=self._fs)

    def highpass(self, cutoff: float, order: int = 4) -> "DASFrame":
        """高通滤波器，移除低于截止频率的成分。

        使用 Butterworth 滤波器设计，通过零相位滤波实现。

        Args:
            cutoff: 截止频率 (Hz)
            order: 滤波器阶数，默认为 4

        Returns:
            DASFrame: 滤波后的新 DASFrame 实例
        """
        return self._apply_signal_op(
            "highpass", cutoff=cutoff, order=order, fs=self._fs
        )

    def notch(self, freq: float, Q: float = 30) -> "DASFrame":
        """陷波滤波器，移除特定频率成分。

        Args:
            freq: 需要移除的频率 (Hz)
            Q: 品质因数，越大带宽越窄

        Returns:
            DASFrame: 滤波后的新 DASFrame
        """
        return self._apply_signal_op("notch", freq=freq, Q=Q, fs=self._fs)

    def median_filter(self, k: int = 5, axis: str = "time") -> "DASFrame":
        """中值滤波。

        Args:
            k: 滤波窗口大小
            axis: 滤波轴，'time' 或 'channel'

        Returns:
            DASFrame: 滤波后的新 DASFrame
        """
        return self._apply_signal_op("median_filter", k=k, axis=axis)

    # --- 频域 (Frequency Domain) ---
    def fft(self) -> "DASFrame":
        """快速傅立叶变换，计算频谱幅度。

        对每个通道沿时间轴进行 FFT，返回频谱幅度（绝对值）。

        Returns:
            DASFrame: 包含频谱幅度的新 DASFrame 实例
        """
        return self._apply_signal_op("fft")

    def stft(self, nperseg: int = 256, noverlap: Optional[int] = None) -> "DASFrame":
        """短时傅立叶变换，进行时频分析。

        将信号分割成重叠的窗口并对每个窗口进行 FFT，获得时频谱表示。

        Args:
            nperseg: 每个窗口的采样点数，默认为 256
            noverlap: 窗口重叠点数，默认为 nperseg // 2

        Returns:
            DASFrame: 包含时频谱的新 DASFrame 实例
        """
        if noverlap is None:
            noverlap = nperseg // 2
        return self._apply_signal_op(
            "stft", nperseg=nperseg, noverlap=noverlap, fs=self._fs
        )

    def hilbert(self) -> "DASFrame":
        """希尔伯特变换，返回解析信号。

        Returns:
            DASFrame: 包含复数解析信号的新 DASFrame
        """
        return self._apply_signal_op("hilbert")

    def envelope(self) -> "DASFrame":
        """提取信号包络。

        通过希尔伯特变换计算解析信号，并取其绝对值作为包络。
        包络表示信号的瞬时幅度变化。

        Returns:
            DASFrame: 包含包络的新 DASFrame 实例
        """
        return self._apply_signal_op("envelope")

    # --- 统计 (Statistics) ---
    def mean(self, axis: Optional[int] = 0) -> np.ndarray:
        """计算均值。

        Args:
            axis: 计算轴，0 表示沿时间轴返回每通道的均值，None 返回全局标量

        Returns:
            np.ndarray: 均值结果
        """
        data = self.collect()
        return np.mean(data, axis=axis)

    def std(self, axis: Optional[int] = 0) -> np.ndarray:
        """计算标准差。

        Args:
            axis: 计算轴，0 表示沿时间轴返回每通道的标准差，None 返回全局标量

        Returns:
            np.ndarray: 标准差结果
        """
        data = self.collect()
        return np.std(data, axis=axis)

    def max(self, axis: Optional[int] = 0) -> np.ndarray:
        """计算最大值。

        Args:
            axis: 计算轴，0 表示沿时间轴返回每通道的最大值，None 返回全局标量

        Returns:
            np.ndarray: 最大值结果
        """
        data = self.collect()
        return np.max(data, axis=axis)

    def min(self, axis: Optional[int] = 0) -> np.ndarray:
        """计算最小值。

        Args:
            axis: 计算轴，0 表示沿时间轴返回每通道的最小值，None 返回全局标量

        Returns:
            np.ndarray: 最小值结果
        """
        data = self.collect()
        return np.min(data, axis=axis)

    def rms(self, window: Optional[int] = None) -> np.ndarray:
        """计算 RMS（均方根）。

        Args:
            window: 滑动窗口大小，默认为 None 表示计算全局 RMS

        Returns:
            np.ndarray: RMS 结果，如指定 window 则返回滑动 RMS
        """
        data = self.collect()
        if window is None:
            return np.sqrt(np.mean(data**2, axis=0))
        else:
            from scipy.ndimage import uniform_filter1d
            return np.sqrt(uniform_filter1d(data**2, size=window, axis=0))

    # --- 检测 (Detection) ---
    def threshold_detect(
        self, threshold: Optional[float] = None, sigma: float = 3.0
    ) -> np.ndarray:
        """阈值检测，检测超过阈值的采样点。

        Args:
            threshold: 自定义阈值，默认为 None 表示使用 mean + sigma * std
            sigma: 标准差倍数，用于自动计算阈值

        Returns:
            np.ndarray: 布尔矩阵，True 表示该点超过阈值

        Example:
            >>> detections = df(data, fs=1000).threshold_detect(sigma=3.0)
        """
        data = self.collect()
        if threshold is None:
            threshold = np.mean(data) + sigma * np.std(data)
        return np.abs(data) > threshold

    # --- 可视化 (Visualization) ---
    def plot_ts(
        self,
        ch: Optional[int] = None,
        title: str = "Time Series",
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> plt.Figure:
        """绘制时间序列。"""
        data = self.collect()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            if ax.figure is None:
                raise ValueError("Provided ax must belong to a figure")
            fig = cast(plt.Figure, ax.figure)

        t = np.arange(data.shape[0]) / self._fs
        if ch is None:
            ax.plot(t, data[:, : min(5, data.shape[1])], alpha=0.7)
        else:
            ax.plot(t, data[:, ch])

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig

    def plot_heatmap(
        self,
        title: str = "DAS Waterfall",
        cmap: str = "RdBu_r",
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> plt.Figure:
        """绘制热图（瀑布图）。"""
        data = self.collect()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            if ax.figure is None:
                raise ValueError("Provided ax must belong to a figure")
            fig = cast(plt.Figure, ax.figure)

        extent = (0, float(data.shape[1]), float(data.shape[0] / self._fs), 0)
        vmax = np.percentile(np.abs(data), 98)
        im = ax.imshow(
            data,
            aspect="auto",
            cmap=cmap,
            extent=extent,
            vmin=-vmax,
            vmax=vmax,
            **kwargs,
        )

        ax.set_xlabel("Channel")
        ax.set_ylabel("Time (s)")
        ax.set_title(title)
        # 仅当创建新 figure 时添加 colorbar，或者是显式要求
        # 简单起见，这里总是尝试添加，但需要注意 layout
        if ax is not None:
            plt.colorbar(im, ax=ax, label="Amplitude")
        else:
            plt.colorbar(im, ax=ax, label="Amplitude")
        return fig

    def plot_spec(
        self,
        ch: int = 0,
        nperseg: int = 256,
        title: str = "Spectrogram",
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> plt.Figure:
        """绘制频谱图。"""
        data = self.collect()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            if ax.figure is None:
                raise ValueError("Provided ax must belong to a figure")
            fig = cast(plt.Figure, ax.figure)

        f, t, Sxx = signal.spectrogram(data[:, ch], fs=self._fs, nperseg=nperseg)

        im = ax.pcolormesh(
            t, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="viridis"
        )
        plt.colorbar(im, ax=ax, label="Power (dB)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_title(title)
        return fig
