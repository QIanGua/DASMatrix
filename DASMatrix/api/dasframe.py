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
    """DAS数据处理核心类 (Hybrid Engine Version)。

    协调 Polars (Metadata) 和 Numba (Signal) 引擎。
    所有操作均为 Lazy，直到调用 plot_* 或 collect。
    """

    def __init__(
        self,
        data: Any,
        fs: float,
        graph: Optional[ComputationGraph] = None,
        node: Optional[Node] = None,  # 当前指向的图节点
        **metadata,
    ):
        self._fs = fs
        self._metadata = metadata
        self._source_data = data  # 保存原始数据引用

        # 引擎初始化 (通常单例或全局)
        self._engine = HybridEngine()

        if graph is None:
            # 初始创建
            self._graph = ComputationGraph.leaf(data)
            self._node = self._graph.root
        else:
            # 链式传递
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
        """触发计算并返回结果数据。"""
        # 构建一个临时的执行图，以当前节点为 root
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
    def detrend(self, axis="time") -> "DASFrame":
        """去趋势。"""
        return self._apply_signal_op("detrend", axis=axis)

    def demean(self, axis="time") -> "DASFrame":
        """去均值。"""
        return self._apply_signal_op("demean", axis=axis)

    def abs(self) -> "DASFrame":
        """绝对值。"""
        return self._apply_signal_op("abs")

    def scale(self, factor=1.0) -> "DASFrame":
        """缩放。"""
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
    def bandpass(self, low, high, order=4, fs=None, design="butter") -> "DASFrame":
        """带通滤波。"""
        return self._apply_signal_op(
            "bandpass", low=low, high=high, order=order, fs=fs or self._fs
        )

    def lowpass(self, cutoff, order=4) -> "DASFrame":
        """低通滤波。"""
        return self._apply_signal_op("lowpass", cutoff=cutoff, order=order, fs=self._fs)

    def highpass(self, cutoff, order=4) -> "DASFrame":
        """高通滤波。"""
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
        """快速傅里叶变换。"""
        return self._apply_signal_op("fft")

    def stft(self, nperseg=256, noverlap=None) -> "DASFrame":
        """短时傅里叶变换。"""
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
        """希尔伯特变换提取包络。"""
        return self._apply_signal_op("envelope")

    # --- 统计 (Statistics) ---
    def mean(self, axis=0):
        """计算均值。

        Args:
            axis: 计算轴，默认 0 返回每通道的均值，None 返回全局标量
        """
        data = self.collect()
        return np.mean(data, axis=axis)

    def std(self, axis=0):
        """计算标准差。

        Args:
            axis: 计算轴，默认 0 返回每通道的标准差，None 返回全局标量
        """
        data = self.collect()
        return np.std(data, axis=axis)

    def max(self, axis=0):
        """计算最大值。

        Args:
            axis: 计算轴，默认 0 返回每通道的最大值，None 返回全局标量
        """
        data = self.collect()
        return np.max(data, axis=axis)

    def min(self, axis=0):
        """计算最小值。

        Args:
            axis: 计算轴，默认 0 返回每通道的最小值，None 返回全局标量
        """
        data = self.collect()
        return np.min(data, axis=axis)

    def rms(self, window=None):
        """计算 RMS（均方根）。"""
        data = self.collect()
        if window is None:
            return np.sqrt(np.mean(data**2, axis=0))
        else:
            # 滑动窗口 RMS
            from scipy.ndimage import uniform_filter1d

            return np.sqrt(uniform_filter1d(data**2, size=window, axis=0))

    # --- 检测 (Detection) ---
    def threshold_detect(self, threshold=None, sigma=3.0):
        """阈值检测。

        Args:
            threshold: 自定义阈值，如果为 None 则使用 mean + sigma * std
            sigma: 标准差倍数

        Returns:
            检测结果矩阵 (bool)
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
