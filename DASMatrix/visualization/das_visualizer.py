import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from ..config.visualization_config import VisualizationConfig
from .plot_base import PlotBase
from .plot_profiles import FKPlot, ProfilePlot
from .plot_signals import SpectrogramPlot, SpectrumPlot, WaterfallPlot, WaveformPlot

__all__ = [
    "PlotBase",
    "SpectrumPlot",
    "WaveformPlot",
    "SpectrogramPlot",
    "WaterfallPlot",
    "FKPlot",
    "ProfilePlot",
    "DASVisualizer",
]


class DASVisualizer:
    """DAS 数据可视化器 (DAS Data Visualizer)"""

    def __init__(
        self,
        output_path: Union[str, Path],
        sampling_frequency: float,
        config: Optional[VisualizationConfig] = None,
    ):
        """初始化 DAS 数据可视化器

        Args:
            output_path: 图像输出目录
            sampling_frequency: 采样频率 (Hz)
            config: 可视化配置，若未指定则使用默认配置
        """
        self.output_path = Path(output_path) if isinstance(output_path, str) else output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.fs = sampling_frequency
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)

        # 初始化绘图器实例
        self.waveform_plotter = WaveformPlot(self.config)
        self.spectrum_plotter = SpectrumPlot(self.config)
        self.spectrogram_plotter = SpectrogramPlot(self.config)
        self.waterfall_plotter = WaterfallPlot(self.config)
        self.profile_plotter = ProfilePlot(self.config)

        # 检测是否在 Jupyter 环境中
        self._in_jupyter = self._check_jupyter()

        # 性能优化：预先导入常用库，避免每次绘图都重新导入
        try:
            import matplotlib.pyplot as plt

            self._plt = plt
            from scipy import signal

            self._signal = signal
        except ImportError:
            self.logger.warning("Could not preload matplotlib or scipy.signal")

    def _check_jupyter(self) -> bool:
        """检查是否在 Jupyter 环境中运行"""
        try:
            from IPython import get_ipython

            if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
                return True
            return False
        except ImportError:
            return False

    def _handle_figure_display(self, fig) -> None:
        """处理图形显示，避免在 Jupyter 中双重显示"""
        if self._in_jupyter:
            # 在 Jupyter 中，导入必要的库来控制显示
            import matplotlib.pyplot as plt
            from IPython.display import display

            # 显示图形并关闭 plt 自动显示
            display(fig)
            plt.close(fig)
            return None
        else:
            # 非 Jupyter 环境，正常返回图形对象
            return fig

    def WaveformPlot(
        self,
        amplitude_data: np.ndarray,
        time_range: Optional[Tuple[float, float]] = None,
        amplitude_range: Optional[Tuple[float, float]] = None,
        file_name: Optional[str] = None,
        title: Optional[str] = None,
        highlight_regions: Optional[List[Tuple[float, float]]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> Any:
        """绘制时域波形图

        Args:
            amplitude_data: 一维数组，包含单个通道的时域数据
            time_range: 可选，时间显示范围 (start_time, end_time)
            amplitude_range: 可选，幅值显示范围 (min_amplitude, max_amplitude)
            file_name: 保存的文件名 (不含扩展名)，若提供则保存图片
            title: 图表标题 (英文)
            highlight_regions: 可选的高亮区域列表，每个元素为(开始时间, 结束时间)元组

        Returns:
            Figure: matplotlib 图形对象，可在 Jupyter notebook 中显示
        """
        # 性能优化：加入数据验证前的快速路径检查
        if amplitude_data is None or len(amplitude_data) == 0:
            self.logger.error("输入数据为空")
            return None

        if not isinstance(amplitude_data, np.ndarray) or amplitude_data.ndim != 1:
            self.logger.error(f"数据必须是一维numpy数组，当前维度: {amplitude_data.ndim}")
            return None

        # 绘图前启用agg后端可以提高非交互式绘图性能
        import matplotlib

        prev_backend = matplotlib.get_backend()
        if not self._in_jupyter:
            matplotlib.use("agg")  # 非交互式后端，更快

        # 绘制波形图
        fig = self.waveform_plotter.plot(
            amplitude_data=amplitude_data,
            fs=self.fs,  # 传递采样频率
            time_range=time_range,
            amplitude_range=amplitude_range,
            title=title,
            highlight_regions=highlight_regions,
            ax=ax,
        )

        # 如果提供了文件名，则保存图片
        if file_name is not None:
            save_path = self.output_path / f"{file_name}.png"
            self.waveform_plotter.save_figure(save_path, close_after_save=False)

        # 还原后端
        if not self._in_jupyter:
            matplotlib.use(prev_backend)

        # 处理图形显示，避免双重显示
        return self._handle_figure_display(fig)

    def SpectrumPlot(
        self,
        data: np.ndarray,
        excitation_freq: Optional[float] = None,
        file_name: Optional[str] = None,
        title: Optional[str] = None,
        peaks: Optional[List[Dict[str, float]]] = None,
        freq_range: Optional[Tuple[float, float]] = None,
        db_range: Optional[Tuple[float, float]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> Any:
        """绘制并保存频谱图

        Args:
            data: 一维数组，包含单个通道的时域信号数据
            excitation_freq: (可选) 激励频率 (Hz)
            file_name: 保存的文件名 (不含扩展名)，若提供则保存图片
            title: 图表标题 (英文)
            peaks: (可选) 峰值列表，每个峰值包含 'frequency' 和 'magnitude'
            freq_range: (可选) 频率范围 (min_freq, max_freq)，用于限制显示的频率范围
            db_range: 可选，dB值显示范围 (min_db, max_db)

        Returns:
            Figure: matplotlib 图形对象，可在 Jupyter notebook 中显示
        """
        # 性能优化：加入数据验证前的快速路径检查
        if data is None or len(data) == 0:
            self.logger.error("输入数据为空")
            return None

        if not isinstance(data, np.ndarray) or data.ndim != 1:
            self.logger.error(f"数据必须是一维numpy数组，当前维度: {data.ndim}")
            return None

        # 性能优化：使用 Welch 方法计算频谱，比直接 FFT 更稳定且可能更快
        from scipy import signal

        # 根据数据长度自动确定合适的窗口大小和重叠参数
        n = len(data)
        if n > 500000:  # 大于50万点的数据
            nperseg = 8192  # 使用较大的窗口增加频率分辨率
            noverlap = nperseg // 2  # 50% 重叠
        elif n > 100000:  # 10万-50万点的数据
            nperseg = 4096
            noverlap = nperseg // 2
        else:  # 较小规模的数据
            nperseg = min(2048, n // 4)  # 确保窗口不超过数据长度的1/4
            noverlap = nperseg // 2

        # 使用 Welch 方法计算功率谱密度 (PSD)
        frequencies, psd = signal.welch(
            data,
            fs=self.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="spectrum",  # 使用 'spectrum' 而不是 'density' 以获得幅度谱
            average="mean",  # 平均多个分段的结果
        )

        # Welch 返回功率谱密度，我们计算幅度谱
        magnitudes = np.sqrt(psd)

        # 如果指定了频率范围，则过滤数据
        if freq_range is not None:
            min_freq, max_freq = freq_range
            mask = (frequencies >= min_freq) & (frequencies <= max_freq)
            frequencies = frequencies[mask]
            magnitudes = magnitudes[mask]

        # 绘图前启用agg后端可以提高非交互式绘图性能
        import matplotlib

        prev_backend = matplotlib.get_backend()
        if not self._in_jupyter:
            matplotlib.use("agg")  # 非交互式后端，更快

        # 绘制频谱图
        fig = self.spectrum_plotter.plot(
            frequencies,
            magnitudes,
            excitation_freq=excitation_freq,
            title=title,
            peaks=peaks,
            db_range=db_range,
            ax=ax,
        )

        # 如果提供了文件名，则保存图片
        if file_name is not None:
            freq_suffix = f"_{excitation_freq:.0f}Hz" if excitation_freq else ""
            save_path = self.output_path / f"{file_name}{freq_suffix}.png"
            self.spectrum_plotter.save_figure(save_path, close_after_save=False)

        # 还原后端
        if not self._in_jupyter:
            matplotlib.use(prev_backend)

        # 处理图形显示，避免双重显示
        return self._handle_figure_display(fig)

    def SpectrogramPlot(
        self,
        data: np.ndarray,
        window_size: int = 1024,
        overlap: float = 0.75,
        file_name: Optional[str] = None,
        title: Optional[str] = None,
        freq_range: Optional[Tuple[float, float]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        cmap: str = "viridis",
        ax: Optional[plt.Axes] = None,
    ) -> Any:
        """绘制时频图（短时傅里叶变换）

        Args:
            data: 一维数组，包含单个通道的时域数据
            window_size: STFT窗口大小
            overlap: 窗口重叠比例
            file_name: 保存的文件名 (不含扩展名)，若提供则保存图片
            title: 图表标题 (英文)
            freq_range: 可选，频率范围 (min_freq, max_freq)
            time_range: 可选，时间范围 (start_time, end_time)
            cmap: 色彩映射名称

        Returns:
            Figure: matplotlib 图形对象，可在 Jupyter notebook 中显示
        """
        # 性能优化：加入数据验证前的快速路径检查
        if data is None or len(data) == 0:
            self.logger.error("输入数据为空")
            return None

        if not isinstance(data, np.ndarray) or data.ndim != 1:
            self.logger.error(f"数据必须是一维numpy数组，当前维度: {data.ndim}")
            return None

        # 性能优化：自动调整大数据的窗口大小和重叠比例
        data_length = len(data)
        # 对于超长数据，自动调整为更合理的参数
        if data_length > 500000:  # 50万以上数据点
            if window_size < 2048:
                window_size = 2048  # 增加窗口大小
            if overlap > 0.5:
                overlap = 0.5  # 减少重叠

        # 绘图前启用agg后端可以提高非交互式绘图性能
        import matplotlib

        prev_backend = matplotlib.get_backend()
        if not self._in_jupyter:
            matplotlib.use("agg")  # 非交互式后端，更快

        # 绘制时频图
        fig = self.spectrogram_plotter.plot(
            data,
            self.fs,
            window_size=window_size,
            overlap=overlap,
            title=title,
            freq_range=freq_range,
            time_range=time_range,
            cmap=cmap,
            ax=ax,
        )

        # 如果提供了文件名，则保存图片
        if file_name is not None:
            save_path = self.output_path / f"{file_name}.png"
            self.spectrogram_plotter.save_figure(save_path, close_after_save=False)

        # 还原后端
        if not self._in_jupyter:
            matplotlib.use(prev_backend)

        # 处理图形显示，避免双重显示
        return self._handle_figure_display(fig)

    def WaterfallPlot(
        self,
        data: np.ndarray,
        x_label: str = "Channels",
        y_label: str = "Time (s)",
        file_name: Optional[str] = None,
        title: Optional[str] = None,
        cmap: str = "viridis",
        colorbar_label: str = "Amplitude",
        aspect: str = "auto",
        origin: str = "upper",
        x_ticks: Optional[np.ndarray] = None,
        y_ticks: Optional[np.ndarray] = None,
        value_range: Optional[Tuple[float, float]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> Any:
        """绘制瀑布图（二维数据可视化）

        Args:
            data: 二维数组，形状为 (time_samples, channels) 或 (channels, time_samples)
            x_label: X轴标签
            y_label: Y轴标签
            file_name: 保存的文件名 (不含扩展名)，若提供则保存图片
            title: 自定义标题 (英文)
            cmap: 色彩映射名称
            colorbar_label: 颜色条标签
            aspect: 图像纵横比，'auto' 或 'equal'
            origin: 数据原点位置，'upper' 或 'lower'
            x_ticks: 可选，X轴刻度值
            y_ticks: 可选，Y轴刻度值
            value_range: 可选，数据值显示范围 (min_value, max_value)

        Returns:
            Figure: matplotlib 图形对象，可在 Jupyter notebook 中显示
        """
        # 数据验证
        if data is None or data.size == 0:
            self.logger.error("输入数据为空")
            return None

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            self.logger.error(f"数据必须是二维numpy数组，当前维度: {data.ndim}")
            return None

        # 绘图前启用agg后端可以提高非交互式绘图性能
        import matplotlib

        prev_backend = matplotlib.get_backend()
        if not self._in_jupyter:
            matplotlib.use("agg")  # 非交互式后端，更快

        # 绘制瀑布图
        fig = self.waterfall_plotter.plot(
            data=data,
            fs=self.fs,
            x_label=x_label,
            y_label=y_label,
            title=title,
            cmap=cmap,
            colorbar_label=colorbar_label,
            aspect=aspect,
            origin=origin,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            value_range=value_range,
            ax=ax,
        )

        # 如果提供了文件名，则保存图片
        if file_name is not None:
            save_path = self.output_path / f"{file_name}.png"
            self.waterfall_plotter.save_figure(save_path, close_after_save=False)

        # 还原后端
        if not self._in_jupyter:
            matplotlib.use(prev_backend)

        # 处理图形显示，避免双重显示
        return self._handle_figure_display(fig)

    def ProfilePlot(
        self,
        values: np.ndarray,
        distances: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        xlabel: str = "Channel",
        ylabel: str = "Amplitude",
        file_name: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ) -> Any:
        """绘制并保存剖面图（跨通道指标图）

        Args:
            values: 一维数组，包含每个通道的指标值
            distances: 可选，一维数组，包含每个通道对应的空间位置
            title: 自定义标题
            xlabel: X轴标签
            ylabel: Y轴标签
            file_name: 保存的文件名 (不含扩展名)，若提供则保存图片
            ax: 可选，指定的 matplotlib Axes 对象
            **kwargs: 传递给 ax.plot 的其他参数

        Returns:
            Figure: matplotlib 图形对象
        """
        fig = self.profile_plotter.plot(
            values=values,
            distances=distances,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            ax=ax,
            **kwargs,
        )

        if file_name is not None:
            save_path = self.output_path / f"{file_name}.png"
            self.profile_plotter.save_figure(save_path, close_after_save=False)

        return self._handle_figure_display(fig)
