import logging
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..config.visualization_config import VisualizationConfig


class PlotBase(ABC):
    """绘图基类"""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """初始化绘图基类

        Args:
            config: 可视化配置，若未指定则使用默认配置
        """
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
        self._setup_style()

    def _setup_style(self) -> None:
        """设置绘图风格"""
        # 延迟导入 matplotlib 和 seaborn
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 设置科学期刊级别的字体
        font_family = "Arial"  # Nature/Science推荐字体

        # 设置专业科学绘图风格 - 顶级期刊标准
        plt.style.use("default")

        # 使用高对比度色盲友好配色 - 参考Nature配色指南
        colors = [
            "#0072B2",
            "#D55E00",
            "#009E73",
            "#CC79A7",
            "#E69F00",
            "#56B4E9",
            "#F0E442",
            "#000000",
            "#0072B2",
            "#D55E00",
        ]
        sns.set_palette(colors)

        # 应用通用样式设置 - Nature/Science风格
        plt.rcParams.update({
            "figure.dpi": self.config.dpi,
            "figure.figsize": self.config.figsize_standard,
            "figure.facecolor": "white",
            "figure.autolayout": True,  # 自动处理布局
            # 坐标轴设置 - 封闭边框
            "axes.grid": self.config.grid,
            "axes.linewidth": 1.0,  # 边框线宽
            "axes.edgecolor": "black",  # 边框颜色
            "axes.spines.top": True,  # 添加顶部边框
            "axes.spines.right": True,  # 添加右侧边框
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.axisbelow": True,  # 网格线在数据下方
            "axes.facecolor": "white",
            "axes.labelcolor": "black",
            "axes.prop_cycle": plt.cycler("color", colors),
            # 网格线设置
            "grid.color": "#E0E0E0",
            "grid.linestyle": self.config.grid_style,
            "grid.alpha": 0.7,
            "grid.linewidth": 0.5,
            # 线条设置
            "lines.linewidth": 1.5,
            "lines.markeredgewidth": 1.0,
            "lines.markersize": 4,
            "lines.solid_capstyle": "round",
            # 字体设置 - 期刊标准
            "font.family": font_family,
            "font.size": 11,
            "font.weight": "normal",
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",  # 加粗标题
            "axes.labelweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.unicode_minus": False,  # 解决负号显示问题
            # 刻度线设置 - 内向刻度
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.top": True,  # 顶部刻度显示
            "ytick.right": True,  # 右侧刻度显示
            # 图例设置
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.fancybox": False,  # 直角边框
            "legend.facecolor": "white",
            "legend.edgecolor": "black",
            # 保存设置
            "savefig.dpi": self.config.dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "savefig.transparent": False,
            # 强制显示所有边框和刻度线
            "axes.autolimit_mode": "data",
            "axes.xmargin": 0.05,
            "axes.ymargin": 0.05,
        })

        # 设置默认的图形，确保边框和刻度线设置正确
        # 避免使用回调函数，直接设置默认样式
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # 确保所有边框可见
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(1.0)
            ax.spines[spine].set_color("black")

        # 确保刻度线在所有边上显示
        ax.tick_params(top=True, right=True, which="both", direction="in")

        # 关闭这个示例图形，避免影响后续绘图
        plt.close(fig)

    def save_figure(
        self, file_path: Union[str, Path], close_after_save: bool = True
    ) -> None:
        """保存图形到指定路径

        Args:
            file_path: 图形保存路径
            close_after_save: 保存后是否关闭图形
        """
        # 延迟导入 matplotlib
        import matplotlib.pyplot as plt

        try:
            # 确保目录存在
            if isinstance(file_path, str):
                file_path = Path(file_path)

            file_path.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(
                file_path,
                bbox_inches="tight",
                dpi=self.config.dpi,
            )
            self.logger.debug(f"Figure saved to {file_path}")

            if close_after_save:
                plt.close()
        except Exception as e:
            self.logger.error(f"Error saving figure: {e}")
            if close_after_save:
                plt.close()

    def _add_watermark(self, fig, ax=None):
        """添加水印或标识（可选）

        Args:
            fig: matplotlib Figure对象
            ax: 可选，特定的Axes对象
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        # 可以在此添加水印或标识（比如项目名称），如果需要的话
        # ax.text(0.99, 0.01, 'DASMatrix', transform=ax.transAxes,
        #        fontsize=7, alpha=0.3, ha='right', va='bottom')
        pass


class FrequencyResponsePlot(PlotBase):
    """频率响应图 (Frequency Response Plot)"""

    def plot(
        self,
        freq_list: np.ndarray,
        response_list: np.ndarray,
        site: Union[int, str],
        title: Optional[str] = None,
    ) -> Any:  # 返回类型改为 Any，避免直接引用 plt.Figure
        """绘制频率响应图

        Args:
            freq_list: 频率数组 (Hz)
            response_list: 响应幅度数组 (通常是 dB)
            site: 点位标识 (Channel ID)
            title: 自定义标题 (英文), 若未指定则使用默认标题

        Returns:
            Figure: matplotlib 图形对象
        """
        # 延迟导入 matplotlib
        import matplotlib.pyplot as plt

        # 创建图形
        fig, ax = plt.subplots(figsize=self.config.figsize_standard)

        # 绘制主要数据
        ax.semilogx(freq_list, response_list, "o-", markersize=3, alpha=0.8)

        # 设置网格
        ax.grid(True, which="both", linestyle="--", alpha=0.7)
        ax.grid(True, which="minor", alpha=0.3)

        # 设置标签
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Response Magnitude (dB)")

        # 使用自定义标题或默认标题
        plot_title = title if title else f"Frequency Response for Channel {site}"
        ax.set_title(plot_title)

        # 优化刻度
        ax.minorticks_on()

        # 自动调整布局
        fig.tight_layout()

        # 添加可选水印
        self._add_watermark(fig, ax)

        return fig


class SpectrumPlot(PlotBase):
    """频谱图 (Spectrum Plot)"""

    def plot(
        self,
        frequencies: np.ndarray,
        magnitudes: np.ndarray,
        channel_id: Union[int, str],
        excitation_freq: Optional[float] = None,
        title: Optional[str] = None,
        peaks: Optional[List[Dict[str, float]]] = None,
    ) -> Any:  # 返回类型改为 Any
        """绘制频谱图

        Args:
            frequencies: 频率数组 (Hz)
            magnitudes: 频谱幅值数组 (通常是对数尺度，如 dB)
            channel_id: 通道标识
            excitation_freq: (可选) 激励频率，用于在图上标记 (Hz)
            title: 自定义标题 (英文), 若未指定则使用默认标题
            peaks: (可选) 峰值列表，每个峰值包含 'frequency' 和 'magnitude'

        Returns:
            Figure: matplotlib 图形对象
        """
        # 延迟导入 matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=self.config.figsize_standard)

        # 幅值通常以 dB 显示
        magnitudes_db = 10 * np.log10(magnitudes + 1e-12)  # 加一个小值避免 log(0)

        # 绘制频谱 - 使用更高质量的线条渲染
        ax.plot(
            frequencies,
            magnitudes_db,
            linewidth=1.5,
            color="#0072B2",
            alpha=1.0,
            zorder=5,
        )

        # 标记峰值（如果提供）
        if peaks:
            peak_freqs = [p["frequency"] for p in peaks]
            peak_mags = [10 * np.log10(p["magnitude"] + 1e-12) for p in peaks]
            ax.plot(
                peak_freqs,
                peak_mags,
                "o",
                color="#D55E00",
                markersize=5,
                markeredgewidth=1.0,
                markeredgecolor="black",
                alpha=1.0,
                zorder=10,
                label="Peak Frequencies",
            )

            # 为前三个峰添加标签
            for i, (f, m) in enumerate(zip(peak_freqs[:3], peak_mags[:3])):
                ax.annotate(
                    f"{f:.1f} Hz",
                    xy=(f, m),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9
                    ),
                )

        # 标记激励频率
        if excitation_freq is not None:
            # 以激励频率为中心，窗口大小取决于数据范围
            data_range = np.max(frequencies) - np.min(frequencies)
            window_size = min(data_range * 0.4, excitation_freq * 0.6)

            ax.set_xlim(
                max(0, excitation_freq - window_size), excitation_freq + window_size
            )

            # 添加垂直线标记激励频率
            ax.axvline(
                excitation_freq,
                color="#D55E00",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                zorder=6,
                label=f"Excitation: {excitation_freq:.1f} Hz",
            )

        # 设置网格 - 精细调整网格样式
        ax.grid(True, which="major", linestyle="--", alpha=0.5, zorder=1)
        ax.grid(True, which="minor", alpha=0.25, linestyle=":", zorder=1)

        # 启用次刻度
        ax.minorticks_on()

        # 改善x轴刻度格式 - 使用科学记数法避免大数字拥挤
        if max(frequencies) > 1000:
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(
                    lambda x, pos: f"{x / 1000:.0f}" if x >= 1000 else f"{x:.0f}"
                )
            )
            ax.set_xlabel("Frequency (kHz)")
        else:
            ax.set_xlabel("Frequency (Hz)")

        ax.set_ylabel("Magnitude (dB)")

        # 图例设置（如果有需要的话）
        if excitation_freq is not None or peaks:
            leg = ax.legend(
                loc="best", frameon=True, framealpha=0.9, fontsize=9, edgecolor="black"
            )
            # 改善图例外观
            for line in leg.get_lines():
                line.set_linewidth(2.0)

        # 使用自定义标题或默认标题
        if title:
            plot_title = title
        elif excitation_freq:
            plot_title = (
                f"Spectrum for Channel {channel_id} around {excitation_freq:.1f} Hz"
            )
        else:
            plot_title = f"Spectrum for Channel {channel_id}"
        ax.set_title(plot_title, fontweight="bold")

        # 精确设置y轴范围 - 修复使用np.min和np.max避免数组操作错误
        ymin = np.min(magnitudes_db) - 3
        ymax = np.max(magnitudes_db) + 3
        ax.set_ylim(ymin, ymax)

        # 自动调整布局
        fig.tight_layout()

        # 添加可选水印
        self._add_watermark(fig, ax)

        return fig


class WaveformPlot(PlotBase):
    """时域波形图 (Time Domain Waveform Plot)"""

    def plot(
        self,
        time_data: np.ndarray,
        amplitude_data: np.ndarray,
        channel_id: Union[int, str],
        title: Optional[str] = None,
        highlight_regions: Optional[List[Tuple[float, float]]] = None,
    ) -> Any:  # 返回类型改为 Any
        """绘制时域波形图

        Args:
            time_data: 时间数组 (s)
            amplitude_data: 幅值数组
            channel_id: 通道标识
            title: 自定义标题 (英文), 若未指定则使用默认标题
            highlight_regions: 可选的高亮区域列表，每个元素为(开始时间, 结束时间)元组

        Returns:
            Figure: matplotlib 图形对象
        """
        # 延迟导入 matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # 创建具有黄金比例的图形
        fig, ax = plt.subplots(figsize=self.config.figsize_wide)

        # 计算数据统计信息，用于优化显示
        mean_val = np.mean(amplitude_data)
        std_val = np.std(amplitude_data)
        abs_max = np.max(np.abs(amplitude_data))

        # 绘制波形 - 高质量线条
        ax.plot(
            time_data,
            amplitude_data,
            color="#0072B2",
            linewidth=1.0,
            alpha=0.9,
            zorder=5,
        )

        # 添加高亮区域（如果有）
        if highlight_regions:
            for i, (start, end) in enumerate(highlight_regions):
                ax.axvspan(start, end, alpha=0.2, color=plt.cm.tab10(i % 10), zorder=2)
                # 添加区域标签
                mid_point = (start + end) / 2
                ax.annotate(
                    f"Region {i + 1}",
                    xy=(mid_point, ax.get_ylim()[1] * 0.9),
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8
                    ),
                )

        # 设置网格 - 科学期刊风格
        ax.grid(True, which="major", linestyle="--", alpha=0.4, zorder=1)
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.2, linestyle=":", zorder=1)

        # 计算并标记零线
        ax.axhline(
            y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5, zorder=3
        )

        # 优化坐标轴范围
        y_margin = max(0.1 * abs_max, 2 * std_val)
        ax.set_ylim(min(amplitude_data) - y_margin, max(amplitude_data) + y_margin)

        # 优化坐标轴
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

        # 如果时间轴较长，使用适当的时间刻度格式
        if max(time_data) - min(time_data) > 60:
            # 超过60秒，以分:秒显示
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(
                    lambda x, pos: f"{int(x / 60)}:{int(x % 60):02d}"
                    if x >= 60
                    else f"{int(x)}"
                )
            )
            ax.set_xlabel("Time (min:sec)")

        # 使用自定义标题或默认标题
        plot_title = (
            title if title else f"Time Domain Waveform for Channel {channel_id}"
        )
        ax.set_title(plot_title, fontweight="bold")

        # 自动调整布局
        fig.tight_layout()

        # 添加可选水印
        self._add_watermark(fig, ax)

        return fig


class SpectrogramPlot(PlotBase):
    """时频图/频谱图 (STFT) 绘图类"""

    def plot(
        self,
        data: np.ndarray,
        fs: float,
        channel_id: Union[int, str],
        window_size: int = 1024,
        overlap: float = 0.75,
        title: Optional[str] = None,
        freq_range: Optional[Tuple[float, float]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        cmap: str = "inferno",
    ) -> Any:
        """绘制时频谱图（短时傅里叶变换）

        Args:
            data: 时域信号数据
            fs: 采样频率 (Hz)
            channel_id: 通道标识
            window_size: STFT窗口大小
            overlap: 窗口重叠比例
            title: 自定义标题
            freq_range: 可选，频率范围 (min_freq, max_freq)
            time_range: 可选，时间范围 (start_time, end_time)
            cmap: 色彩映射名称

        Returns:
            Figure: matplotlib 图形对象
        """
        # 延迟导入必要的库
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        from scipy import signal

        # 创建科学出版物级别的图形和坐标轴
        fig, ax = plt.subplots(figsize=self.config.figsize_standard)

        # 计算窗口重叠点数
        noverlap = int(window_size * overlap)

        # 计算STFT
        f, t, Sxx = signal.spectrogram(
            data,
            fs=fs,
            window="hann",
            nperseg=window_size,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
        )

        # 转换为对数尺度以增强可视化效果 (dB)
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        # 确定色标范围，增强对比度 - Nature级别的阈值优化
        vmin = np.percentile(Sxx_db, 2)  # 2%分位数作为最小值
        vmax = np.percentile(Sxx_db, 98)  # 98%分位数作为最大值

        # 绘制时频谱 - 高品质渲染
        im = ax.pcolormesh(
            t, f, Sxx_db, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax
        )

        # 添加科学出版物级别的颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Power/Frequency (dB/Hz)", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        # 添加主要特征标记 - 增强可读性
        if Sxx_db.size > 0:
            # 找出强度最大的点，标记关键特征
            max_idx = np.unravel_index(np.argmax(Sxx_db), Sxx_db.shape)
            max_t, max_f = t[max_idx[1]], f[max_idx[0]]
            max_power = Sxx_db[max_idx]

            # 标记最强点
            ax.plot(
                max_t,
                max_f,
                "o",
                color="white",
                markersize=5,
                markeredgecolor="black",
                markeredgewidth=1,
                alpha=1.0,
                zorder=10,
            )
            ax.annotate(
                f"Max: {max_f:.1f} Hz",
                xy=(max_t, max_f),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", alpha=0.7),
            )

        # 设置坐标轴范围（如果指定）
        if freq_range:
            ax.set_ylim(freq_range)
        else:
            # 默认显示低频区域（通常更有意义）
            max_display_freq = min(fs / 2, 5000) if fs > 10000 else fs / 2
            ax.set_ylim(0, max_display_freq)

        if time_range:
            ax.set_xlim(time_range)

        # 美化频率轴格式
        if max(f) > 1000:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(
                    lambda y, pos: f"{y / 1000:.1f}" if y >= 1000 else f"{y:.0f}"
                )
            )
            ax.set_ylabel("Frequency (kHz)")
        else:
            ax.set_ylabel("Frequency (Hz)")

        # 设置时间刻度格式
        if max(t) > 60:
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(
                    lambda x, pos: f"{int(x / 60)}:{int(x % 60):02d}"
                    if x >= 60
                    else f"{int(x)}"
                )
            )
            ax.set_xlabel("Time (min:sec)")
        else:
            ax.set_xlabel("Time (s)")

        # 设置标题
        if title:
            plot_title = title
        else:
            plot_title = f"Spectrogram for Channel {channel_id}"
        ax.set_title(plot_title, fontweight="bold")

        # 添加网格以增强可读性
        ax.grid(False)  # 关闭默认网格

        # 自动调整布局
        fig.tight_layout()

        # 添加可选水印
        self._add_watermark(fig, ax)

        return fig


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
        self.output_path = (
            Path(output_path) if isinstance(output_path, str) else output_path
        )
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.fs = sampling_frequency
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)

        # 初始化绘图器实例
        self.freq_resp_plotter = FrequencyResponsePlot(self.config)
        self.spectrum_plotter = SpectrumPlot(self.config)
        self.waveform_plotter = WaveformPlot(self.config)
        self.spectrogram_plotter = SpectrogramPlot(self.config)

    def plot_frequency_response(
        self,
        freq_list: np.ndarray,
        response_list: np.ndarray,
        site: Union[int, str],
        file_name: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """绘制并保存频率响应图

        Args:
            freq_list: 频率数组 (Hz)
            response_list: 响应幅度数组 (dB)
            site: 点位标识 (Channel ID)
            file_name: 保存的文件名 (不含扩展名)，若未指定则自动生成
            title: 图表标题 (英文)
        """
        fig = self.freq_resp_plotter.plot(freq_list, response_list, site, title)
        if file_name is None:
            file_name = f"freq_response_channel_{site}"
        save_path = self.output_path / f"{file_name}.png"
        self.freq_resp_plotter.save_figure(save_path)

    def plot_spectrum(
        self,
        frequencies: np.ndarray,
        magnitudes: np.ndarray,
        channel_id: Union[int, str],
        excitation_freq: Optional[float] = None,
        file_name: Optional[str] = None,
        title: Optional[str] = None,
        peaks: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """绘制并保存频谱图

        Args:
            frequencies: 频率数组 (Hz)
            magnitudes: 幅值数组
            channel_id: 通道标识
            excitation_freq: (可选) 激励频率 (Hz)
            file_name: 保存的文件名 (不含扩展名)，若未指定则自动生成
            title: 图表标题 (英文)
            peaks: (可选) 峰值列表，每个峰值包含 'frequency' 和 'magnitude'
        """
        fig = self.spectrum_plotter.plot(
            frequencies, magnitudes, channel_id, excitation_freq, title, peaks
        )
        if file_name is None:
            freq_suffix = f"_{excitation_freq:.0f}Hz" if excitation_freq else ""
            file_name = f"spectrum_channel_{channel_id}{freq_suffix}"
        save_path = self.output_path / f"{file_name}.png"
        self.spectrum_plotter.save_figure(save_path)

    def plot_waveform(
        self,
        data: np.ndarray,
        channel_id: Union[int, str],
        time_vector: Optional[np.ndarray] = None,
        file_name: Optional[str] = None,
        title: Optional[str] = None,
        highlight_regions: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """绘制并保存时域波形图

        Args:
            data: 包含所有通道数据的数组 (时间轴在 axis=0)
            channel_id: 要绘制的通道标识
            time_vector: (可选) 时间轴向量 (s)，若未提供则根据采样率生成
            file_name: 保存的文件名 (不含扩展名)，若未指定则自动生成
            title: 图表标题 (英文)
            highlight_regions: 可选的高亮区域列表，每个元素为(开始时间, 结束时间)元组
        """
        if channel_id < 0 or channel_id >= data.shape[1]:
            self.logger.error(f"无效的通道 ID: {channel_id}")
            return

        amplitude_data = data[:, channel_id]

        if time_vector is None:
            num_samples = len(amplitude_data)
            time_data = np.arange(num_samples) / self.fs
        else:
            time_data = time_vector

        fig = self.waveform_plotter.plot(
            time_data, amplitude_data, channel_id, title, highlight_regions
        )

        if file_name is None:
            file_name = f"waveform_channel_{channel_id}"
        save_path = self.output_path / f"{file_name}.png"
        self.waveform_plotter.save_figure(save_path)

    def plot_spectrogram(
        self,
        data: np.ndarray,
        channel_id: Union[int, str],
        window_size: int = 1024,
        overlap: float = 0.75,
        file_name: Optional[str] = None,
        title: Optional[str] = None,
        freq_range: Optional[Tuple[float, float]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        cmap: str = "viridis",
    ) -> None:
        """绘制并保存时频图（短时傅里叶变换）

        Args:
            data: 包含所有通道数据的数组 (时间轴在 axis=0)
            channel_id: 要绘制的通道标识
            window_size: STFT窗口大小
            overlap: 窗口重叠比例
            file_name: 保存的文件名 (不含扩展名)，若未指定则自动生成
            title: 图表标题 (英文)
            freq_range: 可选，频率范围 (min_freq, max_freq)
            time_range: 可选，时间范围 (start_time, end_time)
            cmap: 色彩映射名称
        """
        if channel_id < 0 or channel_id >= data.shape[1]:
            self.logger.error(f"无效的通道 ID: {channel_id}")
            return

        # 提取指定通道的数据
        channel_data = data[:, channel_id]

        # 绘制时频图
        fig = self.spectrogram_plotter.plot(
            channel_data,
            self.fs,
            channel_id,
            window_size,
            overlap,
            title,
            freq_range,
            time_range,
            cmap,
        )

        # 保存图形
        if file_name is None:
            file_name = f"spectrogram_channel_{channel_id}"
        save_path = self.output_path / f"{file_name}.png"
        self.spectrogram_plotter.save_figure(save_path)
