import logging
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
        import logging

        import matplotlib.pyplot as plt
        import seaborn as sns

        # 设置科学期刊级别的字体 - 使用 sans-serif 并配合 fallback 列表
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        font_family = "sans-serif"

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
        plt.rcParams.update(
            {
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
                "font.sans-serif": self.config.get_rcparams()["font.sans-serif"],
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
            }
        )

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

    def save_figure(self, file_path: Union[str, Path], close_after_save: bool = True) -> None:
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


class SpectrumPlot(PlotBase):
    """频谱图 (Spectrum Plot)"""

    def plot(
        self,
        frequencies: np.ndarray,
        magnitudes: np.ndarray,
        channel_id: Optional[Union[int, str]] = None,
        excitation_freq: Optional[float] = None,
        title: Optional[str] = None,
        peaks: Optional[List[Dict[str, float]]] = None,
        db_range: Optional[Tuple[float, float]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> Any:  # 返回类型改为 Any
        """绘制频谱图

        Args:
            frequencies: 频率数组 (Hz)
            magnitudes: 频谱幅值数组 (通常是对数尺度，如 dB)
            channel_id: 可选，通道标识
            excitation_freq: (可选) 激励频率，用于在图上标记 (Hz)
            title: 自定义标题 (英文), 若未指定则使用默认标题
            peaks: (可选) 峰值列表，每个峰值包含 'frequency' 和 'magnitude'
            db_range: 可选，dB值显示范围 (min_db, max_db)
            ax: 可选的 Axes 对象

        Returns:
            Figure: matplotlib 图形对象
        """

        # 性能优化：如果频率点数过多，进行降采样
        MAX_FREQ_POINTS = 10000
        if len(frequencies) > MAX_FREQ_POINTS:
            step = len(frequencies) // MAX_FREQ_POINTS
            frequencies = frequencies[::step]
            magnitudes = magnitudes[::step]

        # 创建图形和坐标轴 - 优化创建过程
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize_standard)
        else:
            fig = ax.figure

        # 性能优化：提前计算dB值，避免重复计算
        # 幅值通常以 dB 显示
        magnitudes_db = 10 * np.log10(magnitudes + 1e-12)  # 加一个小值避免 log(0)

        # 绘制频谱 - 使用更高性能的绘图设置
        ax.plot(
            frequencies,
            magnitudes_db,
            linewidth=1.0,  # 减小线宽以加快渲染
            color="#0072B2",
            alpha=1.0,
            zorder=5,
            rasterized=True,  # 光栅化以提高性能
        )

        # 标记峰值（如果提供）- 只在确实需要时执行
        if peaks and len(peaks) > 0:
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

            # 为前三个峰添加标签 - 仅处理一小部分峰值
            for i, (f, m) in enumerate(
                zip(
                    peak_freqs[: min(3, len(peak_freqs))],
                    peak_mags[: min(3, len(peak_mags))],
                )
            ):
                ax.annotate(
                    f"{f:.1f} Hz",
                    xy=(f, m),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
                )

        # 标记激励频率 - 只在确实需要时执行
        if excitation_freq is not None:
            # 以激励频率为中心，窗口大小取决于数据范围
            data_range = np.max(frequencies) - np.min(frequencies)
            window_size = min(data_range * 0.4, excitation_freq * 0.6)

            # 性能优化：避免多次调用set_xlim
            ax.set_xlim(max(0, excitation_freq - window_size), excitation_freq + window_size)

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

        # 设置网格 - 简化网格设置以提高性能
        ax.grid(True, which="major", linestyle="--", alpha=0.5, zorder=1)

        # 禁用次刻度以提高性能
        plt.minorticks_off()

        # 改善x轴刻度格式 - 简化刻度处理
        if max(frequencies) > 1000:
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{x / 1000:.0f}" if x >= 1000 else f"{x:.0f}")
            )
            ax.set_xlabel("Frequency (kHz)")
        else:
            ax.set_xlabel("Frequency (Hz)")

        ax.set_ylabel("Magnitude (dB)")

        # 图例设置（如果有需要的话）
        if (excitation_freq is not None or peaks) and ax.get_legend_handles_labels()[0]:
            leg = ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9, edgecolor="black")
            # 简化图例样式设置
            for line in leg.get_lines():
                line.set_linewidth(2.0)

        # 使用自定义标题或默认标题
        if title:
            plot_title = title
        elif excitation_freq and channel_id is not None:
            plot_title = f"Spectrum for Channel {channel_id} around {excitation_freq:.1f} Hz"
        elif excitation_freq:
            plot_title = f"Spectrum around {excitation_freq:.1f} Hz"
        elif channel_id is not None:
            plot_title = f"Spectrum for Channel {channel_id}"
        else:
            plot_title = "Spectrum"
        ax.set_title(plot_title, fontweight="bold")

        # 设置y轴范围
        if db_range is not None:
            # 如果提供了dB范围，直接使用
            ax.set_ylim(db_range)
        else:
            # 性能优化：直接使用数据范围设置y轴范围，避免额外计算
            ymin = np.min(magnitudes_db)
            ymax = np.max(magnitudes_db)
            y_range = ymax - ymin
            ax.set_ylim(ymin - 0.05 * y_range, ymax + 0.05 * y_range)

        # 自动调整布局
        fig.tight_layout()  # type: ignore

        return fig


class WaveformPlot(PlotBase):
    """时域波形图 (Time Domain Waveform Plot)"""

    def plot(
        self,
        amplitude_data: np.ndarray,
        fs: float,
        time_range: Optional[Tuple[float, float]] = None,
        amplitude_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        highlight_regions: Optional[List[Tuple[float, float]]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> Any:  # 返回类型改为 Any
        """绘制时域波形图

        Args:
            amplitude_data: 幅值数组
            fs: 采样频率 (Hz)
            amplitude_range: 可选，幅值显示范围 (min_amplitude, max_amplitude)
            time_range: 可选，时间显示范围 (start_time, end_time)
            title: 自定义标题 (英文), 若未指定则使用默认标题
            highlight_regions: 可选的高亮区域列表，每个元素为(开始时间, 结束时间)元组

        Returns:
            Figure: matplotlib 图形对象
        """
        # 延迟导入 matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # 根据采样率自动生成时间数组
        num_samples = len(amplitude_data)
        time_data = np.linspace(0, num_samples / fs, num_samples, endpoint=False)

        # 性能优化：对大数据进行降采样，只在时域绘图中使用
        # 对于大于10万点的数据进行智能降采样
        MAX_POINTS_TO_PLOT = 10000  # 最大绘制点数阈值，超过此值进行降采样

        # 如果指定了时间范围，先过滤数据
        if time_range is not None:
            start_time, end_time = time_range
            mask = (time_data >= start_time) & (time_data <= end_time)
            time_data = time_data[mask]
            amplitude_data = amplitude_data[mask]

        if len(amplitude_data) > MAX_POINTS_TO_PLOT:
            # 使用均匀降采样
            step = len(amplitude_data) // MAX_POINTS_TO_PLOT
            # 确保捕获最大和最小值，对可视化非常重要
            plot_time_data = time_data[::step]
            plot_amplitude_data = amplitude_data[::step]

            # 为了确保不错过极值点，在每一段中查找最大值和最小值点
            if step > 2:  # 只有当降采样比例较大时才进行极值保留
                max_vals = []
                min_vals = []
                max_times = []
                min_times = []

                for i in range(0, len(amplitude_data), step):
                    chunk = amplitude_data[i : i + step]
                    if len(chunk) > 0:
                        max_idx = np.argmax(chunk) + i
                        min_idx = np.argmin(chunk) + i
                        max_vals.append(amplitude_data[max_idx])
                        min_vals.append(amplitude_data[min_idx])
                        max_times.append(time_data[max_idx])
                        min_times.append(time_data[min_idx])

                # 合并均匀采样点和极值点
                plot_time_data = np.concatenate(
                    [
                        plot_time_data,
                        np.array(max_times),
                        np.array(min_times),
                    ]
                )
                plot_amplitude_data = np.concatenate(
                    [
                        plot_amplitude_data,
                        np.array(max_vals),
                        np.array(min_vals),
                    ]
                )

                # 按时间排序
                sort_idx = np.argsort(plot_time_data)
                plot_time_data = plot_time_data[sort_idx]
                plot_amplitude_data = plot_amplitude_data[sort_idx]
        else:
            plot_time_data = time_data
            plot_amplitude_data = amplitude_data

        # 创建具有黄金比例的图形 - 优化图形创建过程
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize_wide)
        else:
            fig = ax.figure

        # 绘制波形 - 直接绘制，无需计算统计信息
        # 性能优化：使用更高效的线条渲染，减少中间计算
        (line,) = ax.plot(
            plot_time_data,
            plot_amplitude_data,
            color="#0072B2",
            linewidth=1.0,
            alpha=0.9,
            zorder=5,
            rasterized=True,  # 光栅化可显著提高大量数据点的渲染性能
        )

        # 添加高亮区域（如果有）- 只有在确实需要时才执行
        if highlight_regions:
            for i, (start, end) in enumerate(highlight_regions):
                ax.axvspan(
                    start,
                    end,
                    alpha=0.2,
                    color=plt.cm.get_cmap("tab10")(i % 10),
                    zorder=2,
                )
                # 添加区域标签
                mid_point = (start + end) / 2
                ax.annotate(
                    f"Region {i + 1}",
                    xy=(mid_point, ax.get_ylim()[1] * 0.9),
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
                )

        # 设置网格 - 简化网格样式
        ax.grid(True, which="major", linestyle="--", alpha=0.4, zorder=1)

        # 计算并标记零线
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5, zorder=3)

        # 设置y轴范围
        if amplitude_range is not None:
            # 如果提供了幅值范围，直接使用
            ax.set_ylim(amplitude_range)
        else:
            # 性能优化：直接使用数据的最小最大值设置y轴范围，避免额外计算
            y_min, y_max = np.min(plot_amplitude_data), np.max(plot_amplitude_data)
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # 设置x轴范围（如果提供了time_range但没有过滤数据）
        if time_range is not None and not (time_data[0] >= time_range[0] and time_data[-1] <= time_range[1]):
            ax.set_xlim(time_range)

        # 优化坐标轴
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

        # 简化时间刻度格式
        if max(plot_time_data) - min(plot_time_data) > 60:
            # 超过60秒，以分:秒显示
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{int(x / 60)}:{int(x % 60):02d}" if x >= 60 else f"{int(x)}")
            )
            ax.set_xlabel("Time (min:sec)")

        # 使用自定义标题或默认标题
        if title:
            plot_title = title
        else:
            plot_title = "Time Domain Waveform"
        ax.set_title(plot_title, fontweight="bold")

        # 禁用不必要的刻度，提高性能
        plt.minorticks_off()

        # 自动调整布局
        fig.tight_layout()  # type: ignore

        return fig


class SpectrogramPlot(PlotBase):
    """时频图/频谱图 (STFT) 绘图类"""

    def plot(
        self,
        data: np.ndarray,
        fs: float,
        window_size: int = 1024,
        overlap: float = 0.75,
        title: Optional[str] = None,
        freq_range: Optional[Tuple[float, float]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        cmap: str = "inferno",
        ax: Optional[plt.Axes] = None,
    ) -> Any:
        """绘制时频谱图（短时傅里叶变换）

        Args:
            data: 时域信号数据
            fs: 采样频率 (Hz)
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

        # 性能优化：对大数据处理前进行降采样，降低STFT计算量
        # 对于特别大的数据，使用更大的窗口大小和更少的重叠来减少计算量
        MAX_DATA_POINTS = 100000  # 最大处理点数阈值

        # 针对长信号进行处理优化
        if len(data) > MAX_DATA_POINTS:
            # 如果信号长度超过阈值，增加窗口大小并减少重叠
            if window_size < 2048:
                window_size = 2048  # 使用更大的窗口
            if overlap > 0.5:
                overlap = 0.5  # 减少重叠，减少总窗口数量

        # 对于超大数据，可以先降采样再计算
        if len(data) > 1000000:  # 百万级以上数据点
            # 降采样因子
            downsample_factor = 4
            data = data[::downsample_factor]
            fs = fs / downsample_factor

        # 创建图形和坐标轴 - 优化创建过程
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize_standard)
        else:
            fig = ax.figure

        # 计算窗口重叠点数
        noverlap = int(window_size * overlap)

        # 性能优化：使用更高效的STFT计算参数
        # 计算STFT - 优化STFT计算参数
        f, t, Sxx = signal.spectrogram(
            data,
            fs=fs,
            window="hann",
            nperseg=window_size,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            mode="magnitude",  # 直接使用幅度而不是功率谱，减少一步计算
        )

        # 转换为对数尺度以增强可视化效果 (dB)
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        # 性能优化：使用固定的色标范围
        # 确定色标范围，增强对比度 - 使用固定范围而不是分位数计算
        # 使用简单的min-max加偏置，而不是计算分位数
        vmin = np.mean(Sxx_db) - 2 * np.std(Sxx_db)  # 均值减两个标准差
        vmax = np.mean(Sxx_db) + 2 * np.std(Sxx_db)  # 均值加两个标准差

        # 绘制时频谱 - 使用更高效的渲染设置
        im = ax.pcolormesh(
            t,
            f,
            Sxx_db,
            shading="auto",  # 使用auto而不是gouraud，更高效
            cmap=cmap,
            vmin=float(vmin),
            vmax=float(vmax),
            rasterized=True,  # 光栅化以提高性能
        )

        # 添加颜色条 - 简化颜色条设置
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Power/Frequency (dB/Hz)")

        # 性能优化：移除复杂的标记
        # 移除最大点标记，提高性能，只有在特别需要时才计算

        # 设置坐标轴范围（如果指定）
        if freq_range:
            ax.set_ylim(freq_range)
        else:
            # 默认显示低频区域（通常更有意义）
            max_display_freq = min(fs / 2, 5000) if fs > 10000 else fs / 2
            ax.set_ylim(0, max_display_freq)

        if time_range:
            ax.set_xlim(time_range)

        # 美化频率轴格式 - 简化格式设置
        if max(f) > 1000:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda y, pos: f"{y / 1000:.1f}" if y >= 1000 else f"{y:.0f}")
            )
            ax.set_ylabel("Frequency (kHz)")
        else:
            ax.set_ylabel("Frequency (Hz)")

        # 设置时间刻度格式 - 简化格式设置
        if max(t) > 60:
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{int(x / 60)}:{int(x % 60):02d}" if x >= 60 else f"{int(x)}")
            )
            ax.set_xlabel("Time (min:sec)")
        else:
            ax.set_xlabel("Time (s)")

        # 设置标题
        if title:
            plot_title = title
        else:
            plot_title = "Spectrogram"
        ax.set_title(plot_title, fontweight="bold")

        # 添加网格以增强可读性 - 禁用网格提高性能
        ax.grid(False)

        # 自动调整布局
        fig.tight_layout()  # type: ignore

        return fig


class WaterfallPlot(PlotBase):
    """瀑布图 (Waterfall Plot) 绘图类，用于可视化二维数据"""

    def plot(
        self,
        data: np.ndarray,
        fs: float,
        x_label: str = "Channels",
        y_label: str = "Time (s)",
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
            fs: 采样频率 (Hz)
            x_label: X轴标签
            y_label: Y轴标签
            title: 自定义标题 (英文)
            cmap: 色彩映射名称
            colorbar_label: 颜色条标签
            aspect: 图像纵横比，'auto' 或 'equal'
            origin: 数据原点位置，'upper' 或 'lower'
            x_ticks: 可选，X轴刻度值
            y_ticks: 可选，Y轴刻度值
            value_range: 可选，数据值显示范围 (min_value, max_value)

        Returns:
            Figure: matplotlib 图形对象
        """
        # 延迟导入 matplotlib
        import matplotlib.pyplot as plt

        # 创建图形和坐标轴
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize_wide)
        else:
            fig = ax.figure

        # 确定数据显示范围
        if value_range is None:
            # 自动确定合适的数据范围
            vmin = np.percentile(data, 5)  # 5%分位数作为最小值
            vmax = np.percentile(data, 95)  # 95%分位数作为最大值

            # 对称化范围，找出绝对值最大的范围
            abs_max = max(abs(vmin), abs(vmax))
            if np.min(data) < 0:  # 如果数据有负值，使用对称范围
                vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = value_range

        # 绘制瀑布图
        im = ax.imshow(
            data,
            cmap=cmap,
            aspect=aspect,  # type: ignore
            origin=origin,  # type: ignore
            interpolation="none",  # 禁用插值以保持原始数据的清晰度
            vmin=float(vmin),
            vmax=float(vmax),
        )

        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label(colorbar_label, fontsize=11)

        # 设置自定义刻度（如果提供）
        if x_ticks is not None:
            ax.set_xticks(np.arange(0, data.shape[1], max(1, data.shape[1] // 10)))

            # 如果提供了具体的x刻度值，使用它们
            if len(x_ticks) >= data.shape[1]:
                x_tick_labels = x_ticks[: data.shape[1]]
                # 减少显示的刻度数量，避免拥挤
                tick_indices = np.linspace(0, len(x_tick_labels) - 1, min(10, len(x_tick_labels)), dtype=int)
                ax.set_xticks(tick_indices)
                ax.set_xticklabels([f"{x_tick_labels[i]:.0f}" for i in tick_indices])

        if y_ticks is not None:
            ax.set_yticks(np.arange(0, data.shape[0], max(1, data.shape[0] // 10)))

            # 如果提供了具体的y刻度值，使用它们
            if len(y_ticks) >= data.shape[0]:
                y_tick_labels = y_ticks[: data.shape[0]]
                # 减少显示的刻度数量，避免拥挤
                tick_indices = np.linspace(0, len(y_tick_labels) - 1, min(10, len(y_tick_labels)), dtype=int)
                ax.set_yticks(tick_indices)
                ax.set_yticklabels([f"{y_tick_labels[i]:.1f}" for i in tick_indices])
        else:
            # 根据采样频率计算时间刻度
            if origin == "upper" or origin == "lower":
                # 计算时间轴的实际值（秒）
                time_values = np.arange(data.shape[0]) / fs

                # 设置合适的时间刻度数量
                num_ticks = min(10, data.shape[0])
                tick_indices = np.linspace(0, data.shape[0] - 1, num_ticks, dtype=int)

                ax.set_yticks(tick_indices)

                # 根据时间长度格式化标签
                if max(time_values) > 60:
                    # 如果时间超过60秒，使用分:秒格式
                    ax.set_yticklabels(
                        [f"{int(time_values[i] / 60)}:{int(time_values[i] % 60):02d}" for i in tick_indices]
                    )
                    y_label = "Time (min:sec)"
                else:
                    # 否则使用秒格式
                    ax.set_yticklabels([f"{time_values[i]:.1f}" for i in tick_indices])
                    y_label = "Time (s)"

        # 设置标签
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

        # 设置标题
        if title:
            plot_title = title
        else:
            plot_title = "Waterfall Plot"
        ax.set_title(plot_title, fontweight="bold", fontsize=14)

        # 优化网格和刻度
        ax.grid(False)  # 对于imshow类型的图，通常不需要网格

        # 自动调整布局
        fig.tight_layout()  # type: ignore

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


class FKPlot(PlotBase):
    """F-K 谱图 (F-K Spectrum Plot) 绘图类"""

    def plot(
        self,
        fk_spectrum: np.ndarray,
        freqs: np.ndarray,
        wavenumbers: np.ndarray,
        title: Optional[str] = None,
        cmap: str = "turbo",
        db_range: Optional[Tuple[float, float]] = None,
        v_lines: Optional[List[float]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> Any:
        """绘制 F-K 谱图

        Args:
            fk_spectrum: FK 谱 (complex)
            freqs: 频率轴
            wavenumbers: 波数轴
            title: 标题
            cmap: 色彩映射
            db_range: 显示范围 (min_db, max_db)
            v_lines: 待标记的速度线 (m/s)
            ax: 可选的 Axes 对象

        Returns:
            Figure: matplotlib 图形对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize_standard)
        else:
            fig = ax.figure

        # 计算幅度谱 (dB)
        mag = np.abs(fk_spectrum)
        mag_db = 20 * np.log10(mag + 1e-12)

        # 归一化 dB
        mag_db = mag_db - np.max(mag_db)

        # 确定显示范围
        if db_range:
            vmin, vmax = db_range
        else:
            vmax = 0
            vmin = -60  # 默认显示前 60dB

        # 绘制
        # 使用 pcolormesh 或 imshow
        # 注意：imshow 需要 origin='lower' 且 extent 正确
        extent = (
            float(wavenumbers[0]),
            float(wavenumbers[-1]),
            float(freqs[0]),
            float(freqs[-1]),
        )

        im = ax.imshow(
            mag_db,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Amplitude (dB)")

        ax.set_xlabel("Wavenumber (1/m)")
        ax.set_ylabel("Frequency (Hz)")

        if title:
            ax.set_title(title, fontweight="bold")
        else:
            ax.set_title("F-K Spectrum", fontweight="bold")

        # 绘制速度线 f = v * k
        if v_lines:
            k_min, k_max = wavenumbers[0], wavenumbers[-1]
            f_min, f_max = freqs[0], freqs[-1]

            k_line = np.linspace(k_min, k_max, 100)

            for v in v_lines:
                f_line = v * k_line

                # 只绘制在频率范围内的部分
                mask = (f_line >= f_min) & (f_line <= f_max)
                if np.any(mask):
                    ax.plot(k_line[mask], f_line[mask], "w--", alpha=0.7, linewidth=1)
                    # 标注速度
                    # 找一个合适的位置标注
                    idx = len(k_line) // 2
                    if mask[idx]:
                        ax.text(
                            k_line[idx],
                            f_line[idx],
                            f"{v} m/s",
                            color="white",
                            fontsize=8,
                        )

        fig.tight_layout()  # type: ignore
        return fig


class ProfilePlot(PlotBase):
    """剖面图 (Profile Plot) 绘图类，用于展示沿光缆分布的指标
    （如 RMS、均值、最大值等）
    """

    def plot(
        self,
        values: np.ndarray,
        distances: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        xlabel: str = "Channel",
        ylabel: str = "Amplitude",
        label: Optional[str] = None,
        color: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ) -> plt.Figure:
        """绘制剖面图

        Args:
            values: 1D 数组，包含每个通道的指标值
            distances: 可选，空间距离轴（米）
            title: 图标题
            xlabel: X轴标签
            ylabel: Y轴标签
            label: 图例标签
            color: 线条颜色
            ax: 可选，指定的 matplotlib Axes 对象
            **kwargs: 传递给 ax.plot 的其他参数
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            if ax.figure is None:
                raise ValueError("Provided ax must belong to a figure")
            from typing import cast

            fig = cast(plt.Figure, ax.figure)

        if distances is None:
            distances = np.arange(len(values))
            if xlabel == "Channel":
                xlabel = "Channel Index"

        # 使用点线图展示，这是 DAS 剖面图的标准做法
        ax.plot(
            distances,
            values,
            "o-",
            markersize=3,
            linewidth=1,
            label=label,
            color=color,
            **kwargs,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        ax.grid(True, linestyle="--", alpha=0.5)

        if label:
            ax.legend()

        fig.tight_layout()
        return fig
