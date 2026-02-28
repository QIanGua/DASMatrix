from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from .plot_base import PlotBase


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
