from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from ..core.computation_graph import ComputationGraph


class DASFrame:
    """DAS数据处理核心类，提供信号处理API。

    包含时域处理、频域变换、检测等功能，支持惰性计算。
    """

    def __init__(
        self,
        data: np.ndarray,
        fs: float,
        graph: Optional[ComputationGraph] = None,
        is_stream: bool = False,
        **metadata,
    ):
        """初始化DASFrame对象。

        Args:
            data: DAS数据数组，形状为(时间点数, 通道数)
            fs: 采样频率(Hz)
            graph: 计算图对象，用于惰性计算
            is_stream: 是否为流式数据
            **metadata: 元数据
        """
        self._data = data
        self._fs = fs
        self._graph = graph or ComputationGraph.leaf(data)
        self._is_stream = is_stream
        self._metadata = metadata

    # --- 基础操作 ---
    def slice(self, t=slice(None), x=slice(None)):  # 时间和通道切片
        """切片数据。

        Args:
            t: 时间轴切片
            x: 通道轴切片

        Returns:
            DASFrame: 切片后的DASFrame对象
        """

        def _slice(data):
            return data[t, x]

        return self._return(_slice, name="slice")

    # --- 时域 ---
    def detrend(self, axis="time"):
        """去除趋势。

        Args:
            axis: 时间轴/通道轴

        Returns:
            DASFrame: 处理后的DASFrame对象
        """

        def _detrend(data):
            _axis = 0 if axis == "time" else 1
            return signal.detrend(data, axis=_axis)

        return self._return(_detrend, name="detrend")

    def normalize(self, method="zscore"):
        """归一化。

        Args:
            method: 归一化方法, 'zscore', 'minmax'

        Returns:
            DASFrame: 处理后的DASFrame对象
        """

        def _normalize(data):
            if method == "zscore":
                # Z-score归一化
                mean = np.mean(data, axis=0, keepdims=True)
                std = np.std(data, axis=0, keepdims=True)
                return (data - mean) / std
            elif method == "minmax":
                # Min-max归一化
                min_val = np.min(data, axis=0, keepdims=True)
                max_val = np.max(data, axis=0, keepdims=True)
                return (data - min_val) / (max_val - min_val)
            else:
                raise ValueError(f"不支持的归一化方法: {method}")

        return self._return(_normalize, name="normalize")

    def demean(self, axis="time"):
        """去均值。

        Args:
            axis: 时间轴/通道轴

        Returns:
            DASFrame: 处理后的DASFrame对象
        """

        def _demean(data):
            _axis = 0 if axis == "time" else 1
            return data - np.mean(data, axis=_axis, keepdims=True)

        return self._return(_demean, name="demean")

    def bandpass(self, low, high, order=4, fs=None, design="butter"):
        """带通滤波。

        Args:
            low: 低频截止(Hz)
            high: 高频截止(Hz)
            order: 滤波器阶数
            fs: 采样频率(Hz)，默认使用对象的fs
            design: 滤波器设计类型

        Returns:
            DASFrame: 处理后的DASFrame对象
        """
        if fs is None:
            fs = self._fs

        def _bandpass(data):
            sos = signal.butter(
                order, [low, high], btype="bandpass", fs=fs, output="sos"
            )
            return signal.sosfiltfilt(sos, data, axis=0)

        return self._return(_bandpass, name="bandpass")

    def lowpass(self, cutoff, order=4, fs=None):
        """低通滤波。

        Args:
            cutoff: 截止频率(Hz)
            order: 滤波器阶数
            fs: 采样频率(Hz)，默认使用对象的fs

        Returns:
            DASFrame: 处理后的DASFrame对象
        """
        if fs is None:
            fs = self._fs

        def _lowpass(data):
            sos = signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
            return signal.sosfiltfilt(sos, data, axis=0)

        return self._return(_lowpass, name="lowpass")

    def highpass(self, cutoff, order=4, fs=None):
        """高通滤波。

        Args:
            cutoff: 截止频率(Hz)
            order: 滤波器阶数
            fs: 采样频率(Hz)，默认使用对象的fs

        Returns:
            DASFrame: 处理后的DASFrame对象
        """
        if fs is None:
            fs = self._fs

        def _highpass(data):
            sos = signal.butter(order, cutoff, btype="highpass", fs=fs, output="sos")
            return signal.sosfiltfilt(sos, data, axis=0)

        return self._return(_highpass, name="highpass")

    def notch(self, freq, Q=30, fs=None):
        """陷波滤波。

        Args:
            freq: 陷波频率(Hz)
            Q: 品质因数
            fs: 采样频率(Hz)，默认使用对象的fs

        Returns:
            DASFrame: 处理后的DASFrame对象
        """
        if fs is None:
            fs = self._fs

        def _notch(data):
            b, a = signal.iirnotch(freq, Q, fs)
            return signal.filtfilt(b, a, data, axis=0)

        return self._return(_notch, name="notch")

    def median_filter(self, k=5, axis="time"):
        """中值滤波。

        Args:
            k: 滤波器窗口大小
            axis: 时间轴/通道轴

        Returns:
            DASFrame: 处理后的DASFrame对象
        """

        def _median_filter(data):
            _axis = 0 if axis == "time" else 1
            return signal.medfilt(data, kernel_size=[k, 1] if _axis == 0 else [1, k])

        return self._return(_median_filter, name="median_filter")

    # --- 变换 ---
    def fft(self, n=None, axis="time"):
        """快速傅里叶变换。

        Args:
            n: FFT点数，默认为轴长度
            axis: 时间轴/通道轴

        Returns:
            DASFrame: 转换后的DASFrame对象
        """

        def _fft(data):
            _axis = 0 if axis == "time" else 1
            return np.abs(np.fft.fft(data, n=n, axis=_axis))

        return self._return(_fft, name="fft")

    def ifft(self, axis="time"):
        """逆傅里叶变换。

        Args:
            axis: 时间轴/通道轴

        Returns:
            DASFrame: 转换后的DASFrame对象
        """

        def _ifft(data):
            _axis = 0 if axis == "time" else 1
            return np.real(np.fft.ifft(data, axis=_axis))

        return self._return(_ifft, name="ifft")

    def stft(self, nperseg, noverlap=None, window="hann"):
        """短时傅里叶变换。

        Args:
            nperseg: 窗口长度
            noverlap: 重叠长度，默认为nperseg//2
            window: 窗口类型

        Returns:
            DASFrame: 转换后的DASFrame对象，三维数组(时间,频率,通道)
        """
        fs = self._fs

        def _stft(data):
            # 对每个通道单独计算STFT
            _f, _t, stft_list = [], [], []
            for ch in range(data.shape[1]):
                channel_data = data[:, ch]
                f_ch, t_ch, zxx = signal.stft(
                    channel_data,
                    fs=fs,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap,
                )
                _f, _t = f_ch, t_ch
                stft_list.append(np.abs(zxx))

            # 将所有通道的STFT结果合并为三维数组
            stft_array = np.stack(stft_list, axis=-1)
            return stft_array

        return self._return(_stft, name="stft")

    def wavelet(self, wavelet="morl", scales=None):
        """小波变换。

        Args:
            wavelet: 小波类型
            scales: 尺度列表，默认为None (自动创建)

        Returns:
            DASFrame: 转换后的DASFrame对象
        """
        try:
            import pywt
        except ImportError:
            raise ImportError("需要安装pywt库: pip install PyWavelets")

        def _wavelet(data):
            # 对每个通道单独计算小波变换
            coeffs_list = []
            for ch in range(data.shape[1]):
                channel_data = data[:, ch]
                if scales is None:
                    # 自动创建尺度
                    width = int(len(channel_data) / 10)
                    _scales = np.arange(1, width)
                else:
                    _scales = scales

                coeffs = pywt.cwt(channel_data, _scales, wavelet)
                coeffs_list.append(coeffs[0])

            # 将所有通道的小波变换结果合并为三维数组
            wavelet_array = np.stack(coeffs_list, axis=-1)
            return wavelet_array

        return self._return(_wavelet, name="wavelet")

    def cwt(self, wavelet="morl", scales=None):
        """连续小波变换 (等同于wavelet)。

        Args:
            wavelet: 小波类型
            scales: 尺度列表，默认为None (自动创建)

        Returns:
            DASFrame: 转换后的DASFrame对象
        """
        return self.wavelet(wavelet=wavelet, scales=scales)

    def hilbert(self, axis="time"):
        """希尔伯特变换。

        Args:
            axis: 时间轴/通道轴

        Returns:
            DASFrame: 转换后的DASFrame对象，复数值
        """

        def _hilbert(data):
            _axis = 0 if axis == "time" else 1
            return signal.hilbert(data, axis=_axis)

        return self._return(_hilbert, name="hilbert")

    # --- 衍生特征 ---
    def envelope(self, method="hilbert", axis="time"):
        """计算信号包络。

        Args:
            method: 计算方法，'hilbert'或'peak'
            axis: 时间轴/通道轴

        Returns:
            DASFrame: 包络信号
        """

        def _envelope(data):
            _axis = 0 if axis == "time" else 1
            if method == "hilbert":
                analytic_signal = signal.hilbert(data, axis=_axis)
                return np.abs(analytic_signal)
            elif method == "peak":
                # 使用峰值检测计算包络
                raise NotImplementedError("峰值检测法暂未实现")
            else:
                raise ValueError(f"不支持的包络计算方法: {method}")

        return self._return(_envelope, name="envelope")

    def diff(self, order=1, axis="time"):
        """差分。

        Args:
            order: 差分阶数
            axis: 时间轴/通道轴

        Returns:
            DASFrame: 差分后的数据
        """

        def _diff(data):
            _axis = 0 if axis == "time" else 1
            return np.diff(data, n=order, axis=_axis)

        return self._return(_diff, name="diff")

    def cumsum(self, axis="time"):
        """积分(累加)。

        Args:
            axis: 时间轴/通道轴

        Returns:
            DASFrame: 积分后的数据
        """

        def _cumsum(data):
            _axis = 0 if axis == "time" else 1
            return np.cumsum(data, axis=_axis)

        return self._return(_cumsum, name="cumsum")

    def hilbert_env(self, axis="time"):
        """希尔伯特包络。

        Args:
            axis: 时间轴/通道轴

        Returns:
            DASFrame: 希尔伯特包络
        """
        return self.envelope(method="hilbert", axis=axis)

    # --- 空域 ---
    def spatial_smooth(self, kernel=5):
        """空间平滑。

        Args:
            kernel: 平滑窗口大小

        Returns:
            DASFrame: 平滑后的数据
        """

        def _spatial_smooth(data):
            # 在通道维度上平滑
            from scipy.ndimage import uniform_filter1d

            return uniform_filter1d(data, size=kernel, axis=1)

        return self._return(_spatial_smooth, name="spatial_smooth")

    # --- 统计 ---
    def rms(self, window, axis="time"):
        """计算RMS (均方根)。

        Args:
            window: 窗口大小
            axis: 时间轴/通道轴

        Returns:
            DASFrame: RMS值
        """

        def _rms(data):
            _axis = 0 if axis == "time" else 1
            # 使用滑动窗口计算RMS
            if _axis == 0:
                rms_values = np.zeros((data.shape[0] - window + 1, data.shape[1]))
                for i in range(data.shape[0] - window + 1):
                    window_data = data[i : i + window, :]
                    rms_values[i, :] = np.sqrt(np.mean(window_data**2, axis=0))
                return rms_values
            else:
                rms_values = np.zeros((data.shape[0], data.shape[1] - window + 1))
                for i in range(data.shape[1] - window + 1):
                    window_data = data[:, i : i + window]
                    rms_values[:, i] = np.sqrt(np.mean(window_data**2, axis=1))
                return rms_values

        return self._return(_rms, name="rms")

    def mean(self, axis=None):
        """均值。

        Args:
            axis: 轴，None表示全局均值

        Returns:
            DASFrame: 均值
        """

        def _mean(data):
            return np.mean(data, axis=axis, keepdims=axis is not None)

        return self._return(_mean, name="mean")

    def max(self, axis=None):
        """最大值。

        Args:
            axis: 轴，None表示全局最大值

        Returns:
            DASFrame: 最大值
        """

        def _max(data):
            return np.max(data, axis=axis, keepdims=axis is not None)

        return self._return(_max, name="max")

    def min(self, axis=None):
        """最小值。

        Args:
            axis: 轴，None表示全局最小值

        Returns:
            DASFrame: 最小值
        """

        def _min(data):
            return np.min(data, axis=axis, keepdims=axis is not None)

        return self._return(_min, name="min")

    def std(self, axis=None):
        """标准差。

        Args:
            axis: 轴，None表示全局标准差

        Returns:
            DASFrame: 标准差
        """

        def _std(data):
            return np.std(data, axis=axis, keepdims=axis is not None)

        return self._return(_std, name="std")

    # --- 检测 ---
    def threshold_detect(self, threshold=None, db=None):
        """阈值检测。

        Args:
            threshold: 阈值，默认为均值+3倍标准差
            db: dB阈值，覆盖threshold

        Returns:
            DASFrame: 检测结果，布尔数组
        """

        def _threshold_detect(data):
            if db is not None:
                # 将dB阈值转换为线性阈值
                max_val = np.max(np.abs(data))
                _threshold = max_val * 10 ** (db / 20)
            elif threshold is not None:
                _threshold = threshold
            else:
                # 默认阈值: 均值 + 3倍标准差
                _threshold = np.mean(np.abs(data)) + 3 * np.std(np.abs(data))

            return np.abs(data) > _threshold

        return self._return(_threshold_detect, name="threshold_detect")

    # --- Sink (触发计算) ---
    def plot_ts(self, ch=None, title=None, xlabel="Time (s)", ylabel="Amplitude"):
        """绘制 Nature/Science 级别时间序列图。

        Args:
            ch: 通道索引，None表示前5个通道
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签

        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        from ..visualization.styles import apply_nature_style, setup_axis
        from ..config.visualization_config import VisualizationConfig, FigureSize
        
        # 应用 Nature 样式
        config = VisualizationConfig()
        apply_nature_style(config)
        
        # 触发计算
        data = self._graph.compute()

        # 创建图表 - 使用期刊标准尺寸
        fig, ax = plt.subplots(figsize=FigureSize.WIDE.value, constrained_layout=True)
        time_axis = np.arange(data.shape[0]) / self._fs
        
        # 获取配色
        colors = config.colors.primary

        if ch is None:
            n_channels = min(data.shape[1], 5)  # 限制绘制通道数
            for i in range(n_channels):
                ax.plot(
                    time_axis, data[:, i], 
                    color=colors[i % len(colors)],
                    linewidth=config.line_width,
                    label=f"Ch {i}",
                    alpha=0.9,
                )
        else:
            ax.plot(
                time_axis, data[:, ch], 
                color=colors[0],
                linewidth=config.line_width,
                label=f"Ch {ch}",
            )

        # 应用专业样式
        setup_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title, config=config)
        
        # 图例 - 无边框样式
        if data.shape[1] > 1 or ch is None:
            ax.legend(
                loc="upper right",
                frameon=False,
                fontsize=config.typography.legend,
            )

        return fig

    def plot_heatmap(self, t_range=None, ch_range=None, cmap=None, title=None):
        """绘制 Nature/Science 级别热图。

        Args:
            t_range: 时间范围元组
            ch_range: 通道范围元组
            cmap: 配色方案，默认使用色盲友好方案
            title: 图表标题

        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        from ..visualization.styles import apply_nature_style, setup_axis, add_colorbar
        from ..config.visualization_config import VisualizationConfig, FigureSize
        
        # 应用 Nature 样式
        config = VisualizationConfig()
        apply_nature_style(config)
        
        # 触发计算
        data = self._graph.compute()

        # 处理切片
        if t_range is not None:
            t_slice = slice(*t_range) if isinstance(t_range, tuple) else t_range
        else:
            t_slice = slice(None)

        if ch_range is not None:
            ch_slice = slice(*ch_range) if isinstance(ch_range, tuple) else ch_range
        else:
            ch_slice = slice(None)

        data_to_plot = data[t_slice, ch_slice]
        
        # 计算时间轴
        n_samples = data_to_plot.shape[0]
        time_extent = n_samples / self._fs
        n_channels = data_to_plot.shape[1]

        # 创建图表 - 使用室内等尺寸
        fig, ax = plt.subplots(figsize=FigureSize.SINGLE_COLUMN.value, constrained_layout=True)
        
        # 设置对称色标范围
        vmax = np.percentile(np.abs(data_to_plot), 98)
        vmin = -vmax
        
        # 使用配色方案
        if cmap is None:
            cmap = config.colors.diverging if np.any(data_to_plot < 0) else config.colors.sequential
        
        im = ax.imshow(
            data_to_plot.T,
            aspect="auto",
            cmap=cmap,
            origin="lower",
            interpolation="none",
            extent=[0, time_extent, 0, n_channels],
            vmin=vmin,
            vmax=vmax,
        )

        # 添加专业格式颜色条
        add_colorbar(fig, im, ax, label="Amplitude", config=config)

        # 设置标签
        setup_axis(ax, xlabel="Time (s)", ylabel="Channel", title=title or "DAS Data", config=config)

        return fig

    def plot_spec(self, cmap=None, title=None, db_range=None):
        """绘制 Nature/Science 级别频谱图。

        Args:
            cmap: 色彩映射，默认使用 inferno
            title: 图表标题
            db_range: dB 范围 (min, max)

        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        from ..visualization.styles import apply_nature_style, setup_axis, add_colorbar
        from ..config.visualization_config import VisualizationConfig, FigureSize
        
        # 应用 Nature 样式
        config = VisualizationConfig()
        apply_nature_style(config)
        
        if cmap is None:
            cmap = config.colors.spectrogram
        
        # 触发计算
        data = self._graph.compute()

        # 创建图表
        fig, ax = plt.subplots(figsize=FigureSize.SINGLE_COLUMN.value, constrained_layout=True)
        
        if data.ndim == 3:  # 如果已经是STFT结果
            # 对所有通道求平均
            averaged_spec = np.mean(data, axis=-1)
            spec_db = 10 * np.log10(averaged_spec + 1e-12)
            
            # 设置色标范围
            if db_range is None:
                vmin = np.percentile(spec_db, 5)
                vmax = np.percentile(spec_db, 98)
            else:
                vmin, vmax = db_range
            
            im = ax.imshow(
                spec_db,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            add_colorbar(fig, im, ax, label="Power (dB)", config=config)
            setup_axis(ax, xlabel="Time", ylabel="Frequency", 
                      title=title or "Spectrogram", config=config)
        else:
            # 计算并绘制STFT
            f, t, Zxx = signal.stft(
                data[:, 0], fs=self._fs, window="hann", nperseg=256, noverlap=192
            )
            spec_db = 10 * np.log10(np.abs(Zxx) + 1e-12)
            
            # 设置色标范围
            if db_range is None:
                vmin = np.percentile(spec_db, 5)
                vmax = np.percentile(spec_db, 98)
            else:
                vmin, vmax = db_range

            im = ax.pcolormesh(
                t, f, spec_db, 
                shading="gouraud", 
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,  # 提高 PDF 导出质量
            )
            add_colorbar(fig, im, ax, label="Power (dB)", config=config)
            setup_axis(ax, xlabel="Time (s)", ylabel="Frequency (Hz)",
                      title=title or "Spectrogram", config=config)

        return fig

    def to_h5(self, path):
        """保存数据到HDF5文件。

        Args:
            path: 文件路径

        Returns:
            str: 文件路径
        """
        import h5py

        # 触发计算
        data = self._graph.compute()

        # 保存到HDF5文件
        with h5py.File(path, "w") as f:
            # 保存数据
            f.create_dataset("data", data=data)

            # 保存元数据
            f.attrs["fs"] = self._fs
            for key, value in self._metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    f.attrs[key] = value

        return path

    def to_parquet(self, path):
        """保存数据到Parquet文件。

        Args:
            path: 文件路径

        Returns:
            str: 文件路径
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("需要安装pyarrow: pip install pyarrow")

        # 触发计算
        data = self._graph.compute()

        # 创建PyArrow表
        table = pa.Table.from_arrays(
            [pa.array(data[:, i]) for i in range(data.shape[1])],
            names=[f"channel_{i}" for i in range(data.shape[1])],
        )

        # 保存到Parquet文件
        metadata = {
            "fs": str(self._fs),
            **{
                k: str(v)
                for k, v in self._metadata.items()
                if isinstance(v, (str, int, float, bool))
            },
        }
        pq.write_table(table, path, metadata=metadata)

        return path

    def collect(self):
        """触发计算并返回结果。

        Returns:
            np.ndarray: 计算结果
        """
        return self._graph.compute()

    def run_forever(self):
        """启动流处理循环。

        只适用于流式数据。
        """
        if not self._is_stream:
            raise ValueError("只有流式数据可以调用run_forever")

        # TODO: 实现真正的流处理循环
        print(f"开始处理数据流: {self._metadata.get('stream_url', 'unknown')}")
        print("流处理功能尚未完全实现")

    # --- 内部辅助方法 ---
    def _return(self, operation, *args, name=None, **kwargs):
        """创建新的DASFrame对象，共享同一计算图。

        Args:
            operation: 操作函数
            *args: 操作函数位置参数
            name: 操作名称
            **kwargs: 操作函数关键字参数

        Returns:
            DASFrame: 新的DASFrame对象
        """
        new_graph = self._graph.add(operation, *args, **kwargs)
        return DASFrame(
            data=self._data,  # 原始数据引用
            fs=self._fs,
            graph=new_graph,
            is_stream=self._is_stream,
            **self._metadata,
        )
