import builtins
from typing import Any, List, Optional, Tuple, Union, cast

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import signal

from ..core.inventory import DASInventory


class DASFrame:
    """DAS 数据处理核心类 (Xarray/Dask Backend)。

    所有信号处理操作基于 xarray 和 dask 进行延迟计算。
    """

    _data: xr.DataArray
    _fs: float
    _dx: float
    _metadata: dict[str, Any]

    def __init__(
        self,
        data: Union[xr.DataArray, np.ndarray, da.Array, "DASFrame"],
        fs: float,
        dx: float = 1.0,
        **metadata: Any,
    ) -> None:
        """Initialize DASFrame.

        Args:
            data: Input data.
            fs: Sampling frequency (Hz).
            dx: Channel spacing (m).
            **metadata: Additional metadata.
        """
        if isinstance(data, DASFrame):
            # Copy constructor roughly
            self._data = data._data
            self._fs = data.fs
            self._dx = getattr(data, "_dx", dx)
            # Combine metadata, new metadata overrides old
            self._metadata = {**data._metadata, **metadata}
            # Special handling for inventory in metadata to ensure type safety
            if "inventory" in self._metadata:
                inv = self._metadata["inventory"]
                if isinstance(inv, dict):
                    self._metadata["inventory"] = DASInventory.model_validate(inv)
            return

        self._fs = fs
        self._dx = dx
        self._metadata = metadata

        # Helper to ensure inventory is properly typed if passed
        if "inventory" in self._metadata:
            inv = self._metadata["inventory"]
            if isinstance(inv, dict):
                self._metadata["inventory"] = DASInventory.model_validate(inv)

        if isinstance(data, xr.DataArray):
            # Ensure order is (time, distance)
            if "time" in data.dims and data.dims[0] != "time":
                data = data.transpose("time", ...)
            self._data = data
        else:
            # Wrap numpy/dask array into xarray
            # Assume dims are (time, channel)
            if not isinstance(data, (np.ndarray, da.Array)):
                data = np.asarray(data)

            # Create coordinates
            nt, nx = data.shape
            coords = {
                "time": np.arange(nt) / fs,
                "distance": np.arange(nx) * dx,
            }

            self._data = xr.DataArray(
                data,
                dims=("time", "distance"),
                coords=coords,
                name="das_strain",
                attrs={"fs": fs, "dx": dx, **metadata},
            )

            # Ensure it is chunked (dask backed) if it isn't already
            if not self._data.chunks:
                self._data = self._data.chunk({"time": "auto", "distance": -1})

    @property
    def data(self) -> xr.DataArray:
        """Access underlying xarray DataArray."""
        return self._data

    @property
    def fs(self) -> float:
        return self._fs

    @property
    def inventory(self) -> Optional[DASInventory]:
        """Access the DASInventory object if available."""
        return self._metadata.get("inventory")

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get shape of data."""
        return self._data.shape

    def collect(self) -> np.ndarray:
        """Compute and return numpy array."""
        return self._data.compute().values

    # --- 基础操作 ---
    def slice(self, t: slice = slice(None), x: slice = slice(None)) -> "DASFrame":
        """Slice data."""
        # xarray is slicing by index here (isel)
        sliced = self._data.isel(time=t, distance=x)
        return DASFrame(sliced, self._fs, self._dx, **self._metadata)

    # --- Signal Operations ---

    def bandpass(self, low: float, high: float, order: int = 4) -> "DASFrame":
        """Apply bandpass filter using dask map_overlap or map_blocks."""

        def _filter_func(block, fs, low, high, order):
            nyq = 0.5 * fs
            sos = signal.butter(
                order, [low / nyq, high / nyq], btype="band", output="sos"
            )
            # Apply along the last axis (core dimension moved to end by apply_ufunc)
            return signal.sosfiltfilt(sos, block, axis=-1)

        data_contiguous_time = self._data.chunk({"time": -1})

        filtered = xr.apply_ufunc(
            _filter_func,
            data_contiguous_time,
            kwargs={"fs": self._fs, "low": low, "high": high, "order": order},
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=[data_contiguous_time.dtype],
        )

        return DASFrame(filtered, self._fs, self._dx, **self._metadata)

    def lowpass(self, cutoff: float, order: int = 4) -> "DASFrame":
        """Apply lowpass filter."""

        def _filter_func(block, fs, cutoff, order):
            nyq = 0.5 * fs
            sos = signal.butter(order, cutoff / nyq, btype="low", output="sos")
            return signal.sosfiltfilt(sos, block, axis=-1)

        data_contiguous_time = self._data.chunk({"time": -1})
        filtered = xr.apply_ufunc(
            _filter_func,
            data_contiguous_time,
            kwargs={"fs": self._fs, "cutoff": cutoff, "order": order},
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=[data_contiguous_time.dtype],
        )
        return DASFrame(filtered, self._fs, self._dx, **self._metadata)

    def highpass(self, cutoff: float, order: int = 4) -> "DASFrame":
        """高通滤波器。"""

        def _filter_func(block, fs, cutoff, order):
            nyq = 0.5 * fs
            sos = signal.butter(order, cutoff / nyq, btype="high", output="sos")
            return signal.sosfiltfilt(sos, block, axis=-1)

        data_contiguous_time = self._data.chunk({"time": -1})
        filtered = xr.apply_ufunc(
            _filter_func,
            data_contiguous_time,
            kwargs={"fs": self._fs, "cutoff": cutoff, "order": order},
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=[data_contiguous_time.dtype],
        )
        return DASFrame(filtered, self._fs, self._dx, **self._metadata)

    def notch(self, freq: float, Q: float = 30) -> "DASFrame":
        """陷波滤波器，移除特定频率成分。"""

        def _filter_func(block, fs, freq, Q):
            b, a = signal.iirnotch(freq, Q, fs)
            return signal.filtfilt(b, a, block, axis=-1)

        data_contiguous_time = self._data.chunk({"time": -1})
        filtered = xr.apply_ufunc(
            _filter_func,
            data_contiguous_time,
            kwargs={"fs": self._fs, "freq": freq, "Q": Q},
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=[data_contiguous_time.dtype],
        )
        return DASFrame(filtered, self._fs, self._dx, **self._metadata)

    def detrend(self, axis: str = "time") -> "DASFrame":
        """Detrend data."""

        def _detrend_func(data):
            return signal.detrend(data, axis=-1)

        if axis == "time":
            data = self._data.chunk({"time": -1})
        else:
            data = self._data.chunk({"distance": -1})

        detrended = xr.apply_ufunc(
            _detrend_func,
            data,
            input_core_dims=[[axis]],
            output_core_dims=[[axis]],
            dask="parallelized",
            output_dtypes=[data.dtype],
        )

        return DASFrame(detrended, self._fs, self._dx, **self._metadata)

    def demean(self, axis: str = "time") -> "DASFrame":
        """Remove mean value along specified axis."""
        dim = "time" if axis == "time" else "distance"
        demeaned = self._data - self._data.mean(dim=dim)
        return DASFrame(demeaned, self._fs, self._dx, **self._metadata)

    def normalize(self, method: str = "minmax") -> "DASFrame":
        """归一化。"""
        if method == "zscore":
            mean = self._data.mean(dim="time")
            std = self._data.std(dim="time")
            std = xr.where(std == 0, 1.0, std)
            normalized = (self._data - mean) / std
        else:  # minmax
            min_val = self._data.min(dim="time")
            max_val = self._data.max(dim="time")
            range_val = max_val - min_val
            range_val = xr.where(range_val == 0, 1.0, range_val)
            normalized = 2 * (self._data - min_val) / range_val - 1

        return DASFrame(normalized, self._fs, self._dx, **self._metadata)

    # --- 统计 (Statistics) ---
    def mean(self, axis: Optional[int] = 0) -> np.ndarray:
        """计算均值。

        Args:
            axis: 计算轴，0 表示沿时间轴返回每通道的均值，None 返回全局标量
        """
        if axis is None:
            return self._data.mean().compute().values
        dim = "time" if axis == 0 else "distance"
        return self._data.mean(dim=dim).compute().values

    def std(self, axis: Optional[int] = 0) -> np.ndarray:
        """计算标准差。"""
        if axis is None:
            return self._data.std().compute().values
        dim = "time" if axis == 0 else "distance"
        return self._data.std(dim=dim).compute().values

    def max(self, axis: Optional[int] = 0) -> np.ndarray:
        """计算最大值。"""
        if axis is None:
            return self._data.max().compute().values
        dim = "time" if axis == 0 else "distance"
        return self._data.max(dim=dim).compute().values

    def min(self, axis: Optional[int] = 0) -> np.ndarray:
        """计算最小值。"""
        if axis is None:
            return self._data.min().compute().values
        dim = "time" if axis == 0 else "distance"
        return self._data.min(dim=dim).compute().values

    def rms(self, window: Optional[int] = None) -> np.ndarray:
        """计算 RMS（均方根）。"""
        data_sq = self._data**2
        if window is None:
            return np.sqrt(data_sq.mean(dim="time").compute().values)
        else:
            # 滑动 RMS，使用 xarray 滚动窗口
            rolling_mean = data_sq.rolling(time=window, center=True).mean()
            return np.sqrt(rolling_mean.compute().values)

    def fft(self) -> "DASFrame":
        """快速傅立叶变换 (Real FFT)，返回单边幅度谱。"""

        def _fft_func(data):
            return np.abs(np.fft.rfft(data, axis=-1))

        data = self._data.chunk({"time": -1})
        n_samples = data.sizes["time"]
        freqs = np.fft.rfftfreq(n_samples, 1 / self._fs)

        spectrum = xr.apply_ufunc(
            _fft_func,
            data,
            input_core_dims=[["time"]],
            output_core_dims=[["frequency"]],
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes": {"frequency": len(freqs)}},
        )

        spectrum = spectrum.assign_coords(frequency=freqs)
        # apply_ufunc moves core dims to the end, result is (distance, frequency)
        # transpose back to (frequency, distance) to match convention
        spectrum = spectrum.transpose("frequency", ...)

        # 注意：FFT 变换后，'time' 轴变成了 'frequency' 轴，fs 含义也变化了
        return DASFrame(spectrum, self._fs, self._dx, **self._metadata)

    def hilbert(self) -> "DASFrame":
        """希尔伯特变换，返回解析信号。"""

        def _hilbert_func(data):
            return signal.hilbert(data, axis=-1)

        data = self._data.chunk({"time": -1})
        analytical = xr.apply_ufunc(
            _hilbert_func,
            data,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=[complex],
        )
        return DASFrame(analytical, self._fs, self._dx, **self._metadata)

    def envelope(self) -> "DASFrame":
        """提取信号包络。"""
        analytical = self.hilbert()
        return analytical.abs()

    def spatial_smooth(self, kernel: int = 3) -> "DASFrame":
        """空间平滑处理。

        Args:
            kernel: 滚动窗口大小（通道数）。
        """
        smoothed = self._data.rolling(distance=kernel, center=True).mean()
        return DASFrame(smoothed, self._fs, self._dx, **self._metadata)

    def scale(self, factor: float = 1.0) -> "DASFrame":
        """Scale amplitude."""
        scaled = self._data * factor
        return DASFrame(scaled, self._fs, self._dx, **self._metadata)

    def abs(self) -> "DASFrame":
        """Absolute value."""
        return DASFrame(np.abs(self._data), self._fs, self._dx, **self._metadata)

    def integrate(self) -> "DASFrame":
        """Time domain integration (cumulative sum / fs)."""
        # trapezium integration or simple cumsum
        # cumsum is faster and dask friendly
        integrated = self._data.cumsum(dim="time") / self._fs
        return DASFrame(integrated, self._fs, self._dx, **self._metadata)

    def differentiate(self) -> "DASFrame":
        """Time domain differentiation (diff * fs)."""
        differentiated = self._data.diff(dim="time") * self._fs
        # diff reduce size by 1, pad to keep shape
        differentiated = differentiated.pad(time=(1, 0), mode="edge")
        return DASFrame(differentiated, self._fs, self._dx, **self._metadata)

    def stft(self, nperseg: int = 256, noverlap: Optional[int] = None) -> "DASFrame":
        """短时傅立叶变换，进行时频分析。"""
        if noverlap is None:
            noverlap = nperseg // 2

        data = self.collect()
        # signal.stft for 2D data with axis=0:
        # returns f, t, Zxx where Zxx shape is (freq, distance, time)
        f, t, Zxx = signal.stft(
            data, fs=self._fs, nperseg=nperseg, noverlap=noverlap, axis=0
        )

        # 将 Zxx 从 (freq, distance, time) 转换为 (freq, time, distance)
        Zxx_abs = np.abs(Zxx).transpose(0, 2, 1)

        # 将结果包装回 xr.DataArray
        stft_data = xr.DataArray(
            Zxx_abs,
            dims=("frequency", "time", "distance"),
            coords={
                "frequency": f,
                "time": t,
                "distance": self._data.distance,
            },
            name="stft",
            attrs=self._data.attrs,
        )
        return DASFrame(stft_data, self._fs, self._dx, **self._metadata)

    def fk_filter(
        self,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        dx: float = 1.0,
    ) -> "DASFrame":
        """F-K 滤波 (速度滤波)。"""
        data = self.collect()
        from ..config.sampling_config import SamplingConfig
        from ..processing.das_processor import DASProcessor

        config = SamplingConfig(fs=self._fs, channels=data.shape[1])
        processor = DASProcessor(config)
        filtered = processor.FKFilter(data, v_min=v_min, v_max=v_max, dx=dx)
        return DASFrame(filtered, self._fs, dx, **self._metadata)

    # --- 检测 (Detection) ---
    def threshold_detect(
        self, threshold: Optional[float] = None, sigma: float = 3.0
    ) -> np.ndarray:
        """阈值检测。"""
        data = self.collect()
        if threshold is None:
            threshold = float(np.mean(data) + sigma * np.std(data))
        return np.abs(data) > threshold

    # --- Visualization ---

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

        t = self._data.time.values
        if ch is None:
            ax.plot(t, data[:, : min(5, data.shape[1])], alpha=0.7)
        else:
            ax.plot(t, data[:, ch])

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig

    def plot_spectrum(
        self,
        ch: int = 0,
        title: str = "Spectrum",
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ) -> plt.Figure:
        """绘制频谱图 (FFT)。"""
        data = self.collect()
        ch_data = data[:, ch]

        from ..visualization.das_visualizer import SpectrumPlot

        plotter = SpectrumPlot()
        # 计算频谱 (Welch 方法)
        from scipy import signal

        n = len(ch_data)
        nperseg = min(2048, n // 4)
        freqs, psd = signal.welch(
            ch_data, fs=self._fs, nperseg=nperseg, scaling="spectrum"
        )
        mags = np.sqrt(psd)

        fig = plotter.plot(freqs, mags, title=title, ax=ax, **kwargs)
        return fig

    def plot_spectrogram(
        self,
        ch: int = 0,
        title: str = "Spectrogram",
        window_size: int = 1024,
        overlap: float = 0.75,
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ) -> plt.Figure:
        """绘制时频图 (STFT)。"""
        data = self.collect()
        ch_data = data[:, ch]

        from ..visualization.das_visualizer import SpectrogramPlot

        plotter = SpectrogramPlot()
        fig = plotter.plot(
            ch_data,
            self._fs,
            window_size=window_size,
            overlap=overlap,
            title=title,
            ax=ax,
            **kwargs,
        )
        return fig

    def plot_heatmap(
        self,
        channels: Optional[builtins.slice] = None,
        t_range: Optional[builtins.slice] = None,
        title: str = "DAS Waterfall",
        cmap: str = "RdBu_r",
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ) -> plt.Figure:
        """绘制热图（瀑布图）。

        Args:
            channels: 通道切片，例如 slice(10, 60, 5)。
            t_range: 时间样本切片（以样本为单位），例如 slice(0, 1000)。
            title: 图标题。
            cmap: 颜色映射。
            ax: 可选的 matplotlib Axes 对象。
            **kwargs: 传递给 imshow 的额外参数。

        Returns:
            matplotlib Figure 对象。
        """
        if t_range is not None or channels is not None:
            # 提醒用户优先使用 .slice()
            import warnings

            warnings.warn(
                "Using t_range or channels in plot_heatmap is deprecated. "
                "Please use .slice() instead."
            )
            frame = self.slice(t=t_range or slice(None), x=channels or slice(None))
        else:
            frame = self

        data = frame.collect()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            if ax.figure is None:
                raise ValueError("Provided ax must belong to a figure")
            fig = cast(plt.Figure, ax.figure)

        # 获取绝对坐标边界
        dist_coords = frame._data.distance.values / self._dx
        time_coords = frame._data.time.values

        extent = (
            dist_coords[0] - 0.5,
            dist_coords[-1] + 0.5,
            time_coords[-1] + 0.5 / self._fs,
            time_coords[0] - 0.5 / self._fs,
        )
        vmax = np.percentile(np.abs(data), 98)
        vmax_val = float(vmax)
        im = ax.imshow(
            data,
            aspect="auto",
            cmap=cmap,
            extent=extent,
            vmin=-vmax_val,
            vmax=vmax_val,
            **kwargs,
        )

        ax.set_xlabel("Channel")
        ax.set_ylabel("Time (s)")
        ax.set_title(title)
        if ax is not None:
            plt.colorbar(im, ax=ax, label="Amplitude")
        return fig

    def plot_fk(
        self,
        dx: float = 1.0,
        title: str = "F-K Spectrum",
        v_lines: Optional[List[float]] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ) -> plt.Figure:
        data = self.collect()

        from ..config.sampling_config import SamplingConfig
        from ..processing.das_processor import DASProcessor

        config = SamplingConfig(fs=self._fs, channels=data.shape[1])
        processor = DASProcessor(config)

        fk, freqs, k = processor.f_k_transform(data)
        k = k / dx

        from ..visualization.das_visualizer import FKPlot

        plotter = FKPlot()
        fig = plotter.plot(fk, freqs, k, title=title, v_lines=v_lines, **kwargs)
        return fig

    def plot_profile(
        self,
        stat: str = "rms",
        channels: Optional[builtins.slice] = None,
        x_axis: str = "channel",
        title: Optional[str] = None,
        ylabel: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ) -> plt.Figure:
        """绘制空间剖面图（如 RMS, Mean, Std 等）。

        Args:
            stat: 统计指标类型, 'rms', 'mean', 'std', 'max', 'min'
            channels: 通道切片，例如 slice(400, 600)
            x_axis: X 轴数据，'channel' (点位) 或 'distance' (物理距离)
            title: 图标题
            ylabel: Y轴标签
            ax: 可选标的 matplotlib Axes 对象
            **kwargs: 传递给 ProfilePlot.plot 的其他参数
        """
        # 如果指定了通道范围，先进行切片
        frame = self if channels is None else self.slice(x=channels)

        if stat == "rms":
            values = frame.rms()
            default_title = "RMS Profile"
            default_ylabel = "RMS Amplitude"
        elif stat == "mean":
            values = frame.mean(axis=0).flatten()
            default_title = "Mean Profile"
            default_ylabel = "Mean Amplitude"
        elif stat == "std":
            values = frame.std(axis=0).flatten()
            default_title = "Standard Deviation Profile"
            default_ylabel = "Std Amplitude"
        elif stat == "max":
            values = frame.max(axis=0).flatten()
            default_title = "Max Profile"
            default_ylabel = "Max Amplitude"
        elif stat == "min":
            values = frame.min(axis=0).flatten()
            default_title = "Min Profile"
            default_ylabel = "Min Amplitude"
        else:
            raise ValueError(f"Unsupported stat: {stat}")

        from ..visualization.das_visualizer import ProfilePlot

        plotter = ProfilePlot()

        if x_axis == "channel":
            # 这里的距离坐标除以 dx 得到绝对点位索引
            distances = frame._data.distance.values / self._dx
            if "xlabel" not in kwargs:
                kwargs["xlabel"] = "Channel"
        else:
            distances = frame._data.distance.values
            if "xlabel" not in kwargs:
                kwargs["xlabel"] = "Distance (m)"

        return plotter.plot(
            values=values,
            distances=distances,
            title=title or default_title,
            ylabel=ylabel or default_ylabel,
            ax=ax,
            **kwargs,
        )

    def plot_rms(
        self,
        channels: Optional[builtins.slice] = None,
        x_axis: str = "channel",
        **kwargs: Any,
    ) -> plt.Figure:
        """绘制 RMS 剖面图。"""
        return self.plot_profile(stat="rms", channels=channels, x_axis=x_axis, **kwargs)

    def plot_mean(
        self,
        channels: Optional[builtins.slice] = None,
        x_axis: str = "channel",
        **kwargs: Any,
    ) -> plt.Figure:
        """绘制均值剖面图。"""
        return self.plot_profile(
            stat="mean", channels=channels, x_axis=x_axis, **kwargs
        )

    def plot_std(
        self,
        channels: Optional[builtins.slice] = None,
        x_axis: str = "channel",
        **kwargs: Any,
    ) -> plt.Figure:
        """绘制标准差剖面图。"""
        return self.plot_profile(stat="std", channels=channels, x_axis=x_axis, **kwargs)
