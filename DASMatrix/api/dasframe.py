import builtins
import warnings
from typing import Any, List, Optional, Tuple, Union, cast

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
        """Initialize DASFrame."""
        if isinstance(data, DASFrame):
            self._data = data._data
            self._fs = data.fs
            self._dx = getattr(data, "_dx", dx)
            self._metadata = {**data._metadata, **metadata}
            if "inventory" in self._metadata:
                inv = self._metadata["inventory"]
                if isinstance(inv, dict):
                    self._metadata["inventory"] = DASInventory.model_validate(inv)
            return

        self._fs = fs
        self._dx = dx
        self._metadata = metadata

        if "inventory" in self._metadata:
            inv = self._metadata["inventory"]
            if isinstance(inv, dict):
                self._metadata["inventory"] = DASInventory.model_validate(inv)

        if isinstance(data, xr.DataArray):
            if "time" in data.dims and data.dims[0] != "time":
                data = data.transpose("time", ...)
            self._data = data
            self._metadata = {**data.attrs, **metadata}
            if "sampling_rate" in self._metadata and fs is None:
                self._fs = float(self._metadata["sampling_rate"])
            if "channel_spacing" in self._metadata and dx == 1.0:
                self._dx = float(self._metadata["channel_spacing"])
        else:
            if not isinstance(data, (np.ndarray, da.Array)):
                data = np.asarray(data)

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

    def astype(self, dtype: Any) -> "DASFrame":
        """Cast data to a new type."""
        return DASFrame(self._data.astype(dtype), self._fs, self._dx, **self._metadata)

    def flatten(self) -> np.ndarray:
        """Return a flattened version of the data."""
        return self.collect().flatten()

    def slice(self, t: slice = slice(None), x: slice = slice(None)) -> "DASFrame":
        """Slice data."""
        sliced = self._data.isel(time=t, distance=x)
        return DASFrame(sliced, self._fs, self._dx, **self._metadata)

    def bandpass(self, low: float, high: float, order: int = 4) -> "DASFrame":
        """Apply bandpass filter."""

        def _filter_func(block, fs, low, high, order):
            nyq = 0.5 * fs
            sos = signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
            return signal.sosfiltfilt(sos, block, axis=-1)

        data = self._data
        filtered = xr.apply_ufunc(
            _filter_func,
            data,
            kwargs={"fs": self._fs, "low": low, "high": high, "order": order},
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=[data.dtype],
        )

        return DASFrame(filtered, self._fs, self._dx, **self._metadata)

    def lowpass(self, cutoff: float, order: int = 4) -> "DASFrame":
        """Apply lowpass filter."""

        def _filter_func(block, fs, cutoff, order):
            nyq = 0.5 * fs
            sos = signal.butter(order, cutoff / nyq, btype="low", output="sos")
            return signal.sosfiltfilt(sos, block, axis=-1)

        filtered = xr.apply_ufunc(
            _filter_func,
            self._data,
            kwargs={"fs": self._fs, "cutoff": cutoff, "order": order},
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=[self._data.dtype],
        )
        return DASFrame(filtered, self._fs, self._dx, **self._metadata)

    def highpass(self, cutoff: float, order: int = 4) -> "DASFrame":
        """Apply highpass filter."""

        def _filter_func(block, fs, cutoff, order):
            nyq = 0.5 * fs
            sos = signal.butter(order, cutoff / nyq, btype="high", output="sos")
            return signal.sosfiltfilt(sos, block, axis=-1)

        filtered = xr.apply_ufunc(
            _filter_func,
            self._data,
            kwargs={"fs": self._fs, "cutoff": cutoff, "order": order},
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=[self._data.dtype],
        )
        return DASFrame(filtered, self._fs, self._dx, **self._metadata)

    def notch(self, freq: float, Q: float = 30) -> "DASFrame":
        """陷波滤波器，移除特定频率成分。"""

        def _filter_func(block, fs, freq, Q):
            b, a = signal.iirnotch(freq, Q, fs)
            return signal.filtfilt(b, a, block, axis=-1)

        filtered = xr.apply_ufunc(
            _filter_func,
            self._data,
            kwargs={"fs": self._fs, "freq": freq, "Q": Q},
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=[self._data.dtype],
        )
        return DASFrame(filtered, self._fs, self._dx, **self._metadata)

    def detrend(self, axis: str = "time") -> "DASFrame":
        """Detrend data."""

        def _detrend_func(data):
            return signal.detrend(data, axis=-1)

        detrended = xr.apply_ufunc(
            _detrend_func,
            self._data,
            input_core_dims=[[axis]],
            output_core_dims=[[axis]],
            dask="parallelized",
            output_dtypes=[self._data.dtype],
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
        else:
            min_val = self._data.min(dim="time")
            max_val = self._data.max(dim="time")
            range_val = max_val - min_val
            range_val = xr.where(range_val == 0, 1.0, range_val)
            normalized = 2 * (self._data - min_val) / range_val - 1

        return DASFrame(normalized, self._fs, self._dx, **self._metadata)

    def any(self, axis: Optional[Union[int, str]] = None) -> Union[bool, np.ndarray]:
        """Check if any element is True."""
        if axis is None:
            return bool(self._data.any().compute())
        dim = "time" if axis == 0 or axis == "time" else "distance"
        return self._data.any(dim=dim).compute().values

    def all(self, axis: Optional[Union[int, str]] = None) -> Union[bool, np.ndarray]:
        """Check if all elements are True."""
        if axis is None:
            return bool(self._data.all().compute())
        dim = "time" if axis == 0 or axis == "time" else "distance"
        return self._data.all(dim=dim).compute().values

    def mean(self, axis: Optional[int] = 0) -> np.ndarray:
        """计算均值。"""
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

    def rms(self, window: Optional[int] = None) -> Union[np.ndarray, "DASFrame"]:
        """计算 RMS（均方根）。"""
        data_sq = self._data**2
        if window is None:
            return np.sqrt(data_sq.mean(dim="time").compute().values)
        else:
            rolling_mean = data_sq.rolling(time=window, center=True).mean()
            rms_data = np.sqrt(rolling_mean)
            return DASFrame(rms_data, self._fs, self._dx, **self._metadata)

    def fft(self) -> "DASFrame":
        """快速傅立叶变换 (Real FFT)。"""

        def _fft_func(data):
            return np.abs(np.fft.rfft(data, axis=-1))

        n_samples = self._data.sizes["time"]
        freqs = np.fft.rfftfreq(n_samples, 1 / self._fs)

        spectrum = xr.apply_ufunc(
            _fft_func,
            self._data,
            input_core_dims=[["time"]],
            output_core_dims=[["frequency"]],
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes": {"frequency": len(freqs)}},
        )

        spectrum = spectrum.assign_coords(frequency=freqs)
        spectrum = spectrum.transpose("frequency", "distance")

        return DASFrame(spectrum, self._fs, self._dx, **self._metadata)

    def hilbert(self) -> "DASFrame":
        """希尔伯特变换，返回解析信号。"""

        def _hilbert_func(data):
            return signal.hilbert(data, axis=-1)

        analytical = xr.apply_ufunc(
            _hilbert_func,
            self._data,
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
        """空间平滑处理。"""
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
        """Time domain integration."""
        integrated = self._data.cumsum(dim="time") / self._fs
        return DASFrame(integrated, self._fs, self._dx, **self._metadata)

    def differentiate(self) -> "DASFrame":
        """Time domain differentiation."""
        differentiated = self._data.diff(dim="time") * self._fs
        differentiated = differentiated.pad(time=(1, 0), mode="edge")
        return DASFrame(differentiated, self._fs, self._dx, **self._metadata)

    def stft(self, nperseg: int = 256, noverlap: Optional[int] = None, window: str = "hann") -> "DASFrame":
        """短时傅立叶变换，进行时频分析。"""
        from scipy.signal import ShortTimeFFT, get_window

        if noverlap is None:
            noverlap = nperseg // 2

        hop = nperseg - noverlap
        win = get_window(window, nperseg)
        SFT = ShortTimeFFT(win, hop=hop, fs=self._fs, scale_to="magnitude")

        def _stft_block(block, sft_obj):
            Zxx = sft_obj.stft(block)
            return np.abs(Zxx).astype(np.float32)

        freqs = SFT.f
        n_t = self._data.sizes["time"]
        t_coords = SFT.t(n_t)

        stft_data = xr.apply_ufunc(
            _stft_block,
            self._data,
            kwargs={"sft_obj": SFT},
            input_core_dims=[["time"]],
            output_core_dims=[["frequency", "time_stft"]],
            exclude_dims=set(["time"]),
            dask="parallelized",
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={"output_sizes": {"frequency": len(freqs), "time_stft": len(t_coords)}},
        )

        stft_data = stft_data.rename({"time_stft": "time"})
        stft_data = stft_data.assign_coords({"frequency": freqs, "time": t_coords, "distance": self._data.distance})

        stft_data = stft_data.transpose("frequency", "time", "distance")

        return DASFrame(stft_data, self._fs, self._dx, **self._metadata)

    def fk_filter(
        self,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        dx: Optional[float] = None,
    ) -> "DASFrame":
        """F-K 滤波 (速度滤波)。"""
        from ..config.sampling_config import SamplingConfig
        from ..processing.das_processor import DASProcessor

        if dx is None:
            dx = self._dx

        def _fk_func(block, fs, v_min, v_max, dx):
            config = SamplingConfig(fs=fs, channels=block.shape[1])
            processor = DASProcessor(config)
            return processor.FKFilter(block, v_min=v_min, v_max=v_max, dx=dx)

        filtered = xr.apply_ufunc(
            _fk_func,
            self._data,
            kwargs={"fs": self._fs, "v_min": v_min, "v_max": v_max, "dx": dx},
            input_core_dims=[["time", "distance"]],
            output_core_dims=[["time", "distance"]],
            dask="parallelized",
            output_dtypes=[self._data.dtype],
        )
        return DASFrame(filtered, self._fs, dx, **self._metadata)

    def threshold_detect(self, threshold: Optional[float] = None, sigma: float = 3.0) -> "DASFrame":
        """阈值检测。"""
        if threshold is None:
            mean_val = self._data.mean()
            std_val = self._data.std()
            threshold = float(mean_val.compute() + sigma * std_val.compute())

        detected = np.abs(self._data) > threshold
        return DASFrame(detected, self._fs, self._dx, **self._metadata)

    def sta_lta(self, n_sta: int, n_lta: int) -> "DASFrame":
        """STA/LTA 能量比检测。"""
        from ..analysis.event_detection import sta_lta

        def _sta_lta_func(block, ns, nl):
            return sta_lta(block, ns, nl, axis=-1)

        ratio = xr.apply_ufunc(
            _sta_lta_func,
            self._data,
            kwargs={"ns": n_sta, "nl": n_lta},
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=[self._data.dtype],
        )

        return DASFrame(ratio, self._fs, self._dx, **self._metadata)

    def plot_ts(
        self,
        ch: Optional[int] = None,
        title: str = "Time Series",
        ax: Optional[Axes] = None,
        max_samples: int = 5000,
        **kwargs,
    ) -> Figure:
        """绘制时间序列。"""
        if ch is None:
            frame = self.slice(x=slice(0, min(5, self.shape[1])))
        else:
            frame = self.slice(x=slice(ch, ch + 1))

        nt = frame.shape[0]
        if nt > max_samples:
            step = int(np.ceil(nt / max_samples))
            frame = frame.slice(t=slice(0, nt, step))

        data = frame.collect()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            if ax.figure is None:
                raise ValueError("Provided ax must belong to a figure")
            fig = cast(Figure, ax.figure)

        t = frame._data.time.values
        if ch is None:
            ax.plot(t, data, alpha=0.7, **kwargs)
        else:
            ax.plot(t, data[:, 0], **kwargs)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig

    def plot_spectrum(
        self,
        ch: int = 0,
        title: str = "Spectrum",
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Figure:
        """绘制频谱图 (FFT)。"""
        ch_data = self._data.isel(distance=ch).compute().values

        from ..visualization.das_visualizer import SpectrumPlot

        plotter = SpectrumPlot()
        from scipy import signal

        n = len(ch_data)
        nperseg = min(2048, n // 4)
        freqs, psd = signal.welch(ch_data, fs=self._fs, nperseg=nperseg, scaling="spectrum")
        mags = np.sqrt(psd)

        fig = plotter.plot(freqs, mags, title=title, ax=ax, **kwargs)
        return fig

    def plot_spectrogram(
        self,
        ch: int = 0,
        title: str = "Spectrogram",
        window_size: int = 1024,
        overlap: float = 0.75,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Figure:
        """绘制时频图 (STFT)。"""
        ch_data = self._data.isel(distance=ch).compute().values

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
        ax: Optional[Axes] = None,
        max_samples: int = 2000,
        **kwargs: Any,
    ) -> Figure:
        """绘制热图（瀑布图）。"""
        if t_range is not None or channels is not None:
            import warnings

            warnings.warn("Using t_range or channels in plot_heatmap is deprecated. Please use .slice() instead.")
            frame = self.slice(t=t_range or slice(None), x=channels or slice(None))
        else:
            frame = self

        nt, nx = frame.shape
        if nt > max_samples:
            step = int(np.ceil(nt / max_samples))
            frame = frame.slice(t=slice(0, nt, step))

        data = frame.collect()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            if ax.figure is None:
                raise ValueError("Provided ax must belong to a figure")
            fig = cast(Figure, ax.figure)

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
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Figure:
        """绘制 FK 谱图。"""
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
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Figure:
        """绘制空间剖面图。"""
        frame = self if channels is None else self.slice(x=channels)

        if stat == "rms":
            values = cast(np.ndarray, frame.rms())
            default_title = "RMS Profile"
            default_ylabel = "RMS Amplitude"
        elif stat == "mean":
            values = cast(np.ndarray, frame.mean(axis=0).flatten())
            default_title = "Mean Profile"
            default_ylabel = "Mean Amplitude"
        elif stat == "std":
            values = cast(np.ndarray, frame.std(axis=0).flatten())
            default_title = "Standard Deviation Profile"
            default_ylabel = "Std Amplitude"
        elif stat == "max":
            values = cast(np.ndarray, frame.max(axis=0).flatten())
            default_title = "Max Profile"
            default_ylabel = "Max Amplitude"
        elif stat == "min":
            values = cast(np.ndarray, frame.min(axis=0).flatten())
            default_title = "Min Profile"
            default_ylabel = "Min Amplitude"
        else:
            raise ValueError(f"Unsupported stat: {stat}")

        from ..visualization.das_visualizer import ProfilePlot

        plotter = ProfilePlot()

        if x_axis == "channel":
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
    ) -> Figure:
        """绘制 RMS 剖面图。"""
        return self.plot_profile(stat="rms", channels=channels, x_axis=x_axis, **kwargs)

    def plot_mean(
        self,
        channels: Optional[builtins.slice] = None,
        x_axis: str = "channel",
        **kwargs: Any,
    ) -> Figure:
        """绘制均值剖面图。"""
        return self.plot_profile(stat="mean", channels=channels, x_axis=x_axis, **kwargs)

    def plot_std(
        self,
        channels: Optional[builtins.slice] = None,
        x_axis: str = "channel",
        **kwargs: Any,
    ) -> Figure:
        """绘制标准差剖面图。"""
        return self.plot_profile(stat="std", channels=channels, x_axis=x_axis, **kwargs)

    def to_obspy(self) -> Any:
        """Convert to ObsPy Stream."""
        try:
            from obspy import Stream, Trace, UTCDateTime
        except ImportError:
            raise ImportError("obspy is required for to_obspy(). Install it with `pip install obspy`.")

        data = self.collect()
        nt, nx = data.shape

        start_time = UTCDateTime(0)
        inv = self._metadata.get("inventory")
        if inv and hasattr(inv, "acquisition") and inv.acquisition.start_time:
            start_time = UTCDateTime(inv.acquisition.start_time)

        traces = []
        for i in range(nx):
            stats = {
                "network": "DA",
                "station": f"{i:04d}",
                "channel": "HSF",
                "sampling_rate": self._fs,
                "starttime": start_time,
                "npts": nt,
            }
            tr = Trace(data=data[:, i].astype(np.float32), header=stats)
            traces.append(tr)

        return Stream(traces=traces)

    @classmethod
    def from_obspy(cls, stream: Any, dx: float = 1.0) -> "DASFrame":
        """Create DASFrame from ObsPy Stream."""
        import numpy as np

        data = np.stack([tr.data for tr in stream], axis=1)
        fs = stream[0].stats.sampling_rate

        metadata = {
            "network": stream[0].stats.network,
            "starttime": str(stream[0].stats.starttime),
        }

        return cls(data, fs=fs, dx=dx, **metadata)

    def to_dascore(self) -> Any:
        """Convert to DASCore Patch."""
        try:
            import dascore as dc  # type: ignore
        except ImportError:
            raise ImportError("dascore is required for to_dascore(). Install it with `pip install dascore`.")

        return dc.Patch(
            data=self.collect(),
            coords={
                "time": self._data.time.values,
                "distance": self._data.distance.values,
            },
            dims=("time", "distance"),
            attrs=self._metadata,
        )

    # --- Units and Physical Quantities ---

    # --- Units and Physical Quantities ---

    def to_standard_units(self) -> "DASFrame":
        """将数据转换为标准化的物理单位 (SI)。

        支持的转换逻辑：
        1. Phase (rad) -> Strain (m/m)
        2. Phase Rate (rad/s) -> Strain Rate (m/m/s)
        """
        inv = self.inventory
        if not inv or not inv.acquisition:
            warnings.warn("缺少 Inventory/Acquisition 信息，无法自动转换单位")
            return self

        current_unit = self.get_unit()
        if current_unit == "unknown":
            return self

        # 核心转换逻辑：Phase to Strain
        # Formula: strain = (lambda * phase) / (4 * pi * n * L * G)
        if current_unit in ["rad", "radians", "rad/s"]:
            # 获取必要参数
            wavelength = 1550.0  # Default nm
            if inv.interrogator and inv.interrogator.wavelength:
                wavelength = inv.interrogator.wavelength

            gl = 10.0  # Default m
            if inv.fiber and inv.fiber.gauge_length:
                gl = inv.fiber.gauge_length

            # 常数
            n_refractive = 1.46
            G_factor = 0.78

            # 计算比例系数 (rad -> strain)
            # lambda is in nm, convert to m: * 1e-9
            scale = (wavelength * 1e-9) / (4 * np.pi * n_refractive * gl * G_factor)

            target_unit = "strain" if current_unit != "rad/s" else "strain_rate"

            # 应用转换
            return self.scale(scale)._update_unit_metadata(target_unit)

        return self

    def _update_unit_metadata(self, unit_name: str) -> "DASFrame":
        """Internal helper to update unit metadata."""
        new_meta = self._metadata.copy()
        if "inventory" in new_meta:
            new_meta["inventory"].acquisition.data_unit = unit_name
        new_meta["units"] = unit_name
        self._metadata = new_meta
        return self

    def get_unit(self) -> Any:
        """Get the current data unit from metadata."""
        inv = self.inventory
        if inv and inv.acquisition:
            return inv.acquisition.data_unit
        return self._metadata.get("units", "unknown")

    def convert_units(self, target_unit: str) -> "DASFrame":
        """Explicitly convert data to a target unit using Pint."""
        from ..units import ureg

        current_unit = self.get_unit()
        if current_unit == "unknown":
            raise ValueError("Current units are unknown, cannot convert.")

        q = ureg.Quantity(self.collect(), current_unit)
        converted = q.to(target_unit).magnitude

        return DASFrame(converted, fs=self._fs, dx=self._dx)._update_unit_metadata(target_unit)
