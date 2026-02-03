import builtins
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, cast

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import signal
from scipy.ndimage import median_filter as scipy_median_filter

from ..core.inventory import DASInventory

if TYPE_CHECKING:
    from ..core.event import EventCatalog


class DASFrame:
    """DAS 数据处理核心类 (Xarray/Dask Backend)。

    所有信号处理操作基于 xarray 和 dask 进行延迟计算。
    """

    _data: xr.DataArray
    _fs: float
    _dx: float
    _metadata: dict[str, Any]
    _op_log: List[dict[str, Any]]
    _op_log_complete: bool
    _source_data: xr.DataArray

    def __init__(
        self,
        data: Union[xr.DataArray, np.ndarray, da.Array, "DASFrame"],
        fs: float,
        dx: float = 1.0,
        **metadata: Any,
    ) -> None:
        """Initialize DASFrame."""
        # Avoid passing fs/dx through metadata to prevent duplicate kwargs
        metadata = {k: v for k, v in metadata.items() if k not in ("fs", "dx")}
        if isinstance(data, DASFrame):
            self._data = data._data
            self._fs = data.fs
            self._dx = getattr(data, "_dx", dx)
            self._metadata = {**data._metadata, **metadata}
            self._op_log = list(getattr(data, "_op_log", []))
            self._op_log_complete = bool(getattr(data, "_op_log_complete", True))
            self._source_data = getattr(data, "_source_data", data._data)
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
            if "time" in data.dims and data.dims[0] != "time" and "frequency" not in data.dims:
                data = data.transpose("time", ...)
            self._data = data
            self._metadata = {**data.attrs, **metadata}
            # Normalize metadata to keep fs/dx only in attrs, not kwargs
            self._metadata.pop("fs", None)
            self._metadata.pop("dx", None)
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

        self._op_log = []
        self._op_log_complete = True
        self._source_data = self._data

    def _spawn(
        self,
        data: xr.DataArray,
        op: Optional[str] = None,
        op_kwargs: Optional[dict[str, Any]] = None,
        op_supported: bool = True,
    ) -> "DASFrame":
        """Create a new DASFrame while preserving source data and op log."""
        frame = DASFrame(data, self._fs, self._dx, **self._metadata)
        frame._source_data = self._source_data
        frame._op_log = list(self._op_log)
        frame._op_log_complete = self._op_log_complete
        if op is not None:
            if frame._op_log_complete and op_supported:
                frame._op_log.append({"op": op, "kwargs": op_kwargs or {}})
            else:
                frame._op_log_complete = False
        return frame

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

    def collect(self, engine: str = "xarray") -> np.ndarray:
        """Compute and return numpy array."""
        if engine == "xarray":
            return self._data.compute().values
        if engine != "hybrid":
            raise ValueError(f"Unsupported engine: {engine}")

        if not self._op_log_complete or not self._op_log:
            warnings.warn("Hybrid engine fallback to xarray due to incomplete op log.")
            return self._data.compute().values

        from ..core.computation_graph import ComputationGraph, NodeDomain, OperationNode
        from ..processing.engine import HybridEngine

        base = self._source_data.compute().values
        graph = ComputationGraph.leaf(base)
        current = graph.root
        for item in self._op_log:
            node = OperationNode(
                item["op"],
                [current],
                kwargs=item.get("kwargs", {}),
                domain=NodeDomain.SIGNAL,
            )
            graph = graph.add_node(node)
            current = node

        engine_impl = HybridEngine()
        return engine_impl.compute(graph)

    def astype(self, dtype: Any) -> "DASFrame":
        """Cast data to a new type."""
        return DASFrame(self._data.astype(dtype), self._fs, self._dx, **self._metadata)

    def flatten(self) -> np.ndarray:
        """Return a flattened version of the data."""
        return self.collect().flatten()

    def slice(self, t: slice = slice(None), x: slice = slice(None)) -> "DASFrame":
        """Slice data."""
        sliced = self._data.isel(time=t, distance=x)
        return self._spawn(sliced, op="slice", op_kwargs={"t": t, "x": x})

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
            dask_gufunc_kwargs={"allow_rechunk": True},
            output_dtypes=[data.dtype],
        )

        return self._spawn(
            filtered,
            op="bandpass",
            op_kwargs={"low": low, "high": high, "order": order, "fs": self._fs},
        )

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
        return self._spawn(
            filtered,
            op="lowpass",
            op_kwargs={"cutoff": cutoff, "order": order, "fs": self._fs},
        )

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
        return self._spawn(
            filtered,
            op="highpass",
            op_kwargs={"cutoff": cutoff, "order": order, "fs": self._fs},
        )

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
        return self._spawn(
            filtered,
            op="notch",
            op_kwargs={"freq": freq, "Q": Q, "fs": self._fs},
        )

    def median_filter(self, k: int = 5, axis: str = "time") -> "DASFrame":
        """中值滤波。"""
        if axis == "time":
            size = (k, 1)
        else:
            size = (1, k)

        def _median_filter_func(block, size_val):
            return scipy_median_filter(block, size=size_val)

        filtered = xr.apply_ufunc(
            _median_filter_func,
            self._data,
            kwargs={"size_val": size},
            input_core_dims=[["time", "distance"]],
            output_core_dims=[["time", "distance"]],
            dask="parallelized",
            dask_gufunc_kwargs={"allow_rechunk": True},
            output_dtypes=[self._data.dtype],
        )

        return self._spawn(
            filtered,
            op="median_filter",
            op_kwargs={"k": k, "axis": axis},
        )

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
            dask_gufunc_kwargs={"allow_rechunk": True},
            output_dtypes=[self._data.dtype],
        )

        op_supported = axis == "time"
        return self._spawn(
            detrended,
            op="detrend",
            op_supported=op_supported,
        )

    def demean(self, axis: str = "time") -> "DASFrame":
        """Remove mean value along specified axis."""
        dim = "time" if axis == "time" else "distance"
        demeaned = self._data - self._data.mean(dim=dim)
        op_supported = axis == "time"
        return self._spawn(
            demeaned,
            op="demean",
            op_supported=op_supported,
        )

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

        return self._spawn(
            normalized,
            op="normalize",
            op_kwargs={"method": method},
        )

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

        return self._spawn(spectrum, op="fft")

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
        return self._spawn(analytical, op="hilbert")

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
        return self._spawn(scaled, op="scale", op_kwargs={"factor": factor})

    def abs(self) -> "DASFrame":
        """Absolute value."""
        return self._spawn(np.abs(self._data), op="abs")

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
        from scipy import signal as scipy_signal

        if noverlap is None:
            noverlap = nperseg // 2

        data = self.collect()
        n_channels = data.shape[1] if data.ndim > 1 else 1
        results = []
        for ch in range(n_channels):
            ch_data = data[:, ch] if data.ndim > 1 else data
            f, t, Zxx = scipy_signal.stft(
                ch_data,
                fs=self._fs,
                nperseg=nperseg,
                noverlap=noverlap,
                window=window,
            )
            results.append(np.abs(Zxx).astype(np.float32))

        stft_arr = np.stack(results, axis=-1)
        stft_data = xr.DataArray(
            stft_arr,
            dims=("frequency", "time", "distance"),
            coords={
                "frequency": f,
                "time": t,
                "distance": self._data.distance.values,
            },
            attrs={**self._metadata, "fs": self._fs, "dx": self._dx},
        )

        return self._spawn(
            stft_data,
            op="stft",
            op_kwargs={"nperseg": nperseg, "noverlap": noverlap, "fs": self._fs, "window": window},
            op_supported=True,
        )

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
        return self._spawn(
            filtered,
            op="fk_filter",
            op_kwargs={"v_min": v_min, "v_max": v_max, "dx": dx, "fs": self._fs},
        )

    def threshold_detect(self, threshold: Optional[float] = None, sigma: float = 3.0) -> "DASFrame":
        """阈值检测。"""
        if threshold is None:
            mean_val = self._data.mean().compute()
            std_val = self._data.std().compute()
            threshold = float(mean_val + sigma * std_val)

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

    def trigger_detection(
        self,
        threshold: float,
        trigger_off: Optional[float] = None,
        lazy: bool = True,
    ) -> "EventCatalog":
        """基于阈值检测事件并返回 EventCatalog。

        Args:
            threshold: 触发阈值。
            trigger_off: 结束阈值 (默认为 threshold)。
            lazy: 是否避免加载完整数据 (默认 True)。

        Returns:
            EventCatalog: 包含检测到的事件列表。
        """
        from datetime import datetime, timedelta

        from ..core.event import DASEvent, EventCatalog

        if trigger_off is None:
            trigger_off = threshold

        # 1. 获取检测掩码 (Lazy)
        if lazy:
            mask = np.abs(self._data) > threshold
        else:
            mask = self.abs().collect() > threshold

        # 2. 简单的连通域分析或逐通道触发
        # 为了性能，这里先实现简单的逐通道触发逻辑
        # TODO: 使用 skimage.measure.label 进行 2D 连通域标记以支持时空事件

        events = []
        nt = int(mask.shape[0])
        t_coords = self._data.time.values

        # 获取绝对开始时间
        base_time = datetime.now()
        inv = self.inventory
        if inv and inv.acquisition and inv.acquisition.start_time:
            # Handle potential string or datetime
            st = inv.acquisition.start_time
            if isinstance(st, str):
                try:
                    base_time = datetime.fromisoformat(st.replace("Z", "+00:00"))
                except Exception:
                    pass
            elif isinstance(st, datetime):
                base_time = st

        # 简单的全图遍历 (优化：使用 numba 加速这部分)
        # 这里仅作演示，实际应使用更高效的算法
        # 寻找每一列的触发区间

        # 简化策略：只要任意通道触发，就算一个事件
        # 对空间轴取 max，得到时间轴上的 1D 触发序列
        if lazy:
            time_trigger = mask.any(dim="distance").compute().values.astype(int)
        else:
            time_trigger = np.any(mask, axis=1).astype(int)
        diff = np.diff(time_trigger, prepend=0)

        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        # 处理边界
        if len(starts) > len(ends):
            ends = np.append(ends, nt)

        for i, (s, e) in enumerate(zip(starts, ends)):
            # 找到触发的通道范围
            if lazy:
                event_slice = mask.isel(time=slice(s, e))
                ch_mask = event_slice.any(dim="time").compute().values
            else:
                event_slice = mask[s:e, :]
                ch_mask = np.any(event_slice, axis=0)
            ch_indices = np.where(ch_mask)[0]

            if len(ch_indices) == 0:
                continue

            min_ch, max_ch = ch_indices[0], ch_indices[-1]

            evt = DASEvent(
                id=f"evt_{base_time.timestamp()}_{i}",
                start_time=base_time + timedelta(seconds=float(t_coords[s])),
                end_time=base_time + timedelta(seconds=float(t_coords[e - 1])),
                min_channel=int(min_ch),
                max_channel=int(max_ch),
                confidence=1.0,
                event_type="trigger",
            )
            events.append(evt)

        return EventCatalog(events)

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
        max_samples: int = 20000,
        **kwargs: Any,
    ) -> Figure:
        """绘制频谱图 (FFT)。"""
        ch_data = self._data.isel(distance=ch)
        n = int(ch_data.shape[0])
        step = 1
        if max_samples and n > max_samples:
            step = int(np.ceil(n / max_samples))
            ch_data = ch_data.isel(time=slice(0, n, step))

        ch_data = ch_data.compute().values
        fs_eff = self._fs / step

        from ..visualization.das_visualizer import SpectrumPlot

        plotter = SpectrumPlot()
        from scipy import signal

        n = len(ch_data)
        nperseg = min(2048, max(8, n // 4))
        freqs, psd = signal.welch(ch_data, fs=fs_eff, nperseg=nperseg, scaling="spectrum")
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
        max_samples: int = 20000,
        **kwargs: Any,
    ) -> Figure:
        """绘制时频图 (STFT)。"""
        ch_data = self._data.isel(distance=ch)
        n = int(ch_data.shape[0])
        step = 1
        if max_samples and n > max_samples:
            step = int(np.ceil(n / max_samples))
            ch_data = ch_data.isel(time=slice(0, n, step))

        ch_data = ch_data.compute().values
        fs_eff = self._fs / step

        from ..visualization.das_visualizer import SpectrogramPlot

        plotter = SpectrogramPlot()
        fig = plotter.plot(
            ch_data,
            fs_eff,
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
        events: Optional["EventCatalog"] = None,
        **kwargs: Any,
    ) -> Figure:
        """绘制热图（瀑布图）。"""
        if t_range is not None or channels is not None:
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

        # 绘制事件叠加层
        if events is not None and ax is not None:
            from datetime import datetime

            import matplotlib.patches as patches

            # 需要将 datetime 转换为绘图坐标 (时间轴相对秒数)
            # 获取当前 frame 的基准时间
            base_time = None
            inv = self.inventory
            if inv and inv.acquisition and inv.acquisition.start_time:
                st = inv.acquisition.start_time
                if isinstance(st, str):
                    try:
                        base_time = datetime.fromisoformat(st.replace("Z", "+00:00"))
                    except Exception:
                        pass
                elif isinstance(st, datetime):
                    base_time = st

            if base_time:
                for row in events.df.iter_rows(named=True):
                    # 计算相对时间
                    t_start = (row["start_time"] - base_time).total_seconds()
                    t_end = (row["end_time"] - base_time).total_seconds() if row["end_time"] else t_start + 1.0

                    # 检查是否在当前显示范围内
                    # time_coords 是相对时间数组
                    if t_end < time_coords[0] or t_start > time_coords[-1]:
                        continue

                    # 绘制矩形框
                    # x轴是 channel (distance), y轴是 time (seconds)
                    # extent = (dist_min, dist_max, time_max, time_min)
                    # 注意 imshow 的 extent y 轴通常是反向的或者从上到下的

                    # 转换为 channel index 或 distance
                    # 这里的 plot_heatmap 使用的是 dist_coords (物理距离) 或 channel index
                    # extent[0] = dist_start, extent[1] = dist_end

                    x_start = row["min_channel"] * self._dx
                    x_width = (row["max_channel"] - row["min_channel"]) * self._dx
                    y_start = t_start
                    height = t_end - t_start

                    rect = patches.Rectangle(
                        (x_start, y_start), x_width, height, linewidth=1, edgecolor="r", facecolor="none", alpha=0.8
                    )
                    ax.add_patch(rect)
                    ax.text(x_start, y_start, row["event_type"], color="red", fontsize=8)

        return fig

    def plot_fk(
        self,
        dx: float = 1.0,
        title: str = "F-K Spectrum",
        v_lines: Optional[List[float]] = None,
        ax: Optional[Axes] = None,
        max_samples: int = 2000,
        max_channels: int = 512,
        **kwargs: Any,
    ) -> Figure:
        """绘制 FK 谱图。"""
        frame = self
        nt, nx = frame.shape
        t_step = 1
        x_step = 1
        if max_samples and nt > max_samples:
            t_step = int(np.ceil(nt / max_samples))
            frame = frame.slice(t=slice(0, nt, t_step))
        if max_channels and nx > max_channels:
            x_step = int(np.ceil(nx / max_channels))
            frame = frame.slice(x=slice(0, nx, x_step))

        data = frame.collect()

        from ..config.sampling_config import SamplingConfig
        from ..processing.das_processor import DASProcessor

        fs_eff = self._fs / t_step
        dx_eff = dx * x_step

        config = SamplingConfig(fs=fs_eff, channels=data.shape[1])
        processor = DASProcessor(config)

        fk, freqs, k = processor.f_k_transform(data)
        k = k / dx_eff

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

    # --- Analysis and Detection ---

    def template_match(self, template: Union[np.ndarray, "DASFrame"], axis: str = "time") -> "DASFrame":
        """Perform Normalized Cross-Correlation (NCC) template matching.

        If template is 1D, it matches along the specified axis (independently per channel if axis=time).
        If template is 2D, it performs spatio-temporal matching.
        """
        from ..analysis.template_matching import sliding_ncc_1d, sliding_ncc_2d

        if isinstance(template, DASFrame):
            t_data = template.collect()
        else:
            t_data = np.asarray(template)

        if t_data.ndim == 1:

            def _match_func(block, temp):
                return sliding_ncc_1d(block, temp)

            # Apply along core dimension
            res = xr.apply_ufunc(
                _match_func,
                self._data,
                kwargs={"temp": t_data},
                input_core_dims=[[axis]],
                output_core_dims=[[f"{axis}_match"]],
                exclude_dims=set([axis]),
                dask="parallelized",
                output_dtypes=[self._data.dtype],
                dask_gufunc_kwargs={"output_sizes": {f"{axis}_match": self._data.sizes[axis] - t_data.shape[0] + 1}},
            )
            return DASFrame(res.rename({f"{axis}_match": axis}), self._fs, self._dx, **self._metadata)

        elif t_data.ndim == 2:
            # Spatio-temporal matching (2D convolution)
            # This is complex for apply_ufunc if distance chunks > 1.
            # For simplicity, handle time axis matching per spatial block.
            def _match_2d_func(block, temp):
                return sliding_ncc_2d(block, temp)

            # For 2D, we treat both dims as core
            res = xr.apply_ufunc(
                _match_2d_func,
                self._data,
                kwargs={"temp": t_data},
                input_core_dims=[["time", "distance"]],
                output_core_dims=[["time_match", "distance_match"]],
                exclude_dims=set(["time", "distance"]),
                dask="parallelized",
                output_dtypes=[self._data.dtype],
                dask_gufunc_kwargs={
                    "output_sizes": {
                        "time_match": self._data.sizes["time"] - t_data.shape[0] + 1,
                        "distance_match": self._data.sizes["distance"] - t_data.shape[1] + 1,
                    }
                },
            )
            return DASFrame(
                res.rename({"time_match": "time", "distance_match": "distance"}), self._fs, self._dx, **self._metadata
            )

        return self
