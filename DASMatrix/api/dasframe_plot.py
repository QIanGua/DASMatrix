"""Plotting helpers for DASFrame."""

import logging
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import signal

if TYPE_CHECKING:
    from ..core.event import EventCatalog
    from .dasframe import DASFrame

logger = logging.getLogger(__name__)


def plot_ts(
    frame: "DASFrame",
    ch: Optional[int] = None,
    title: str = "Time Series",
    ax: Optional[Axes] = None,
    max_samples: int = 5000,
    **kwargs: Any,
) -> Figure:
    """绘制时间序列。"""
    if ch is None:
        current = frame.slice(x=slice(0, min(5, frame.shape[1])))
    else:
        current = frame.slice(x=slice(ch, ch + 1))

    nt = current.shape[0]
    if nt > max_samples:
        step = int(np.ceil(nt / max_samples))
        current = current.slice(t=slice(0, nt, step))

    data = current.collect()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        if ax.figure is None:
            raise ValueError("Provided ax must belong to a figure")
        fig = cast(Figure, ax.figure)

    t = current._data.time.values
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
    frame: "DASFrame",
    ch: int = 0,
    title: str = "Spectrum",
    ax: Optional[Axes] = None,
    max_samples: int = 20000,
    **kwargs: Any,
) -> Figure:
    """绘制频谱图 (FFT)。"""
    ch_data = frame._data.isel(distance=ch)
    n = int(ch_data.shape[0])
    step = 1
    if max_samples and n > max_samples:
        step = int(np.ceil(n / max_samples))
        ch_data = ch_data.isel(time=slice(0, n, step))

    ch_data = ch_data.compute().values
    fs_eff = frame._fs / step

    from ..visualization.das_visualizer import SpectrumPlot

    n = len(ch_data)
    nperseg = min(2048, max(8, n // 4))
    freqs, psd = signal.welch(ch_data, fs=fs_eff, nperseg=nperseg, scaling="spectrum")
    mags = np.sqrt(psd)

    plotter = SpectrumPlot()
    return plotter.plot(freqs, mags, title=title, ax=ax, **kwargs)


def plot_spectrogram(
    frame: "DASFrame",
    ch: int = 0,
    title: str = "Spectrogram",
    window_size: int = 1024,
    overlap: float = 0.75,
    ax: Optional[Axes] = None,
    max_samples: int = 20000,
    **kwargs: Any,
) -> Figure:
    """绘制时频图 (STFT)。"""
    ch_data = frame._data.isel(distance=ch)
    n = int(ch_data.shape[0])
    step = 1
    if max_samples and n > max_samples:
        step = int(np.ceil(n / max_samples))
        ch_data = ch_data.isel(time=slice(0, n, step))

    ch_data = ch_data.compute().values
    fs_eff = frame._fs / step

    from ..visualization.das_visualizer import SpectrogramPlot

    plotter = SpectrogramPlot()
    return plotter.plot(
        ch_data,
        fs_eff,
        window_size=window_size,
        overlap=overlap,
        title=title,
        ax=ax,
        **kwargs,
    )


def plot_heatmap(
    frame: "DASFrame",
    channels: Optional[slice] = None,
    t_range: Optional[slice] = None,
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
        current = frame.slice(t=t_range or slice(None), x=channels or slice(None))
    else:
        current = frame

    nt, _ = current.shape
    if nt > max_samples:
        step = int(np.ceil(nt / max_samples))
        current = current.slice(t=slice(0, nt, step))

    data = current.collect()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        if ax.figure is None:
            raise ValueError("Provided ax must belong to a figure")
        fig = cast(Figure, ax.figure)

    dist_coords = current._data.distance.values / frame._dx
    time_coords = current._data.time.values

    extent = (
        dist_coords[0] - 0.5,
        dist_coords[-1] + 0.5,
        time_coords[-1] + 0.5 / frame._fs,
        time_coords[0] - 0.5 / frame._fs,
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
    plt.colorbar(im, ax=ax, label="Amplitude")

    if events is None:
        return fig

    import matplotlib.patches as patches

    base_time = None
    inv = frame.inventory
    if inv and inv.acquisition and inv.acquisition.start_time:
        st = inv.acquisition.start_time
        if isinstance(st, str):
            try:
                base_time = datetime.fromisoformat(st.replace("Z", "+00:00"))
            except (TypeError, ValueError) as exc:
                logger.debug("Failed to parse event overlay base_time: %s", exc)
        elif isinstance(st, datetime):
            base_time = st

    if not base_time:
        return fig

    for row in events.df.iter_rows(named=True):
        t_start = (row["start_time"] - base_time).total_seconds()
        t_end = (row["end_time"] - base_time).total_seconds() if row["end_time"] else t_start + 1.0
        if t_end < time_coords[0] or t_start > time_coords[-1]:
            continue

        x_start = row["min_channel"] * frame._dx
        x_width = (row["max_channel"] - row["min_channel"]) * frame._dx
        y_start = t_start
        height = t_end - t_start
        rect = patches.Rectangle(
            (x_start, y_start), x_width, height, linewidth=1, edgecolor="r", facecolor="none", alpha=0.8
        )
        ax.add_patch(rect)
        ax.text(x_start, y_start, row["event_type"], color="red", fontsize=8)

    return fig


def plot_fk(
    frame: "DASFrame",
    dx: float = 1.0,
    title: str = "F-K Spectrum",
    v_lines: Optional[list[float]] = None,
    ax: Optional[Axes] = None,
    max_samples: int = 2000,
    max_channels: int = 512,
    **kwargs: Any,
) -> Figure:
    """绘制 FK 谱图。"""
    current = frame
    nt, nx = current.shape
    t_step = 1
    x_step = 1
    if max_samples and nt > max_samples:
        t_step = int(np.ceil(nt / max_samples))
        current = current.slice(t=slice(0, nt, t_step))
    if max_channels and nx > max_channels:
        x_step = int(np.ceil(nx / max_channels))
        current = current.slice(x=slice(0, nx, x_step))

    data = current.collect()

    from ..config.sampling_config import SamplingConfig
    from ..processing.das_processor import DASProcessor
    from ..visualization.das_visualizer import FKPlot

    fs_eff = frame._fs / t_step
    dx_eff = dx * x_step

    config = SamplingConfig(fs=fs_eff, channels=data.shape[1])
    processor = DASProcessor(config)
    fk, freqs, k = processor.f_k_transform(data)
    k = k / dx_eff

    plotter = FKPlot()
    return plotter.plot(fk, freqs, k, title=title, v_lines=v_lines, ax=ax, **kwargs)


def plot_profile(
    frame: "DASFrame",
    stat: str = "rms",
    channels: Optional[slice] = None,
    x_axis: str = "channel",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Figure:
    """绘制空间剖面图。"""
    current = frame if channels is None else frame.slice(x=channels)

    if stat == "rms":
        values = cast(np.ndarray, current.rms())
        default_title = "RMS Profile"
        default_ylabel = "RMS Amplitude"
    elif stat == "mean":
        values = cast(np.ndarray, current.mean(axis=0).flatten())
        default_title = "Mean Profile"
        default_ylabel = "Mean Amplitude"
    elif stat == "std":
        values = cast(np.ndarray, current.std(axis=0).flatten())
        default_title = "Standard Deviation Profile"
        default_ylabel = "Std Amplitude"
    elif stat == "max":
        values = cast(np.ndarray, current.max(axis=0).flatten())
        default_title = "Max Profile"
        default_ylabel = "Max Amplitude"
    elif stat == "min":
        values = cast(np.ndarray, current.min(axis=0).flatten())
        default_title = "Min Profile"
        default_ylabel = "Min Amplitude"
    else:
        raise ValueError(f"Unsupported stat: {stat}")

    from ..visualization.das_visualizer import ProfilePlot

    if x_axis == "channel":
        distances = current._data.distance.values / frame._dx
        kwargs.setdefault("xlabel", "Channel")
    else:
        distances = current._data.distance.values
        kwargs.setdefault("xlabel", "Distance (m)")

    plotter = ProfilePlot()
    return plotter.plot(
        values=values,
        distances=distances,
        title=title or default_title,
        ylabel=ylabel or default_ylabel,
        ax=ax,
        **kwargs,
    )


def plot_rms(
    frame: "DASFrame",
    channels: Optional[slice] = None,
    x_axis: str = "channel",
    **kwargs: Any,
) -> Figure:
    """绘制 RMS 剖面图。"""
    return plot_profile(frame, stat="rms", channels=channels, x_axis=x_axis, **kwargs)


def plot_mean(
    frame: "DASFrame",
    channels: Optional[slice] = None,
    x_axis: str = "channel",
    **kwargs: Any,
) -> Figure:
    """绘制均值剖面图。"""
    return plot_profile(frame, stat="mean", channels=channels, x_axis=x_axis, **kwargs)


def plot_std(
    frame: "DASFrame",
    channels: Optional[slice] = None,
    x_axis: str = "channel",
    **kwargs: Any,
) -> Figure:
    """绘制标准差剖面图。"""
    return plot_profile(frame, stat="std", channels=channels, x_axis=x_axis, **kwargs)
