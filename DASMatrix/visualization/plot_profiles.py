"""Additional plotters extracted from das_visualizer for modularity."""

from typing import Any, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np

from .plot_base import PlotBase


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
        """绘制 F-K 谱图。"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize_standard)
        else:
            fig = ax.figure

        mag = np.abs(fk_spectrum)
        mag_db = 20 * np.log10(mag + 1e-12)
        mag_db = mag_db - np.max(mag_db)

        if db_range:
            vmin, vmax = db_range
        else:
            vmax = 0
            vmin = -60

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
        ax.set_title(title or "F-K Spectrum", fontweight="bold")

        if v_lines:
            k_min, k_max = wavenumbers[0], wavenumbers[-1]
            f_min, f_max = freqs[0], freqs[-1]
            k_line = np.linspace(k_min, k_max, 100)
            for v in v_lines:
                f_line = v * k_line
                mask = (f_line >= f_min) & (f_line <= f_max)
                if np.any(mask):
                    ax.plot(k_line[mask], f_line[mask], "w--", alpha=0.7, linewidth=1)
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
    """剖面图 (Profile Plot) 绘图类。"""

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
        """绘制剖面图。"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            if ax.figure is None:
                raise ValueError("Provided ax must belong to a figure")
            fig = cast(plt.Figure, ax.figure)

        if distances is None:
            distances = np.arange(len(values))
            if xlabel == "Channel":
                xlabel = "Channel Index"

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
