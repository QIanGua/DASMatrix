"""
Real-time Visualization
=======================

High-performance visualization for real-time DAS stream monitoring.
Uses matplotlib blitting to achieve high FPS.
"""

import time
from typing import Any, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from ..visualization.styles import apply_nature_style


class RealtimeVisualizer:
    """
    A visualizer optimized for real-time data updates using blitting.
    """

    def __init__(
        self,
        n_channels: int,
        duration: float,
        fs: float,
        title: str = "Real-time Monitoring",
        figsize: Tuple[int, int] = (10, 8),
    ):
        """
        Initialize the visualizer.

        Args:
            n_channels: Number of channels.
            duration: Window duration in seconds.
            fs: Sampling rate.
            title: Window title.
        """
        self.n_channels = n_channels
        self.duration = duration
        self.fs = fs

        # Setup plot
        apply_nature_style()
        self.fig, self.axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
        self.ax_heatmap: Axes = self.axes[0]
        self.ax_line: Axes = self.axes[1]

        self.fig.suptitle(title, weight="bold")

        # --- Heatmap Setup ---
        # Initialize with zeros
        n_samples = int(duration * fs)
        self.extent = [0, duration, 0, n_channels]

        self.im: AxesImage = self.ax_heatmap.imshow(
            np.zeros((n_channels, n_samples), dtype=np.float32),
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            interpolation="nearest",
            vmin=-1,
            vmax=1,
            animated=True,  # Critical for blitting
            extent=self.extent,
        )
        self.ax_heatmap.set_ylabel("Channel")
        self.ax_heatmap.set_title("Waterfall / Heatmap")

        # --- Line Plot Setup ---
        self.x_data = np.linspace(0, duration, n_samples)
        self.line: Line2D = self.ax_line.plot(
            self.x_data,
            np.zeros(n_samples),
            animated=True,
            color="#0072B2",
            linewidth=1.0,
        )[0]
        self.ax_line.set_xlabel("Time (s)")
        self.ax_line.set_ylabel("Amplitude")
        self.ax_line.set_ylim(-1, 1)
        self.ax_line.set_xlim(0, duration)
        self.ax_line.set_title("Selected Channel Trace")
        self.ax_line.grid(True, alpha=0.3)

        # Blitting cache
        self.bg_cache = None

        # FPS counter
        self._last_frame_time = time.time()
        self._fps_text = self.fig.text(0.95, 0.95, "FPS: 0", ha="right")

        # Connect draw event to capture background
        self.cid = self.fig.canvas.mpl_connect("draw_event", self._on_draw)

        plt.show(block=False)
        plt.pause(0.1)

    def _on_draw(self, event):
        """Capture background on initial draw or resize."""
        if event is not None:
            # Cast to Any to suppress linter warnings about copy_from_bbox
            canvas = cast(Any, self.fig.canvas)
            self.bg_cache = canvas.copy_from_bbox(self.fig.bbox)
            self._draw_artists()

    def _draw_artists(self):
        """Draw animated artists."""
        self.ax_heatmap.draw_artist(self.im)
        self.ax_line.draw_artist(self.line)
        # self.fig.canvas.blit(self.fig.bbox) # Don't blit here, done in update

    def update(self, heatmap_data: np.ndarray, line_data: Optional[np.ndarray] = None):
        """
        Update the plot with new data.

        Args:
            heatmap_data: 2D array (samples, channels) - Note: will be transposed for imshow.
            line_data: 1D array (samples,) for the line plot.
        """
        # Update FPS
        now = time.time()
        dt = now - self._last_frame_time
        if dt > 0:
            fps = 1.0 / dt
            self._fps_text.set_text(f"FPS: {fps:.1f}")
        self._last_frame_time = now

        # Update Heatmap
        # imshow expects (rows/channels, cols/time)
        # Input is usually (time, channels), so transpose
        self.im.set_data(heatmap_data.T)

        # Auto-scale color limits occasionally or if fixed?
        # For performance, prefer fixed, or explicit dynamic
        # vmin, vmax = np.percentile(heatmap_data, [2, 98])
        # self.im.set_clim(vmin, vmax)

        # Update Line
        if line_data is not None:
            # Ensure length matches
            if len(line_data) == len(self.x_data):
                self.line.set_ydata(line_data)

        # Blitting
        if self.bg_cache:
            canvas = cast(Any, self.fig.canvas)
            canvas.restore_region(self.bg_cache)
            self.ax_heatmap.draw_artist(self.im)
            self.ax_line.draw_artist(self.line)
            canvas.blit(self.fig.bbox)
            canvas.flush_events()
        else:
            self.fig.canvas.draw()

    def close(self):
        plt.close(self.fig)
