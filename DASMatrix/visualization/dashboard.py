"""DASMatrix 工业级实时监测看板 (Premium Dashboard)

提供行业顶级的实时数据可视化体验，支持深色模式、中英文切、实时指标追踪及多维度分析。
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Circle

from .styles import apply_nature_style, setup_axis, add_colorbar
from ..config.visualization_config import VisualizationConfig


class DASDashboard:
    """DAS 工业级实时仪表盘 (Premium)

    特点：
    - 深色级黑模式设计 (Dark Mode Aesthetics)
    - 中英文双语支持 (i18n)
    - 状态指示灯 (LED Status Indicators)
    - 焦点通道详情面板 (Focus Channel Analysis)
    - 高性能实时渲染 (High-Performance Blitting)
    """

    # 国际化字典
    I18N = {
        "cn": {
            "title": "DASMatrix 实时在线监测系统",
            "waterfall": "实时时空阵列 (瀑布图)",
            "metrics": "信号指标追踪 (Max/RMS)",
            "log": "事件检测日志",
            "stats": "系统运行状态",
            "ch_detail": "焦点通道分析 (CH: {ch})",
            "status": "系统状态",
            "idle": "就绪 / 待机",
            "active": "正在监测",
            "event": "警告 - 检测到事件",
            "channels": "通道总数",
            "sampling": "采样频率",
            "max_amp": "最大幅值",
            "rms": "均方根值",
            "events_count": "事件总数",
            "no_events": "暂无检测事件",
            "time": "时间 (s)",
            "channel": "通道",
            "amplitude": "幅值",
        },
        "en": {
            "title": "DASMatrix Online Monitoring System",
            "waterfall": "Real-time Spatio-Temporal Array",
            "metrics": "Signal Metrics Tracking (Max/RMS)",
            "log": "Detection Events Log",
            "stats": "System Runtime Stats",
            "ch_detail": "Focus Channel Analysis (CH: {ch})",
            "status": "System Status",
            "idle": "IDLE / READY",
            "active": "ACTIVE MONITORING",
            "event": "WARNING - EVENT DETECTED",
            "channels": "Total Channels",
            "sampling": "Sampling Rate",
            "max_amp": "Max Amplitude",
            "rms": "RMS Value",
            "events_count": "Total Events",
            "no_events": "No events detected.",
            "time": "Time (s)",
            "channel": "Channel",
            "amplitude": "Amplitude",
        }
    }

    def __init__(
        self,
        n_channels: int,
        fs: float,
        buffer_duration: float = 10.0,
        lang: str = "cn",
        config: Optional[VisualizationConfig] = None,
        focus_channel: int = 0,
    ):
        """初始化高级看板

        Args:
            n_channels: 通道数
            fs: 采样率 (Hz)
            buffer_duration: 历史显示长度 (s)
            lang: 语言 ('cn' 或 'en')
            config: 可视化配置
            focus_channel: 初始选中的焦点通道
        """
        self.n_channels = n_channels
        self.fs = fs
        self.buffer_duration = buffer_duration
        self.lang = lang.lower() if lang.lower() in self.I18N else "en"
        self.config = config or VisualizationConfig.for_screen()
        self.focus_channel = focus_channel
        self.logger = logging.getLogger(__name__)

        # 配色方案 (Dark Industrial)
        self.colors = {
            "bg": "#0B0B0B",
            "panel": "#141414",
            "border": "#333333",
            "text": "#E0E0E0",
            "accent_1": "#00E5FF",  # Cyan (Trend Max)
            "accent_2": "#FF9100",  # Orange (Trend RMS)
            "led_ok": "#00C853",
            "led_warn": "#FFD600",
            "led_error": "#FF1744",
            "grid": "#222222"
        }

        # 数据缓冲区
        self.n_samples_buffer = int(buffer_duration * fs)
        self.data_buffer = np.zeros((self.n_samples_buffer, n_channels))
        
        # 指标历史
        self.metrics_history = {"max": [], "rms": [], "times": []}
        self.max_metrics_points = 100
        self.event_log = []
        self.start_timestamp = time.time()

        # 初始化图形 - 继承全局配置
        apply_nature_style(self.config)
        
        # 微调深色主题设置，确保不覆盖中文字体列表
        cjk_fonts = plt.rcParams["font.sans-serif"]
        plt.rcParams.update({
            "figure.facecolor": self.colors["bg"],
            "axes.facecolor": self.colors["panel"],
            "axes.edgecolor": self.colors["border"],
            "axes.labelcolor": self.colors["text"],
            "xtick.color": self.colors["text"],
            "ytick.color": self.colors["text"],
            "text.color": self.colors["text"],
            "grid.color": self.colors["grid"],
        })
        # 显式恢复中文字体列表
        plt.rcParams["font.sans-serif"] = cjk_fonts

        # 初始化布局
        self._setup_layout()

    def _t(self, key: str, **kwargs) -> str:
        """翻译辅助函数"""
        text = self.I18N[self.lang].get(key, key)
        if kwargs:
            return text.format(**kwargs)
        return text

    def _setup_layout(self):
        """核心 UI 布局构建 (极致均衡比例)"""
        # (rcParams 已在 __init__ 中设置)
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title(self._t("title"))
        
        # 使用 GridSpec 划分区域
        # 高度：标题 0.4，两个图表各 1.3，底部文字 0.8 (更紧凑)
        # 宽度：左侧 2.4 (约 60%)，侧边栏 1.6 (约 40%)
        gs = gridspec.GridSpec(4, 2, figure=self.fig, 
                               height_ratios=[0.4, 1.3, 1.3, 0.8],
                               width_ratios=[2.4, 1.6],
                               hspace=0.45, wspace=0.25)
        
        # --- 1. 顶部标题和 LED 状态栏 ---
        self.ax_header = self.fig.add_subplot(gs[0, :])
        self.ax_header.axis("off")
        self.header_text = self.ax_header.text(
            0.5, 0.5, self._t("title"),
            ha="center", va="center", fontsize=24, fontweight="bold", color=self.colors["accent_1"]
        )
        
        self.led = Circle((0.02, 0.5), 0.01, color=self.led_color("idle"), transform=self.ax_header.transAxes)
        self.ax_header.add_patch(self.led)
        self.status_label = self.ax_header.text(
            0.04, 0.5, self._t("status") + ": " + self._t("idle"),
            va="center", fontsize=11, fontweight="bold"
        )

        # --- 2. 瀑布图视图 ---
        self.ax_waterfall = self.fig.add_subplot(gs[1:4, 0])
        setup_axis(self.ax_waterfall, 
                   xlabel=self._t("time"), 
                   ylabel=self._t("channel"), 
                   title=self._t("waterfall"))
        
        self.im = self.ax_waterfall.imshow(
            self.data_buffer.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[-self.buffer_duration, 0, 0, self.n_channels],
            animated=True
        )
        # 极致压缩色标空间
        add_colorbar(self.fig, self.im, self.ax_waterfall, label=self._t("amplitude"), fraction=0.015, pad=0.02)

        # --- 3. 趋势指标面板 ---
        self.ax_metrics = self.fig.add_subplot(gs[1, 1])
        setup_axis(self.ax_metrics, xlabel=self._t("time"), title=self._t("metrics"))
        self.line_max, = self.ax_metrics.plot([], [], color=self.colors["accent_1"], lw=1.5, label="Max")
        self.line_rms, = self.ax_metrics.plot([], [], color=self.colors["accent_2"], lw=1.5, label="RMS")
        self.ax_metrics.legend(loc="upper left", fontsize=7, frameon=False)

        # --- 4. 焦点通道细节图 ---
        self.ax_detail = self.fig.add_subplot(gs[2, 1])
        setup_axis(self.ax_detail, title=self._t("ch_detail", ch=self.focus_channel))
        self.line_detail, = self.ax_detail.plot([], [], color=self.colors["accent_1"], lw=0.8)

        # --- 5. 事件日志与系统统计 ---
        self.ax_info = self.fig.add_subplot(gs[3, 1])
        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(
            0.0, 1.0, "System metadata initializing...",
            va="top", fontsize=9, linespacing=1.6
        )

    def led_color(self, state: str) -> str:
        """获取状态灯颜色"""
        return {
            "idle": self.colors["led_warn"],
            "active": self.colors["led_ok"],
            "event": self.colors["led_error"]
        }.get(state, self.colors["border"])

    def update(self, chunk: np.ndarray, events: Optional[np.ndarray] = None, metadata: Optional[Dict] = None):
        """高性能实时刷新"""
        n_samples = chunk.shape[0]
        
        # 数据平滑滚动
        if n_samples >= self.n_samples_buffer:
            self.data_buffer = chunk[-self.n_samples_buffer:]
        else:
            self.data_buffer = np.roll(self.data_buffer, -n_samples, axis=0)
            self.data_buffer[-n_samples:] = chunk

        # 1. 更新瀑布图
        self.im.set_data(self.data_buffer.T)
        vmax = np.percentile(np.abs(self.data_buffer), 99.5) or 1.0
        self.im.set_clim(-vmax, vmax)

        # 2. 计算并更新指标
        now = time.time() - self.start_timestamp
        max_v = np.max(np.abs(chunk))
        rms_v = np.sqrt(np.mean(chunk**2))
        
        self.metrics_history["times"].append(now)
        self.metrics_history["max"].append(max_v)
        self.metrics_history["rms"].append(rms_v)
        
        if len(self.metrics_history["times"]) > self.max_metrics_points:
            for k in self.metrics_history: self.metrics_history[k].pop(0)

        t_axis = self.metrics_history["times"]
        self.line_max.set_data(t_axis, self.metrics_history["max"])
        self.line_rms.set_data(t_axis, self.metrics_history["rms"])
        self.ax_metrics.set_xlim(min(t_axis), max(t_axis) + 0.5)
        self.ax_metrics.set_ylim(0, max(self.metrics_history["max"]) * 1.3 + 0.1)

        # 3. 更新焦点通道
        focus_data = chunk[:, self.focus_channel]
        t_detail = np.linspace(0, n_samples/self.fs, n_samples)
        self.line_detail.set_data(t_detail, focus_data)
        self.ax_detail.set_xlim(0, t_detail[-1])
        self.ax_detail.set_ylim(np.min(focus_data)*1.2, np.max(focus_data)*1.2 + 0.01)

        # 4. 更新状态与日志
        state = "active"
        n_events = 0
        if events is not None:
            n_events = np.sum(events > 0)
            if n_events > 0:
                state = "event"
                ts = time.strftime("%H:%M:%S")
                self.event_log.append(f"[{ts}] Detected {n_events} anomalies")
                if len(self.event_log) > 4: self.event_log.pop(0)

        # 更新 LED 和文字
        self.led.set_color(self.led_color(state))
        self.status_label.set_text(f"{self._t('status')}: {self._t(state)}")
        self.status_label.set_color(self.led_color(state))

        stats_lines = [
            f"{self._t('channels'):<15}: {self.n_channels}",
            f"{self._t('sampling'):<15}: {self.fs} Hz",
            f"{self._t('max_amp'):<15}: {max_v:.4f}",
            f"{self._t('rms'):<15}: {rms_v:.4f}",
            f"{self._t('events_count'):<15}: {len(self.event_log)}",
            "-" * 30,
            "\n".join(self.event_log[::-1]) if self.event_log else self._t("no_events")
        ]
        self.info_text.set_text("\n".join(stats_lines))

        # 强制绘制
        self.fig.canvas.draw_idle()
        if plt.get_backend().lower() not in ["agg", "pdf", "svg"]:
            plt.pause(0.001)

    def show(self):
        plt.ion()
        plt.show()

    def close(self):
        plt.close(self.fig)
