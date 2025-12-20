"""可视化模块

提供 Nature/Science 期刊级别的可视化工具。
"""

from .das_visualizer import (
    DASVisualizer,
    PlotBase,
    SpectrogramPlot,
    SpectrumPlot,
    WaterfallPlot,
    WaveformPlot,
)
from .dashboard import DASDashboard
from .styles import (
    add_colorbar,
    add_panel_label,
    add_scalebar,
    add_significance_bracket,
    apply_nature_style,
    create_figure,
    get_colors,
    nature_style,
    save_figure,
    setup_axis,
)
from .web.web_dashboard import DASWebDashboard

__all__ = [
    # 可视化器
    "DASVisualizer",
    "PlotBase",
    "SpectrumPlot",
    "SpectrogramPlot",
    "WaveformPlot",
    "WaterfallPlot",
    "DASDashboard",
    "DASWebDashboard",
    # 样式工具
    "apply_nature_style",
    "create_figure",
    "nature_style",
    "save_figure",
    "setup_axis",
    "add_colorbar",
    "add_panel_label",
    "add_scalebar",
    "add_significance_bracket",
    "get_colors",
]
