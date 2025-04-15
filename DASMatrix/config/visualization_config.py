from dataclasses import dataclass
from typing import Tuple


@dataclass
class VisualizationConfig:
    """可视化配置类"""

    dpi: int = 300
    figsize_standard: Tuple[int, int] = (10, 6)
    figsize_wide: Tuple[int, int] = (12, 4)
    grid: bool = True
    grid_style: str = "--"
    grid_alpha: float = 0.5
    line_width: float = 1.5
    font_size: int = 10
    label_size: int = 10
    tick_size: int = 9
    legend_size: int = 9
    plot_title_font_size: int = 12
