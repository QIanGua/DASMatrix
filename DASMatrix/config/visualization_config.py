"""Nature/Science 期刊级别可视化配置

专为达到顶级期刊出版质量设计的可视化配置系统。
遵循 Nature、Science、Cell 等期刊的图表制作指南。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class JournalStyle(Enum):
    """期刊风格枚举"""

    NATURE = "nature"
    SCIENCE = "science"
    CELL = "cell"
    PNAS = "pnas"
    DEFAULT = "default"


class FigureSize(Enum):
    """Nature 期刊标准图表尺寸（英寸）

    基于 Nature 投稿指南：单栏 89mm，1.5栏 120mm，双栏 183mm
    """

    SINGLE_COLUMN = (3.5, 2.625)  # 89mm 宽，黄金比例高度
    ONE_HALF_COLUMN = (4.72, 3.54)  # 120mm 宽
    DOUBLE_COLUMN = (7.2, 4.32)  # 183mm 宽
    FULL_PAGE = (7.2, 9.6)  # 全页
    SQUARE = (3.5, 3.5)  # 正方形
    WIDE = (7.2, 3.0)  # 宽幅（时间序列）


@dataclass
class ColorPalette:
    """色盲友好的配色方案

    基于 Nature 色盲友好配色指南和 ColorBrewer。
    """

    # 主色板 - 色盲友好（来自 Wong, B. 2011 Nature Methods）
    primary: List[str] = field(
        default_factory=lambda: [
            "#0072B2",  # 蓝色
            "#D55E00",  # 朱红色
            "#009E73",  # 蓝绿色
            "#CC79A7",  # 紫红色
            "#E69F00",  # 橙色
            "#56B4E9",  # 天蓝色
            "#F0E442",  # 黄色
            "#000000",  # 黑色
        ]
    )

    # 顺序色板（连续数据）
    sequential: str = "viridis"  # Matplotlib 默认，色盲友好

    # 发散色板（有中心的双向数据）
    diverging: str = "RdBu_r"  # 红-蓝，经典科学配色

    # 时频图配色
    spectrogram: str = "inferno"  # 高对比度，打印友好

    # 热图配色
    heatmap: str = "magma"  # 均匀感知，打印友好

    # 强调色
    highlight: str = "#D55E00"  # 朱红色用于强调
    annotation: str = "#0072B2"  # 蓝色用于注释

    # 网格和边框
    grid: str = "#CCCCCC"
    spine: str = "#333333"

    def get_color(self, index: int) -> str:
        """获取指定索引的颜色"""
        return self.primary[index % len(self.primary)]


@dataclass
class Typography:
    """排版设置 - 遵循期刊标准

    Nature 要求：Arial/Helvetica, 5-7pt 最小字号
    Science 要求：Helvetica/Arial, 6-8pt 最小字号
    """

    # 字体族
    family: str = "Arial"  # Nature/Science 推荐
    math_family: str = "stix"  # 数学字体
    # 中文字体（macOS: STHeiti, Windows: Microsoft YaHei, Linux: Noto Serif SC）
    cjk_family: str = "STHeiti"

    # 字号设置（单位：pt）
    title: int = 10  # 标题
    label: int = 8  # 轴标签
    tick: int = 7  # 刻度标签
    legend: int = 7  # 图例
    annotation: int = 7  # 注释

    # 字重
    title_weight: str = "bold"
    label_weight: str = "normal"


@dataclass
class VisualizationConfig:
    """Nature/Science 级别可视化配置类

    Examples:
        >>> config = VisualizationConfig()
        >>> config = VisualizationConfig(style=JournalStyle.NATURE)
        >>> config = VisualizationConfig.for_print()  # 打印优化
        >>> config = VisualizationConfig.for_screen() # 屏幕优化
    """

    # 基础设置
    style: JournalStyle = JournalStyle.NATURE
    dpi: int = 300  # 期刊最低要求 300 DPI

    # 图形尺寸
    figsize: Tuple[float, float] = FigureSize.SINGLE_COLUMN.value
    figsize_standard: Tuple[float, float] = FigureSize.SINGLE_COLUMN.value
    figsize_wide: Tuple[float, float] = FigureSize.WIDE.value

    # 配色
    colors: ColorPalette = field(default_factory=ColorPalette)

    # 排版
    typography: Typography = field(default_factory=Typography)

    # 网格
    grid: bool = False  # Nature 通常不使用网格
    grid_style: str = "-"
    grid_alpha: float = 0.3
    grid_linewidth: float = 0.5

    # 边框和刻度
    spine_linewidth: float = 0.8
    tick_direction: str = "in"  # 内向刻度（Nature 标准）
    tick_major_size: float = 3.0
    tick_minor_size: float = 1.5
    tick_major_width: float = 0.8
    tick_minor_width: float = 0.5
    show_minor_ticks: bool = False  # 默认不显示次刻度

    # 线条
    line_width: float = 1.0  # 主数据线宽
    line_width_thin: float = 0.5  # 辅助线宽
    marker_size: float = 4.0

    # 图例
    legend_frameon: bool = False  # Nature 风格通常无边框
    legend_loc: str = "best"

    # 颜色条
    colorbar_width: float = 0.02
    colorbar_pad: float = 0.02

    # 间距
    label_pad: float = 4.0
    title_pad: float = 8.0

    # 兼容性属性
    font_size: int = 8
    label_size: int = 8
    tick_size: int = 7
    legend_size: int = 7
    plot_title_font_size: int = 10

    @classmethod
    def for_print(cls) -> "VisualizationConfig":
        """创建适合打印的配置（高对比度）"""
        config = cls()
        config.dpi = 600
        config.line_width = 1.2
        config.spine_linewidth = 1.0
        return config

    @classmethod
    def for_screen(cls) -> "VisualizationConfig":
        """创建适合屏幕显示的配置"""
        config = cls()
        config.dpi = 150
        config.line_width = 1.5
        config.typography.title = 12
        config.typography.label = 10
        config.typography.tick = 9
        return config

    @classmethod
    def for_presentation(cls) -> "VisualizationConfig":
        """创建适合演示文稿的配置"""
        config = cls()
        config.dpi = 150
        config.figsize = FigureSize.DOUBLE_COLUMN.value
        config.line_width = 2.0
        config.marker_size = 6.0
        config.typography.title = 16
        config.typography.label = 14
        config.typography.tick = 12
        config.typography.legend = 12
        return config

    def get_rcparams(self) -> Dict:
        """生成 matplotlib rcParams 字典"""
        return {
            # 图形设置
            "figure.dpi": self.dpi,
            "figure.figsize": self.figsize,
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "figure.autolayout": False,
            "figure.constrained_layout.use": True,
            # 坐标轴
            "axes.linewidth": self.spine_linewidth,
            "axes.edgecolor": self.colors.spine,
            "axes.facecolor": "white",
            "axes.labelcolor": "black",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.grid": self.grid,
            "axes.axisbelow": True,
            "axes.prop_cycle": f"cycler('color', {self.colors.primary})",
            # 网格
            "grid.color": self.colors.grid,
            "grid.linestyle": self.grid_style,
            "grid.alpha": self.grid_alpha,
            "grid.linewidth": self.grid_linewidth,
            # 刻度
            "xtick.direction": self.tick_direction,
            "ytick.direction": self.tick_direction,
            "xtick.major.size": self.tick_major_size,
            "ytick.major.size": self.tick_major_size,
            "xtick.minor.size": self.tick_minor_size,
            "ytick.minor.size": self.tick_minor_size,
            "xtick.major.width": self.tick_major_width,
            "ytick.major.width": self.tick_major_width,
            "xtick.minor.width": self.tick_minor_width,
            "ytick.minor.width": self.tick_minor_width,
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": self.show_minor_ticks,
            "ytick.minor.visible": self.show_minor_ticks,
            # 字体 - 中文字体优先，确保中文正确显示
            "font.family": "sans-serif",
            "font.sans-serif": [
                self.typography.cjk_family,  # 中文字体优先（默认 STHeiti）
                "Songti SC",  # macOS 宋体
                "Heiti TC",  # macOS 黑体繁体
                "Noto Serif SC",  # Linux 中文字体
                "Microsoft YaHei",  # Windows 中文字体
                self.typography.family,
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": self.typography.label,
            "axes.labelsize": self.typography.label,
            "axes.titlesize": self.typography.title,
            "axes.titleweight": self.typography.title_weight,
            "axes.labelweight": self.typography.label_weight,
            "xtick.labelsize": self.typography.tick,
            "ytick.labelsize": self.typography.tick,
            "legend.fontsize": self.typography.legend,
            "axes.unicode_minus": False,
            # 线条
            "lines.linewidth": self.line_width,
            "lines.markersize": self.marker_size,
            "lines.markeredgewidth": 0.5,
            "lines.solid_capstyle": "round",
            # 图例
            "legend.frameon": self.legend_frameon,
            "legend.framealpha": 0.8,
            "legend.fancybox": False,
            "legend.borderaxespad": 0.5,
            # 保存
            "savefig.dpi": self.dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.transparent": False,
            "savefig.format": "pdf",  # Nature 要求 PDF 或 EPS
            # 数学文本
            "mathtext.fontset": self.typography.math_family,
        }
