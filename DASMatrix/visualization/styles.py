"""Nature/Science 期刊级别样式设置

提供专业的 matplotlib 样式上下文管理器和工具函数。
"""

import contextlib
import logging
import warnings
from typing import Any, Optional, Tuple, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from ..config.visualization_config import (
    ColorPalette,
    FigureSize,
    VisualizationConfig,
)


def apply_nature_style(config: Optional[VisualizationConfig] = None) -> None:
    """应用 Nature/Science 期刊风格到全局 matplotlib 设置

    Args:
        config: 可视化配置，默认使用 Nature 风格

    Examples:
        >>> from DASMatrix.visualization.styles import apply_nature_style
        >>> apply_nature_style()
        >>> # 之后所有图表都将使用 Nature 风格
    """
    if config is None:
        config = VisualizationConfig()

    # 抑制因字体回退导致的警告（例如 Arial 不含中文，但我们会回落到其他字体）
    warnings.filterwarnings("ignore", message=".*Glyph.*missing from current font.*")
    warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*")

    # 抑制 Linux 下缺少 Arial 的日志噪音
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    # 重置为默认样式
    plt.style.use("default")

    # 应用配置
    rcparams = config.get_rcparams()

    # 处理 prop_cycle（需要特殊处理）
    rcparams.pop("axes.prop_cycle", None)
    plt.rcParams.update(rcparams)

    # 设置颜色循环
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", config.colors.primary)


@contextlib.contextmanager
def nature_style(config: Optional[VisualizationConfig] = None):
    """Nature 风格上下文管理器

    Args:
        config: 可视化配置

    Examples:
        >>> with nature_style():
        ...     fig, ax = plt.subplots()
        ...     ax.plot(x, y)
    """
    if config is None:
        config = VisualizationConfig()

    # 保存当前样式
    old_rcparams = plt.rcParams.copy()

    try:
        apply_nature_style(config)
        yield
    finally:
        # 恢复原样式
        plt.rcParams.update(old_rcparams)


def create_figure(
    size: Union[FigureSize, Tuple[float, float], str] = FigureSize.SINGLE_COLUMN,
    nrows: int = 1,
    ncols: int = 1,
    config: Optional[VisualizationConfig] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, np.ndarray]:
    """创建期刊级别的图形

    Args:
        size: 图形尺寸，可以是 FigureSize 枚举、元组或字符串
        nrows: 子图行数
        ncols: 子图列数
        config: 可视化配置
        **kwargs: 传递给 plt.subplots 的其他参数

    Returns:
        tuple: (Figure, Axes)

    Examples:
        >>> fig, ax = create_figure()
        >>> fig, axes = create_figure(size="double", nrows=2, ncols=2)
    """
    if config is None:
        config = VisualizationConfig()

    apply_nature_style(config)

    # 解析尺寸
    if isinstance(size, FigureSize):
        figsize = size.value
    elif isinstance(size, str):
        size_map = {
            "single": FigureSize.SINGLE_COLUMN.value,
            "1.5": FigureSize.ONE_HALF_COLUMN.value,
            "double": FigureSize.DOUBLE_COLUMN.value,
            "full": FigureSize.FULL_PAGE.value,
            "square": FigureSize.SQUARE.value,
            "wide": FigureSize.WIDE.value,
        }
        figsize = size_map.get(size.lower(), FigureSize.SINGLE_COLUMN.value)
    else:
        figsize = size

    # 调整多子图的高度
    if nrows > 1:
        figsize = (figsize[0], figsize[1] * nrows * 0.8)
    if ncols > 1:
        figsize = (figsize[0] * ncols * 0.6, figsize[1])

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True, **kwargs)

    # 确保 axes 始终是数组
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])

    return fig, axes


def setup_axis(
    ax: plt.Axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    config: Optional[VisualizationConfig] = None,
) -> plt.Axes:
    """配置坐标轴为期刊级别样式

    Args:
        ax: matplotlib Axes 对象
        xlabel: X轴标签
        ylabel: Y轴标签
        title: 标题
        xlim: X轴范围
        ylim: Y轴范围
        config: 可视化配置

    Returns:
        配置后的 Axes 对象
    """
    if config is None:
        config = VisualizationConfig()

    # 确保所有边框可见且样式一致
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(config.spine_linewidth)
        ax.spines[spine].set_color(config.colors.spine)

    # 设置刻度方向和样式
    ax.tick_params(
        which="both",
        direction=config.tick_direction,
        top=True,
        right=True,
        width=config.tick_major_width,
        length=config.tick_major_size,
    )

    if config.show_minor_ticks:
        ax.minorticks_on()
        ax.tick_params(
            which="minor",
            direction=config.tick_direction,
            top=True,
            right=True,
            width=config.tick_minor_width,
            length=config.tick_minor_size,
        )

    # 设置标签
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=config.typography.label)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=config.typography.label)
    if title:
        ax.set_title(title, fontsize=config.typography.title, fontweight="bold")

    # 设置范围
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    return ax


def add_colorbar(
    fig: plt.Figure,
    mappable: plt.cm.ScalarMappable,
    ax: plt.Axes,
    label: Optional[str] = None,
    config: Optional[VisualizationConfig] = None,
    **kwargs: Any,
) -> Any:
    """添加专业格式的颜色条

    Args:
        fig: Figure 对象
        mappable: 可映射对象（如 imshow 返回值）
        ax: 关联的 Axes
        label: 颜色条标签
        config: 可视化配置
        **kwargs: 传递给 colorbar 的其他参数

    Returns:
        matplotlib.colorbar.Colorbar: Colorbar 对象
    """
    if config is None:
        config = VisualizationConfig()

    # 安全地设置默认值，允许 kwargs 覆盖
    kwargs.setdefault("pad", config.colorbar_pad)
    kwargs.setdefault("fraction", config.colorbar_width * 2)

    cbar = fig.colorbar(
        mappable,
        ax=ax,
        **kwargs,
    )

    if label:
        cbar.set_label(label, fontsize=config.typography.label)

    cbar.ax.tick_params(
        direction=config.tick_direction,
        width=config.tick_major_width,
        length=config.tick_major_size,
    )

    return cbar


def save_figure(
    fig: plt.Figure,
    filename: str,
    formats: Tuple[str, ...] = ("pdf", "png"),
    config: Optional[VisualizationConfig] = None,
    **kwargs: Any,
) -> None:
    """保存图形为多种格式

    Nature 要求 PDF/EPS 矢量格式，同时生成 PNG 用于预览。

    Args:
        fig: Figure 对象
        filename: 文件名（不含扩展名）
        formats: 输出格式列表
        config: 可视化配置
        **kwargs: 传递给 savefig 的其他参数
    """
    from pathlib import Path

    if config is None:
        config = VisualizationConfig()

    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        dpi = config.dpi if fmt != "pdf" else 300  # PDF 使用矢量，DPI 主要影响栅格元素
        fig.savefig(
            filepath.with_suffix(f".{fmt}"),
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
            edgecolor="none",
            **kwargs,
        )


# ============================================================================
# 专业配色工具
# ============================================================================


def get_colors(n: int, palette: Optional[ColorPalette] = None) -> list:
    """获取指定数量的颜色

    Args:
        n: 需要的颜色数量
        palette: 配色方案

    Returns:
        颜色列表
    """
    if palette is None:
        palette = ColorPalette()

    if n <= len(palette.primary):
        return palette.primary[:n]
    else:
        # 需要更多颜色，使用 colormap 插值
        cmap = plt.colormaps.get_cmap("tab20")
        return [cmap(i / n) for i in range(n)]


def create_sequential_cmap(name: str = "viridis") -> colors.Colormap:
    """创建顺序色图"""
    return plt.cm.get_cmap(name)


def create_diverging_cmap(name: str = "RdBu_r", center: float = 0) -> colors.Colormap:
    """创建发散色图"""
    return plt.cm.get_cmap(name)


# ============================================================================
# 注释和标记工具
# ============================================================================


def add_panel_label(
    ax: plt.Axes,
    label: str,
    loc: str = "upper left",
    fontsize: int = 12,
    fontweight: str = "bold",
) -> None:
    """添加子图标签（如 a, b, c）

    Args:
        ax: Axes 对象
        label: 标签文本（如 "a", "b"）
        loc: 位置
        fontsize: 字号
        fontweight: 字重
    """
    loc_map = {
        "upper left": (-0.1, 1.1),
        "upper right": (1.05, 1.1),
        "lower left": (-0.1, -0.1),
        "lower right": (1.05, -0.1),
    }

    x, y = loc_map.get(loc, (-0.1, 1.1))

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        va="top",
        ha="left" if "left" in loc else "right",
    )


def add_scalebar(
    ax: plt.Axes,
    length: float,
    label: str,
    loc: str = "lower right",
    color: str = "black",
    height_fraction: float = 0.02,
) -> None:
    """添加比例尺

    Args:
        ax: Axes 对象
        length: 比例尺长度（数据坐标）
        label: 标签文本
        loc: 位置
        color: 颜色
        height_fraction: 高度占图高的比例
    """
    import matplotlib.font_manager as fm
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    fontprops = fm.FontProperties(size=7)
    scalebar = AnchoredSizeBar(
        ax.transData,
        length,
        label,
        loc=loc,
        pad=0.5,
        color=color,
        frameon=False,
        size_vertical=ax.get_ylim()[1] * height_fraction,
        fontproperties=fontprops,
    )
    ax.add_artist(scalebar)


def add_significance_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    text: str = "*",
    color: str = "black",
    linewidth: float = 1.0,
) -> None:
    """添加显著性标记括号

    Args:
        ax: Axes 对象
        x1: 括号起始 X 坐标
        x2: 括号终止 X 坐标
        y: 括号 Y 坐标
        text: 显著性标记（*, **, ***）
        color: 颜色
        linewidth: 线宽
    """
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02

    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        color=color,
        linewidth=linewidth,
        clip_on=False,
    )
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=10, color=color)
