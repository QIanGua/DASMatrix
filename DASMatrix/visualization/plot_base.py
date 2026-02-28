"""Shared plotting base class for visualization modules."""

import logging
from abc import ABC
from pathlib import Path
from typing import Optional, Union

from ..config.visualization_config import VisualizationConfig


class PlotBase(ABC):
    """绘图基类。"""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
        self._setup_style()

    def _setup_style(self) -> None:
        import logging as _logging

        import matplotlib.pyplot as plt
        import seaborn as sns

        _logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)
        font_family = "sans-serif"
        plt.style.use("default")

        colors = [
            "#0072B2",
            "#D55E00",
            "#009E73",
            "#CC79A7",
            "#E69F00",
            "#56B4E9",
            "#F0E442",
            "#000000",
            "#0072B2",
            "#D55E00",
        ]
        sns.set_palette(colors)

        plt.rcParams.update(
            {
                "figure.dpi": self.config.dpi,
                "figure.figsize": self.config.figsize_standard,
                "figure.facecolor": "white",
                "figure.autolayout": True,
                "axes.grid": self.config.grid,
                "axes.linewidth": 1.0,
                "axes.edgecolor": "black",
                "axes.spines.top": True,
                "axes.spines.right": True,
                "axes.spines.bottom": True,
                "axes.spines.left": True,
                "axes.axisbelow": True,
                "axes.facecolor": "white",
                "axes.labelcolor": "black",
                "axes.prop_cycle": plt.cycler("color", colors),
                "grid.color": "#E0E0E0",
                "grid.linestyle": self.config.grid_style,
                "grid.alpha": 0.7,
                "grid.linewidth": 0.5,
                "lines.linewidth": 1.5,
                "lines.markeredgewidth": 1.0,
                "lines.markersize": 4,
                "lines.solid_capstyle": "round",
                "font.family": font_family,
                "font.sans-serif": self.config.get_rcparams()["font.sans-serif"],
                "font.size": 11,
                "font.weight": "normal",
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "axes.titleweight": "bold",
                "axes.labelweight": "normal",
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "axes.unicode_minus": False,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.major.width": 1.0,
                "ytick.major.width": 1.0,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5,
                "xtick.major.size": 4.0,
                "ytick.major.size": 4.0,
                "xtick.minor.size": 2.0,
                "ytick.minor.size": 2.0,
                "xtick.top": True,
                "ytick.right": True,
                "legend.frameon": True,
                "legend.framealpha": 0.8,
                "legend.fancybox": False,
                "legend.facecolor": "white",
                "legend.edgecolor": "black",
                "savefig.dpi": self.config.dpi,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.05,
                "savefig.transparent": False,
                "axes.autolimit_mode": "data",
                "axes.xmargin": 0.05,
                "axes.ymargin": 0.05,
            }
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(1.0)
            ax.spines[spine].set_color("black")
        ax.tick_params(top=True, right=True, which="both", direction="in")
        plt.close(fig)

    def save_figure(self, file_path: Union[str, Path], close_after_save: bool = True) -> None:
        import matplotlib.pyplot as plt

        try:
            if isinstance(file_path, str):
                file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(file_path, bbox_inches="tight", dpi=self.config.dpi)
            self.logger.debug("Figure saved to %s", file_path)
            if close_after_save:
                plt.close()
        except Exception as e:
            self.logger.error("Error saving figure: %s", e)
            if close_after_save:
                plt.close()

    def _add_watermark(self, fig, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        return None
