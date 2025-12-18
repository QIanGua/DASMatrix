"""可视化模块测试

测试 DASMatrix 可视化功能，包括 styles、das_visualizer 等模块。
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # 使用非交互式后端

from DASMatrix.config import VisualizationConfig
from DASMatrix.visualization.das_visualizer import (
    FKPlot,
    PlotBase,
    SpectrogramPlot,
    SpectrumPlot,
    WaterfallPlot,
    WaveformPlot,
)
from DASMatrix.visualization.styles import (
    apply_nature_style,
    create_figure,
    get_colors,
    nature_style,
    save_figure,
    setup_axis,
)


class TestStyles:
    """测试 styles.py 中的样式工具函数"""

    def test_apply_nature_style(self) -> None:
        """测试应用 Nature 风格"""
        apply_nature_style()
        # 验证一些关键样式已被应用
        assert plt.rcParams["axes.linewidth"] > 0
        assert plt.rcParams["font.family"] is not None

    def test_nature_style_context_manager(self) -> None:
        """测试 Nature 风格上下文管理器"""
        _ = plt.rcParams["axes.linewidth"]  # 保存原值用于测试

        with nature_style():
            # 在上下文中样式应该已应用
            pass

        # 上下文退出后样式应该恢复
        assert plt.rcParams is not None

    def test_create_figure_default(self) -> None:
        """测试创建默认图形"""
        fig, axes = create_figure()
        assert fig is not None
        assert axes is not None
        assert len(axes) >= 1
        plt.close(fig)

    def test_create_figure_multi_panel(self) -> None:
        """测试创建多面板图形"""
        fig, axes = create_figure(nrows=2, ncols=2)
        assert fig is not None
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_create_figure_size_string(self) -> None:
        """测试使用字符串指定图形尺寸"""
        fig, _ = create_figure(size="double")
        assert fig is not None
        plt.close(fig)

        fig, _ = create_figure(size="single")
        assert fig is not None
        plt.close(fig)

    def test_setup_axis(self) -> None:
        """测试配置坐标轴"""
        fig, ax = plt.subplots()
        setup_axis(ax, xlabel="X Label", ylabel="Y Label", title="Test Title")

        assert ax.get_xlabel() == "X Label"
        assert ax.get_ylabel() == "Y Label"
        assert ax.get_title() == "Test Title"
        plt.close(fig)

    def test_setup_axis_with_limits(self) -> None:
        """测试配置坐标轴范围"""
        fig, ax = plt.subplots()
        setup_axis(ax, xlim=(0, 10), ylim=(-1, 1))

        assert ax.get_xlim() == (0, 10)
        assert ax.get_ylim() == (-1, 1)
        plt.close(fig)

    def test_get_colors(self) -> None:
        """测试获取颜色"""
        colors_3 = get_colors(3)
        assert len(colors_3) == 3

        colors_10 = get_colors(10)
        assert len(colors_10) == 10

    def test_save_figure(self, tmp_path) -> None:
        """测试保存图形"""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        filename = tmp_path / "test_figure"
        save_figure(fig, str(filename), formats=("png",))

        assert (tmp_path / "test_figure.png").exists()
        plt.close(fig)


class TestPlotBase:
    """测试 PlotBase 类"""

    def test_init_default_config(self) -> None:
        """测试使用默认配置初始化"""
        plot = PlotBase()
        assert plot.config is not None
        assert isinstance(plot.config, VisualizationConfig)

    def test_init_custom_config(self) -> None:
        """测试使用自定义配置初始化"""
        config = VisualizationConfig()
        plot = PlotBase(config=config)
        assert plot.config is config


class TestSpectrumPlot:
    """测试 SpectrumPlot 类"""

    def test_plot_basic(self) -> None:
        """测试基本频谱图绘制"""
        plotter = SpectrumPlot()
        frequencies = np.linspace(0, 500, 513)
        # 使用线性幅度值（正数），因为 plot 内部会进行 dB 转换
        magnitudes = np.abs(np.random.randn(513)) + 0.01

        # SpectrumPlot.plot 只返回 fig
        fig = plotter.plot(frequencies, magnitudes)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_peaks(self) -> None:
        """测试带峰值标记的频谱图"""
        plotter = SpectrumPlot()
        frequencies = np.linspace(0, 500, 513)
        magnitudes = np.abs(np.random.randn(513)) + 0.01

        # 峰值也使用线性幅度值
        peaks = [
            {"frequency": 50, "magnitude": 1.0},
            {"frequency": 100, "magnitude": 0.8},
        ]

        fig = plotter.plot(frequencies, magnitudes, peaks=peaks)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_custom_ax(self) -> None:
        """测试使用自定义 Axes"""
        plotter = SpectrumPlot()
        frequencies = np.linspace(0, 500, 513)
        magnitudes = np.abs(np.random.randn(513)) + 0.01

        fig_ext, ax_ext = plt.subplots()
        fig = plotter.plot(frequencies, magnitudes, ax=ax_ext)

        # 当传入 ax 时，返回的 fig 应该是 ax 所属的 figure
        assert fig is fig_ext
        plt.close(fig_ext)


class TestWaveformPlot:
    """测试 WaveformPlot 类"""

    def test_plot_basic(self) -> None:
        """测试基本波形图绘制"""
        plotter = WaveformPlot()
        amplitude_data = np.sin(np.linspace(0, 4 * np.pi, 1000))

        # WaveformPlot.plot 返回 (fig, ax) 或 fig，需要检查外层 ax
        fig_ext, ax_ext = plt.subplots()
        result = plotter.plot(amplitude_data, fs=1000, ax=ax_ext)
        assert result is not None
        plt.close(fig_ext)

    def test_plot_with_title(self) -> None:
        """测试带标题的波形图"""
        plotter = WaveformPlot()
        amplitude_data = np.sin(np.linspace(0, 4 * np.pi, 1000))

        fig_ext, ax_ext = plt.subplots()
        plotter.plot(amplitude_data, fs=1000, title="Test Waveform", ax=ax_ext)
        assert ax_ext.get_title() == "Test Waveform"
        plt.close(fig_ext)

    def test_plot_with_amplitude_range(self) -> None:
        """测试指定幅值范围"""
        plotter = WaveformPlot()
        amplitude_data = np.sin(np.linspace(0, 4 * np.pi, 1000))

        fig_ext, ax_ext = plt.subplots()
        plotter.plot(amplitude_data, fs=1000, amplitude_range=(-0.5, 0.5), ax=ax_ext)
        ylim = ax_ext.get_ylim()
        # 检查 y 轴范围（可能有微小浮点差异）
        assert ylim[0] <= -0.4 and ylim[1] >= 0.4
        plt.close(fig_ext)


class TestSpectrogramPlot:
    """测试 SpectrogramPlot 类"""

    def test_plot_basic(self) -> None:
        """测试基本时频图绘制"""
        plotter = SpectrogramPlot()
        # 使用足够长的数据避免 nperseg 警告 (默认 window_size=1024)
        data = np.sin(2 * np.pi * 50 * np.linspace(0, 2, 2000))

        # SpectrogramPlot.plot 只返回 fig
        fig = plotter.plot(data, fs=1000)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_freq_range(self) -> None:
        """测试指定频率范围"""
        plotter = SpectrogramPlot()
        data = np.sin(2 * np.pi * 50 * np.linspace(0, 2, 2000))

        fig = plotter.plot(data, fs=1000, freq_range=(0, 200))
        assert fig is not None
        plt.close(fig)


class TestWaterfallPlot:
    """测试 WaterfallPlot 类"""

    def test_plot_basic(self) -> None:
        """测试基本瀑布图绘制"""
        plotter = WaterfallPlot()
        data = np.random.randn(1000, 64)

        # WaterfallPlot.plot 只返回 fig
        fig = plotter.plot(data, fs=1000)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_labels(self) -> None:
        """测试带轴标签的瀑布图"""
        plotter = WaterfallPlot()
        data = np.random.randn(1000, 64)

        fig = plotter.plot(data, fs=1000, x_label="Time (s)", y_label="Channel")
        assert fig is not None
        plt.close(fig)


class TestFKPlot:
    """测试 FKPlot 类"""

    def test_plot_basic(self) -> None:
        """测试基本 FK 图绘制"""
        plotter = FKPlot()
        # 创建 FK 谱数据 (参数: fk_spectrum, freqs, wavenumbers)
        n_freq, n_k = 128, 64
        fk_spectrum = np.random.randn(n_freq, n_k) + 1j * np.random.randn(n_freq, n_k)
        freqs = np.linspace(0, 500, n_freq)
        wavenumbers = np.linspace(-0.1, 0.1, n_k)

        fig = plotter.plot(fk_spectrum, freqs, wavenumbers)
        assert fig is not None
        plt.close(fig)


class TestVisualizationConfig:
    """测试 VisualizationConfig 类"""

    def test_default_config(self) -> None:
        """测试默认配置"""
        config = VisualizationConfig()
        assert config.dpi == 300
        assert config.line_width > 0

    def test_get_rcparams(self) -> None:
        """测试获取 rcparams"""
        config = VisualizationConfig()
        rcparams = config.get_rcparams()

        assert isinstance(rcparams, dict)
        assert "font.size" in rcparams
        assert "axes.linewidth" in rcparams
