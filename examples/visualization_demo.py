"""Nature/Science 级别可视化示例

演示 DASMatrix 的出版级别可视化功能。
"""

import numpy as np
import matplotlib.pyplot as plt

# 导入 DASMatrix 模块
from DASMatrix.api import df
from DASMatrix.api.dasframe import DASFrame
from DASMatrix.visualization import (
    apply_nature_style,
    create_figure,
    setup_axis,
    add_colorbar,
    add_panel_label,
    save_figure,
)
from DASMatrix.config import VisualizationConfig, FigureSize


def create_synthetic_das_data(
    duration: float = 2.0,
    fs: float = 1000,
    n_channels: int = 64,
    signal_freq: float = 50,
    noise_level: float = 0.3,
) -> np.ndarray:
    """创建合成 DAS 数据用于演示
    
    Args:
        duration: 信号持续时间（秒）
        fs: 采样频率
        n_channels: 通道数
        signal_freq: 信号频率
        noise_level: 噪声水平
    
    Returns:
        合成 DAS 数据数组 (samples, channels)
    """
    t = np.linspace(0, duration, int(duration * fs))
    n_samples = len(t)
    
    # 创建基础信号
    data = np.zeros((n_samples, n_channels))
    
    for ch in range(n_channels):
        # 添加主信号（带相位延迟模拟波传播）
        phase_delay = ch * 0.02
        signal = np.sin(2 * np.pi * signal_freq * (t - phase_delay))
        
        # 添加事件（模拟地震或扰动）
        if 0.3 < (ch / n_channels) < 0.7:
            event_center = 0.5 + ch * 0.01
            event = 2.0 * np.exp(-((t - event_center) ** 2) / 0.01)
            signal += event
        
        # 添加噪声
        noise = noise_level * np.random.randn(n_samples)
        data[:, ch] = signal + noise
    
    return data


def demo_time_series():
    """演示时间序列图"""
    print("生成时间序列图...")
    
    # 创建数据
    data = create_synthetic_das_data()
    frame = DASFrame(data, fs=1000)
    
    # 使用新的可视化方法
    fig = frame.plot_ts(title="DAS Time Series")
    
    # 保存
    save_figure(fig, "output/demo_time_series", formats=("png",))
    plt.close(fig)
    print("  -> 保存到 output/demo_time_series.png")


def demo_heatmap():
    """演示热图"""
    print("生成热图...")
    
    data = create_synthetic_das_data()
    frame = DASFrame(data, fs=1000)
    
    fig = frame.plot_heatmap(title="DAS Waterfall")
    
    save_figure(fig, "output/demo_heatmap", formats=("png",))
    plt.close(fig)
    print("  -> 保存到 output/demo_heatmap.png")


def demo_spectrogram():
    """演示频谱图"""
    print("生成频谱图...")
    
    data = create_synthetic_das_data()
    frame = DASFrame(data, fs=1000)
    
    fig = frame.plot_spec(title="DAS Spectrogram")
    
    save_figure(fig, "output/demo_spectrogram", formats=("png",))
    plt.close(fig)
    print("  -> 保存到 output/demo_spectrogram.png")


def demo_multi_panel():
    """演示多面板组合图 - Nature 风格"""
    print("生成多面板组合图...")
    
    apply_nature_style()
    config = VisualizationConfig()
    
    # 创建数据
    data = create_synthetic_das_data()
    t = np.linspace(0, 2.0, data.shape[0])
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    
    # a) 时间序列
    ax = axes[0, 0]
    for i in range(3):
        ax.plot(t, data[:, i * 20], color=config.colors.primary[i], 
                linewidth=config.line_width, label=f"Ch {i*20}")
    setup_axis(ax, xlabel="Time (s)", ylabel="Amplitude", config=config)
    ax.legend(frameon=False, fontsize=7)
    add_panel_label(ax, "a")
    
    # b) 热图
    ax = axes[0, 1]
    im = ax.imshow(
        data.T, aspect="auto", origin="lower",
        cmap=config.colors.diverging,
        extent=[0, 2, 0, data.shape[1]],
        vmin=-3, vmax=3
    )
    add_colorbar(fig, im, ax, label="Amplitude", config=config)
    setup_axis(ax, xlabel="Time (s)", ylabel="Channel", config=config)
    add_panel_label(ax, "b")
    
    # c) 单通道频谱
    ax = axes[1, 0]
    from scipy import signal
    f, Pxx = signal.welch(data[:, 32], fs=1000, nperseg=256)
    ax.semilogy(f, Pxx, color=config.colors.primary[0], linewidth=config.line_width)
    setup_axis(ax, xlabel="Frequency (Hz)", ylabel="PSD", config=config)
    ax.set_xlim(0, 200)
    add_panel_label(ax, "c")
    
    # d) 时频图
    ax = axes[1, 1]
    f, t_spec, Sxx = signal.spectrogram(data[:, 32], fs=1000, nperseg=128)
    im = ax.pcolormesh(
        t_spec, f, 10 * np.log10(Sxx + 1e-12),
        shading="gouraud", cmap=config.colors.spectrogram,
        rasterized=True
    )
    add_colorbar(fig, im, ax, label="Power (dB)", config=config)
    setup_axis(ax, xlabel="Time (s)", ylabel="Frequency (Hz)", config=config)
    ax.set_ylim(0, 200)
    add_panel_label(ax, "d")
    
    # 保存为多种格式
    save_figure(fig, "output/demo_multi_panel", formats=("pdf", "png"))
    plt.close(fig)
    print("  -> 保存到 output/demo_multi_panel.pdf 和 .png")


def demo_publication_ready():
    """演示出版就绪的完整流程"""
    print("\n=== DASMatrix Nature/Science 级别可视化演示 ===\n")
    
    import os
    os.makedirs("output", exist_ok=True)
    
    demo_time_series()
    demo_heatmap()
    demo_spectrogram()
    demo_multi_panel()
    
    print("\n所有演示图表已生成！")
    print("提示: PDF 格式适合期刊投稿，PNG 格式适合预览")


if __name__ == "__main__":
    demo_publication_ready()
