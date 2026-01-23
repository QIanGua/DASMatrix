"""DASMatrix F-K 滤波示例

演示如何使用 F-K (频率-波数) 滤波分离不同传播速度的波场。
这是 DAS 数据处理中常用的高级技术。
"""

import matplotlib.pyplot as plt
import numpy as np

from DASMatrix import from_array
from DASMatrix.config import VisualizationConfig
from DASMatrix.visualization.styles import apply_nature_style, save_figure


def create_multi_velocity_data(
    fs: float = 1000,
    duration: float = 2.0,
    n_channels: int = 128,
    dx: float = 10.0,
) -> np.ndarray:
    """创建包含多种传播速度的合成 DAS 数据

    Args:
        fs: 采样频率 (Hz)
        duration: 信号持续时间 (s)
        n_channels: 通道数
        dx: 通道间距 (m)

    Returns:
        合成数据 (samples, channels)
    """
    t = np.linspace(0, duration, int(duration * fs))
    n_samples = len(t)
    x = np.arange(n_channels) * dx  # 空间坐标 (m)

    data = np.zeros((n_samples, n_channels))

    # 波 1: 快速波 (v = 2000 m/s, f = 30 Hz)
    v1, f1, A1 = 2000, 30, 1.0
    for i, xi in enumerate(x):
        delay = xi / v1
        if delay < duration:
            # 使用高斯包络的波包
            env = np.exp(-((t - delay - 0.3) ** 2) / 0.02)
            data[:, i] += A1 * env * np.sin(2 * np.pi * f1 * (t - delay))

    # 波 2: 慢速波 (v = 500 m/s, f = 50 Hz)
    v2, f2, A2 = 500, 50, 0.8
    for i, xi in enumerate(x):
        delay = xi / v2
        if delay < duration:
            env = np.exp(-((t - delay - 0.5) ** 2) / 0.03)
            data[:, i] += A2 * env * np.sin(2 * np.pi * f2 * (t - delay))

    # 添加噪声
    data += 0.1 * np.random.randn(*data.shape)

    return data


def demo_fk_filter():
    """演示 F-K 滤波功能"""
    print("=== DASMatrix F-K 滤波示例 ===\n")

    # 参数
    fs = 1000
    dx = 10.0  # 通道间距 10m

    # 创建合成数据
    print("1. 创建包含多速度波场的合成数据...")
    data = create_multi_velocity_data(fs=fs, dx=dx)
    frame = from_array(data, fs=fs)
    print(f"   数据形状: {frame.shape}")

    # 应用 F-K 滤波 - 保留快速波 (v > 1000 m/s)
    print("\n2. 应用 F-K 滤波 (保留 v > 1000 m/s)...")
    fast_waves = frame.fk_filter(v_min=1000, dx=dx)

    # 应用 F-K 滤波 - 保留慢速波 (300 < v < 800 m/s)
    print("3. 应用 F-K 滤波 (保留 300 < v < 800 m/s)...")
    slow_waves = frame.fk_filter(v_min=300, v_max=800, dx=dx)

    # 可视化
    print("\n4. 生成对比可视化...")
    apply_nature_style()
    config = VisualizationConfig()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    # 原始数据
    im = axes[0, 0].imshow(
        data.T,
        aspect="auto",
        origin="lower",
        cmap=config.colors.diverging,
        extent=[0, data.shape[0] / fs, 0, data.shape[1]],
        vmin=-2,
        vmax=2,
    )
    axes[0, 0].set_title("原始数据", fontweight="bold")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Channel")
    fig.colorbar(im, ax=axes[0, 0], label="Amplitude")

    # 快速波
    fast_data = fast_waves.collect()
    im = axes[0, 1].imshow(
        fast_data.T,
        aspect="auto",
        origin="lower",
        cmap=config.colors.diverging,
        extent=[0, fast_data.shape[0] / fs, 0, fast_data.shape[1]],
        vmin=-2,
        vmax=2,
    )
    axes[0, 1].set_title("快速波 (v > 1000 m/s)", fontweight="bold")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Channel")
    fig.colorbar(im, ax=axes[0, 1], label="Amplitude")

    # 慢速波
    slow_data = slow_waves.collect()
    im = axes[1, 0].imshow(
        slow_data.T,
        aspect="auto",
        origin="lower",
        cmap=config.colors.diverging,
        extent=[0, slow_data.shape[0] / fs, 0, slow_data.shape[1]],
        vmin=-2,
        vmax=2,
    )
    axes[1, 0].set_title("慢速波 (300 < v < 800 m/s)", fontweight="bold")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Channel")
    fig.colorbar(im, ax=axes[1, 0], label="Amplitude")

    # 单通道对比
    ch = 64
    t = np.arange(data.shape[0]) / fs
    axes[1, 1].plot(t, data[:, ch], "k-", alpha=0.5, label="原始", linewidth=0.8)
    axes[1, 1].plot(t, fast_data[:, ch], "b-", label="快速波", linewidth=config.line_width)
    axes[1, 1].plot(t, slow_data[:, ch], "r-", label="慢速波", linewidth=config.line_width)
    axes[1, 1].set_title(f"通道 {ch} 波形对比", fontweight="bold")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].set_xlim(0, 2)

    # 保存
    import os

    os.makedirs("output", exist_ok=True)
    save_figure(fig, "output/fk_filter_demo", formats=("png",))
    plt.close(fig)

    print("\n✓ F-K 滤波示例完成！")
    print("  输出文件: output/fk_filter_demo.png")


if __name__ == "__main__":
    demo_fk_filter()
