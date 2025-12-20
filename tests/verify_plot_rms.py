import matplotlib.pyplot as plt
import numpy as np

from DASMatrix.api import df


def verify_integrated_plots():
    print("Starting verification of integrated plotting methods...")

    # 1. 创建合成数据
    fs = 1000.0
    n_samples = 2000
    n_channels = 64
    data = np.random.randn(n_samples, n_channels)

    # 在 20-40 通道添加一些显著信号
    data[:, 20:40] += (
        5.0 * np.sin(2 * np.pi * 50 * np.linspace(0, 2, n_samples))[:, np.newaxis]
    )

    # 2. 创建 DASFrame
    frame = df.from_array(data, fs=fs, dx=1.0)
    print(f"Created DASFrame with shape: {frame.shape}")

    # 3. 测试 plot_rms
    print("Testing plot_rms()...")
    fig_rms = frame.plot_rms(title="Integrated RMS Verification")
    assert isinstance(fig_rms, plt.Figure)
    assert fig_rms.axes[0].get_title() == "Integrated RMS Verification"
    plt.close(fig_rms)

    # 4. 测试 plot_mean
    print("Testing plot_mean()...")
    fig_mean = frame.plot_mean(title="Integrated Mean Verification")
    assert isinstance(fig_mean, plt.Figure)
    plt.close(fig_mean)

    # 5. 测试 plot_std
    print("Testing plot_std()...")
    fig_std = frame.plot_std(title="Integrated Std Verification")
    assert isinstance(fig_std, plt.Figure)
    plt.close(fig_std)

    # 6. 测试链式调用
    print("Testing chained call: frame.detrend().plot_rms()...")
    fig_chained = frame.detrend().plot_rms()
    assert isinstance(fig_chained, plt.Figure)
    plt.close(fig_chained)

    print("All integrated plot tests passed successfully!")


if __name__ == "__main__":
    verify_integrated_plots()
