from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from DASMatrix.acquisition import DASReader, DataType
from DASMatrix.config import SamplingConfig
from DASMatrix.visualization import apply_nature_style

# 1. 配置采样参数
config = SamplingConfig(fs=16000, channels=46, wn=0.1, byte_order="little")

# 2. 读取数据
reader = DASReader(config, data_type=DataType.DAT)
data_path = Path(
    "/Users/qianlong/Downloads/test/test/CH1_16000Hz_10s_46col_20251224_141158(0).dat"
)
data = reader.ReadRawData(data_path)


# ===== 绘制点位 10-46 的 STFT 时频图 =====
print("\n正在绘制 STFT 时频图 (点位 10-46)...")

# 预计算数据
if hasattr(data, "compute"):
    data_np = np.asarray(data.compute())  # type: ignore[union-attr]
else:
    data_np = np.asarray(data)

# 选择的通道范围 (0-indexed: 通道 10-46 对应索引 9-45)
start_ch, end_ch = 10, 46  # 1-indexed 通道号
ch_start_idx, ch_end_idx = start_ch - 1, end_ch  # 转换为 0-indexed 切片
n_plot_channels = end_ch - start_ch + 1  # 37 个通道

# 计算子图网格布局
n_cols = 6
n_rows = int(np.ceil(n_plot_channels / n_cols))

# STFT 参数 - 增大窗口提高频率分辨率
window_size = 2048  # 频率分辨率 = fs/window_size = 16000/2048 ≈ 7.8 Hz
overlap = 0.875  # 87.5% 重叠

# 使用 SpectrogramPlot 绘制每个通道
apply_nature_style()

# 创建子图
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(24, 3.5 * n_rows), constrained_layout=True
)
axes_flat = axes.flatten()

# 计算所有通道的 STFT 以获取统一的颜色范围
print("计算所有通道的频谱数据...")
all_Sxx_db = []
sample_f = None
sample_t = None

noverlap = int(window_size * overlap)

for ch_idx in range(ch_start_idx, ch_end_idx):
    ch_data = data_np[:, ch_idx].astype(np.float64)
    f, t, Sxx = signal.spectrogram(
        ch_data,
        fs=config.fs,
        window="hann",
        nperseg=window_size,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        mode="magnitude",  # 线性幅度
    )
    # 线性尺度，不转换为 dB
    all_Sxx_db.append(Sxx)
    if sample_f is None:
        sample_f = f
        sample_t = t

# 计算统一的颜色范围 (线性尺度)
# all_values = np.concatenate([s.flatten() for s in all_Sxx_db])
# vmin = np.percentile(all_values, 1)
# vmax = np.percentile(all_values, 99)
# print(f"频谱幅度范围 (1%-99% 百分位): [{vmin:.4f}, {vmax:.4f}]")

# 打印频段能量统计
assert sample_f is not None and sample_t is not None, "No spectrogram data computed"
print("\n各频段平均幅度统计 (Ch 30):")
ch30_idx = 30 - start_ch  # 转换为列表索引
freq_bands = [
    (0, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 6000),
    (6000, 8000),
]
for fmin, fmax in freq_bands:
    mask = (sample_f >= fmin) & (sample_f < fmax)
    band_mean = all_Sxx_db[ch30_idx][mask, :].mean()
    print(f"  {fmin:4d}-{fmax:4d} Hz: mean={band_mean:.6f}")

# 绘制每个通道
print("\n绘制子图...")
for i, ch_idx in enumerate(range(ch_start_idx, ch_end_idx)):
    ax = axes_flat[i]
    ch_num = ch_idx + 1

    im = ax.pcolormesh(
        sample_t,
        sample_f,
        all_Sxx_db[i],
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=0.01,
        rasterized=True,
    )
    ax.set_ylabel("Freq (Hz)", fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title(f"Ch {ch_num}", fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", labelsize=7)
    # 限制频率范围，便于观察
    ax.set_ylim(0, 8000)

# 隐藏空白子图
for j in range(n_plot_channels, len(axes_flat)):
    axes_flat[j].axis("off")

# 添加统一的 colorbar
cbar = fig.colorbar(
    im,
    ax=axes_flat[:n_plot_channels],
    label="Magnitude (Linear)",
    shrink=0.8,
    aspect=40,
)

fig.suptitle(
    "STFT Spectrogram (Linear Scale) - Channels 10 to 46",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)
plt.savefig("./plots/stft_ch10_46_linear_off.png", dpi=200, bbox_inches="tight")
print("\nSTFT 时频图已保存至 ./plots/stft_ch10_46_linear_off.png")
