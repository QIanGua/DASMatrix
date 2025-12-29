from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from DASMatrix.acquisition import DASReader, DataType
from DASMatrix.config import SamplingConfig
from DASMatrix.visualization import apply_nature_style

# 1. 配置采样参数
config = SamplingConfig(fs=16000, channels=46, byte_order="little")

# 2. 读取数据
reader = DASReader(config, data_type=DataType.DAT)
data_path = Path(
    "/Users/qianlong/Downloads/test/test/CH1_16000Hz_10s_46col_20251224_141409(0).dat"
)
data = reader.ReadRawData(data_path)


# 3. 计算统计量
def compute_correlation_matrix(data, eps=1e-8):
    """
    计算所有通道之间的相关系数矩阵

    Args:
        data: [T, C] 形状的原始数据 (T采样点，C通道数)
        eps: 避免除零的小常数

    Returns:
        corr_matrix: [C, C] 相关系数矩阵
        adjacent_corrs: [C-1] 相邻通道相关系数
    """
    print("正在计算相关系数矩阵...")
    if hasattr(data, "compute"):
        data = data.compute()

    # 转换为 float64
    X = data.astype(np.float64)  # [T, C]
    print(f"数据形状: {X.shape} (采样点数, 通道数)")

    # Step 1: 去均值 (每个通道减去时间平均值)
    X_centered = X - X.mean(axis=0, keepdims=True)

    # Step 2: L2 归一化 (每个通道除以其 L2 范数)
    norm = np.linalg.norm(X_centered, axis=0, keepdims=True)
    norm = np.where(norm < eps, 1.0, norm)  # 避免除零
    X_normalized = X_centered / norm

    # Step 3: 计算相关系数矩阵 (归一化向量的点积)
    # corr_matrix[i, j] = X_normalized[:, i] · X_normalized[:, j]
    corr_matrix = X_normalized.T @ X_normalized  # [C, C]

    # 提取相邻通道的相关系数 (对角线上方第一条)
    adjacent_corrs = np.diag(corr_matrix, k=1)  # [C-1]

    print(f"相关系数矩阵形状: {corr_matrix.shape}")
    print(f"相邻通道相关系数数量: {len(adjacent_corrs)}")

    return corr_matrix, adjacent_corrs


def compute_channel_rms(data):
    """
    计算每个通道的 RMS 值 (使用标准差 std 作为 AC RMS)
    Args:
        data: [T, C] 原始数据
    Returns:
        rms: [C] 每个通道的 RMS 值
    """
    print("正在计算通道 RMS...")
    if hasattr(data, "compute"):
        data = data.compute()

    X = data.astype(np.float64)
    # 使用标准差作为 RMS (去除直流分量)
    rms = np.std(X, axis=0)
    print(f"RMS 计算完成，形状: {rms.shape}")
    return rms


# 执行计算
corr_matrix, adjacent_corrs = compute_correlation_matrix(data)
channel_rms = compute_channel_rms(data)

# 4. 可视化
apply_nature_style()

# 阈值设置
high_corr_threshold = 0.3
adjacent_corrs_abs = np.abs(adjacent_corrs)  # 使用绝对值
high_corr_pairs = np.where(adjacent_corrs_abs > high_corr_threshold)[0]

# ===== 图1: 相关系数热力图 =====
fig1, ax1 = plt.subplots(figsize=(8, 7))
im = ax1.imshow(
    corr_matrix, aspect="equal", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"
)
ax1.set_xlabel("Channel")
ax1.set_ylabel("Channel")
ax1.set_title("Correlation Matrix")
cbar = plt.colorbar(im, ax=ax1, label="Correlation ($r$)", shrink=0.8)

# 添加通道刻度
n_channels = corr_matrix.shape[0]
tick_step = max(1, n_channels // 10)
ticks = np.arange(0, n_channels, tick_step)
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)

plt.savefig("./plots/correlation_matrix.png", dpi=300, bbox_inches="tight")
# plt.show()

# ===== 图2: 相邻点位相关图 (绝对值) =====
fig2, ax2 = plt.subplots(figsize=(10, 6))
channel_pairs = np.arange(len(adjacent_corrs_abs))

ax2.bar(
    channel_pairs, adjacent_corrs_abs, color="#0072B2", alpha=0.8, edgecolor="white"
)
ax2.axhline(
    y=high_corr_threshold,
    color="#D55E00",
    linestyle="--",
    alpha=0.7,
    label=f"$|r|$ = {high_corr_threshold}",
)

ax2.set_xlabel("Channel Pair Index ($i, i+1$)")
ax2.set_ylabel("Absolute Correlation ($|r|$)")
ax2.set_title("Adjacent Channel Correlation (Absolute)")
ax2.set_ylim(0, 1.1)
ax2.grid(True, axis="y", linestyle=":", alpha=0.4)

# 高亮高相关性对并标注点位号
if len(high_corr_pairs) > 0:
    ax2.bar(
        high_corr_pairs,
        adjacent_corrs_abs[high_corr_pairs],
        color="#CC79A7",
        alpha=0.9,
        edgecolor="white",
        label=f"$|r|$ > {high_corr_threshold} (n={len(high_corr_pairs)})",
    )
    # 标注点位号
    for idx in high_corr_pairs:
        ax2.annotate(
            f"{idx}-{idx + 1}",
            xy=(idx, adjacent_corrs_abs[idx]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#CC79A7",
        )
ax2.legend(loc="upper right")

plt.savefig("./plots/adjacent_correlation.png", dpi=300, bbox_inches="tight")
# plt.show()

# ===== 图3: RMS 分布图 =====
fig3, ax3 = plt.subplots(figsize=(10, 6))
channels = np.arange(len(channel_rms))

ax3.bar(channels, channel_rms, color="#009E73", alpha=0.8, edgecolor="white")

# 标注均值
mean_rms = float(np.mean(channel_rms))
ax3.axhline(
    y=mean_rms,
    color="#D55E00",
    linestyle="--",
    alpha=0.7,
    label=f"Mean RMS: {mean_rms:.2f}",
)

ax3.set_xlabel("Channel Index")
ax3.set_ylabel("RMS Amplitude")
ax3.set_title("Channel RMS Distribution")
ax3.grid(True, axis="y", linestyle=":", alpha=0.4)

# 标注最大/最小 RMS 通道
max_rms_idx = int(np.argmax(channel_rms))

ax3.annotate(
    f"Max: {channel_rms[max_rms_idx]:.2f}",
    xy=(max_rms_idx, float(channel_rms[max_rms_idx])),
    xytext=(0, 5),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=8,
    fontweight="bold",
    color="#009E73",
)

ax3.legend(loc="upper right")

plt.savefig("./plots/rms_distribution.png", dpi=300, bbox_inches="tight")
# plt.show()


# 打印统计信息
print("\n" + "=" * 50)
print("相关性分析结果汇总")
print("=" * 50)
print(f"通道数: {corr_matrix.shape[0]}")
print("相邻通道相关系数绝对值统计:")
print(f"  均值: {adjacent_corrs_abs.mean():.4f}")
print(f"  标准差: {adjacent_corrs_abs.std():.4f}")
min_idx = adjacent_corrs_abs.argmin()
print(f"  最小值: {adjacent_corrs_abs.min():.4f} (通道 {min_idx} ↔ {min_idx + 1})")
max_idx = adjacent_corrs_abs.argmax()
print(f"  最大值: {adjacent_corrs_abs.max():.4f} (通道 {max_idx} ↔ {max_idx + 1})")
print(f"\n高相关性通道对 (|r| > {high_corr_threshold}):")
for idx in high_corr_pairs:
    print(f"  通道 {idx} ↔ {idx + 1}: |r| = {adjacent_corrs_abs[idx]:.4f}")

print("\n" + "-" * 50)
print("RMS 统计结果")
print(f"RMS 均值: {mean_rms:.4f}")
print(f"RMS 标准差: {np.std(channel_rms):.4f}")
print(f"RMS 最小值: {channel_rms.min():.4f} (通道 {channel_rms.argmin()})")
print(f"RMS 最大值: {channel_rms.max():.4f} (通道 {channel_rms.argmax()})")
print("=" * 50)
