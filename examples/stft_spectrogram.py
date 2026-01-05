"""
皮带机 DAS 信号批量分析脚本 (性能优化版)

遍历指定目录下的所有 .dat 文件，为每个文件生成：
1. 多通道波形图 (Waveform)
2. 多通道功率谱密度图 (PSD)
3. 多通道 STFT 时频图 (Spectrogram)
4. 多通道 RMS 分布图

使用多进程并行处理以提高效率。
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import (  # ty:ignore[unresolved-import]
    ShortTimeFFT,
    butter,
    sosfiltfilt,
)
from scipy.signal.windows import hann

# Matplotlib 配置：使用非交互式后端
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
    "Bitstream Vera Sans",
    "sans-serif",
]

# ============================================================================
# 配置参数 (Dataclass)
# ============================================================================


@dataclass(frozen=True)
class ProcessingConfig:
    # 采样配置
    fs: int = 16000
    channels: int = 46
    wn: float = 0.1
    byte_order: str = "little"

    # 分析范围 (0-indexed, inclusive on both ends)
    start_ch: int = 0
    end_ch: int = 45  # Channels 0-45 (46 channels total)

    # STFT 参数
    window_size: int = 2048
    hop: int = 256

    # I/O 配置
    data_dir: Path = Path("/Users/qianlong/Downloads/data2/data")
    output_root: Path = Path("./plots/batch_analysis_1231")

    # 性能配置
    dpi: int = 100
    skip_existing: bool = True
    max_workers: int = max(1, mp.cpu_count() - 2)

    @property
    def output_cols(self):
        # Convert inclusive end_ch to exclusive slice end (end_ch + 1)
        # e.g., start_ch=0, end_ch=45 -> slice [0:46] includes channels 0-45
        return (self.start_ch, self.end_ch + 1)


# 全局配置实例
CFG = ProcessingConfig()

# 全局 Worker 状态 (将在 init_worker 中初始化)
WORKER_CTX = {}


def init_worker():
    """Worker 进程初始化函数：预计算并缓存重型对象"""
    # 1. 预计算窗函数和 STFT 对象
    win = hann(CFG.window_size, sym=True)
    sft = ShortTimeFFT(
        win, hop=CFG.hop, fs=CFG.fs, mfft=CFG.window_size * 2, scale_to="magnitude"
    )

    # 2. 预计算滤波器系数 (SOS)
    sos = butter(N=4, Wn=CFG.wn, btype="highpass", fs=CFG.fs, output="sos")

    # 存入全局字典
    WORKER_CTX["win"] = win
    WORKER_CTX["sft"] = sft
    WORKER_CTX["sos"] = sos


def readDat(fpath, fs, channel_num, output_cols):
    """
    读取并预处理 DAT 文件
    """
    finfo = os.stat(fpath)
    byteCount = finfo.st_size
    rowNum = byteCount // (channel_num * 2)
    # 根据endian指定字节序
    dtype = ">i2" if CFG.byte_order == "big" else "<i2"
    with open(fpath, "rb") as file:
        diff_data = np.fromfile(file, dtype=dtype, count=rowNum * channel_num)
    diff_data = diff_data.reshape((rowNum, channel_num))

    # 只输出 output_cols 列
    scale = np.pi / 8192.0
    diffMat = diff_data[:, output_cols[0] : output_cols[1]].astype(np.float32)
    diffMat *= scale

    # 使用 Worker 预计算的 SOS 进行滤波
    sos = WORKER_CTX["sos"]
    diffMat = sosfiltfilt(sos, diffMat, axis=0)

    # 矢量化去均值 (In-place)
    diffMat -= diffMat.mean(axis=0)

    return diffMat


# ============================================================================
# 绘图函数 (优化版)
# ============================================================================


def plot_multichannel_waveform(
    data_input: np.ndarray,
    output_path: Path,
    file_label: str,
    time_range: tuple = (0, 1),
    max_plot_points: int = 5000,
) -> None:
    """绘制多通道波形图 (带降采样优化)"""
    start_ch = CFG.start_ch

    n_plot_channels = data_input.shape[1]
    n_cols = 6
    n_rows = int(np.ceil(n_plot_channels / n_cols))

    fs = CFG.fs
    start_sample = int(time_range[0] * fs)
    end_sample = int(time_range[1] * fs)

    # 降采样逻辑
    n_samples = end_sample - start_sample
    step = max(1, n_samples // max_plot_points)
    slice_idx = slice(start_sample, end_sample, step)

    time_axis = np.arange(start_sample, end_sample, step) / fs
    all_data_subset = data_input[slice_idx, :]

    y_min, y_max = np.min(all_data_subset), np.max(all_data_subset)
    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2 * n_rows))
    fig.subplots_adjust(
        top=0.92, bottom=0.06, left=0.04, right=0.98, hspace=0.5, wspace=0.3
    )
    axes_flat = axes.flatten()

    n_ch = data_input.shape[1]
    for i in range(n_ch):
        ax = axes_flat[i]
        # 使用降采样后的数据绘图
        ax.plot(time_axis, data_input[slice_idx, i], linewidth=0.3, color="#2980b9")
        ax.set_title(f"Ch {start_ch + i}", fontsize=8, fontweight="bold")
        ax.tick_params(axis="both", labelsize=5)
        ax.set_xlim(time_range)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    for j in range(n_plot_channels, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"Waveform - {file_label}", fontsize=10, fontweight="bold")
    plt.savefig(output_path, dpi=CFG.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_multichannel_psd(
    all_freqs: list,
    all_psd: list,
    output_path: Path,
    file_label: str,
    freq_range: tuple = (0, 8000),
) -> None:
    """绘制多通道功率谱密度图 (PSD)"""
    start_ch = CFG.start_ch
    n_plot_channels = len(all_psd)

    n_cols = 6
    n_rows = int(np.ceil(n_plot_channels / n_cols))

    all_psd_values = np.concatenate(all_psd)
    psd_min = np.min(all_psd_values[all_psd_values > 0])
    psd_max = np.max(all_psd_values)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    fig.subplots_adjust(
        top=0.92, bottom=0.06, left=0.04, right=0.98, hspace=0.4, wspace=0.3
    )
    axes_flat = axes.flatten()

    for i in range(n_plot_channels):
        ax = axes_flat[i]
        ax.semilogy(all_freqs[i], all_psd[i], linewidth=0.5, color="#2980b9")
        ax.fill_between(all_freqs[i], all_psd[i], alpha=0.3, color="#3498db")
        ax.set_title(f"Ch {start_ch + i}", fontsize=8, fontweight="bold")
        ax.tick_params(axis="both", labelsize=5)
        ax.set_xlim(freq_range)
        ax.set_ylim(psd_min * 0.5, psd_max * 2)

    for j in range(n_plot_channels, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"PSD - {file_label}", fontsize=10, fontweight="bold")
    plt.savefig(output_path, dpi=CFG.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_multichannel_stft(
    all_Sx: list,
    extent: tuple,
    output_path: Path,
    file_label: str,
    freq_range: tuple = (0, 8000),
    vmin: float = 0,
    vmax: float = 0.01,
    max_time_pixels: int = 2000,  # 语谱图降采样目标像素
) -> None:
    """绘制多通道 STFT 时频图 (带降采样优化)"""
    start_ch = CFG.start_ch
    n_plot_channels = len(all_Sx)
    if n_plot_channels == 0:
        return

    n_cols = 6
    n_rows = int(np.ceil(n_plot_channels / n_cols))

    # 准备降采样切片
    original_width = all_Sx[0].shape[1]
    step = max(1, original_width // max_time_pixels)
    slice_idx = slice(None, None, step)

    # x轴范围也需要调整吗？extent 应该保持不变，因为是物理坐标
    # 但 imshow 的数据矩阵变小了，align 会对齐到 extent

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(22, 3 * n_rows), constrained_layout=True
    )
    axes_flat = axes.flatten()

    im = None  # Initialize im outside the loop
    for i in range(n_plot_channels):
        ax = axes_flat[i]
        # 降采样渲染
        # Assign im only once for the colorbar
        if im is None:
            im = ax.imshow(
                all_Sx[i][:, slice_idx],
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )  # nearest更快
        else:
            ax.imshow(
                all_Sx[i][:, slice_idx],
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )  # nearest更快
        ax.set_title(f"Ch {start_ch + i}", fontsize=8, fontweight="bold")
        ax.tick_params(axis="both", labelsize=5)
        ax.set_ylim(freq_range)

    for j in range(n_plot_channels, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.colorbar(
        im, ax=axes_flat[:n_plot_channels], label="Mag", shrink=0.5, aspect=25, pad=0.01
    )
    fig.suptitle(f"STFT - {file_label}", fontsize=10, fontweight="bold")
    plt.savefig(output_path, dpi=CFG.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_rms_distribution(
    rms_values: np.ndarray,
    output_path: Path,
    file_label: str,
) -> None:
    """绘制多通道 RMS 分布图 - 使用预计算的 RMS 值"""
    start_ch = CFG.start_ch
    end_ch = CFG.end_ch
    channel_numbers = np.arange(start_ch, end_ch + 1)  # Array [0, 1, 2, ..., 45]

    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.bar(channel_numbers, rms_values, color="#3498db", alpha=0.8)
    ax.plot(
        channel_numbers, rms_values, "o-", color="#e74c3c", linewidth=1.5, markersize=3
    )
    ax.set_xlabel("Channel", fontsize=10)
    ax.set_ylabel("RMS", fontsize=10)
    ax.set_title(f"RMS - {file_label}", fontsize=12, fontweight="bold")
    ax.set_xticks(channel_numbers[::4])
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    stats_text = f"Mean: {np.mean(rms_values):.4f}, Max: {np.max(rms_values):.4f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.savefig(output_path, dpi=CFG.dpi, bbox_inches="tight")
    plt.close(fig)


def process_single_file(dat_path: Path) -> tuple[str, bool, str]:
    """
    处理单个 DAT 文件 (独立进程中运行)

    Returns:
        (文件名, 是否成功, 消息)
    """
    try:
        # 从全局 Worker Context 获取对象
        win = WORKER_CTX["win"]
        sft = WORKER_CTX["sft"]

        file_label = dat_path.stem
        # time_stamp = file_label.split("_")[-1].replace("(0)", "")
        # 解析新格式文件名: "20251231_151243.dat"
        # 格式: YYYYMMDD_HHMMSS
        time_stamp = file_label  # 直接使用完整的 stem 作为时间戳
        output_dir = CFG.output_root / time_stamp

        # 跳过已存在的
        if (
            CFG.skip_existing
            and output_dir.exists()
            and (output_dir / "rms.png").exists()
        ):
            return (dat_path.name, True, "跳过 (已存在)")

        output_dir.mkdir(parents=True, exist_ok=True)

        # 读取数据
        data_np = readDat(
            dat_path, fs=CFG.fs, channel_num=CFG.channels, output_cols=CFG.output_cols
        )

        if data_np.size == 0:
            return (dat_path.name, False, "Empty data array")

        n_samples = data_np.shape[0]
        n_ch = data_np.shape[1]

        # ===== 极速计算：整合分析流程 =====
        # 预分配空间
        all_Sx = []
        rms_values = np.zeros(n_ch)
        psd_list = []

        # 获取频率轴
        freqs = sft.f

        for i in range(n_ch):
            ch_data = data_np[:, i]  # 已经是 float32

            # 1. 核心计算：STFT
            Sx = sft.stft(ch_data)
            Sx_mag = np.abs(Sx)
            all_Sx.append(Sx_mag)

            # 2. 派生 PSD：从 STFT 结果推导 (Periodogram 平均)
            # 2. 派生 PSD：从 STFT 结果推导 (Periodogram 平均)
            psd = np.mean(Sx_mag**2, axis=1) * (2 / (CFG.fs * np.sum(win**2)))
            psd_list.append(psd)

            # 3. 极速 RMS：时域计算最快
            rms_values[i] = np.sqrt(np.mean(ch_data**2))

        all_freqs = [freqs] * n_ch
        extent = sft.extent(n_samples)

        # ===== 生成图表 =====
        plot_multichannel_waveform(data_np, output_dir / "waveform.png", time_stamp)
        plot_multichannel_psd(all_freqs, psd_list, output_dir / "psd.png", time_stamp)
        plot_multichannel_stft(all_Sx, extent, output_dir / "stft.png", time_stamp)
        plot_rms_distribution(rms_values, output_dir / "rms.png", time_stamp)

        return (dat_path.name, True, "完成")

    except Exception as e:
        return (dat_path.name, False, str(e))


# ============================================================================
# 主程序
# ============================================================================


def main():
    """主程序入口 (多进程版本)"""
    print("=" * 60)
    print("=" * 60)
    print("皮带机 DAS 信号批量分析 (极致优化版 Phase 2)")
    print("=" * 60)

    CFG.output_root.mkdir(parents=True, exist_ok=True)

    # 获取所有 .dat 文件
    dat_files = sorted(CFG.data_dir.glob("*.dat"))
    total_files = len(dat_files)
    print(f"\n找到 {total_files} 个 DAT 文件")
    print(f"使用 {CFG.max_workers} 个并行进程 (带 Worker Initializer)")
    print(f"跳过已存在: {CFG.skip_existing}")

    # 使用进程池并行处理
    completed = 0
    failed = 0
    skipped = 0

    # 使用 initializer 预加载资源
    with ProcessPoolExecutor(
        max_workers=CFG.max_workers, initializer=init_worker
    ) as executor:
        futures = {executor.submit(process_single_file, f): f for f in dat_files}

        for future in as_completed(futures):
            filename, success, msg = future.result()
            completed += 1

            if "跳过" in msg:
                skipped += 1
                status = "⏭️"
            elif success:
                status = "✓"
            else:
                status = "✗"
                failed += 1

            print(f"[{completed:3d}/{total_files}] {status} {filename[:40]}... {msg}")

    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"  成功: {completed - failed - skipped}")
    print(f"  跳过: {skipped}")
    print(f"  失败: {failed}")
    print(f"输出目录: {CFG.output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
