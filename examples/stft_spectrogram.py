"""
皮带机 DAS 信号批量分析脚本 (性能优化版)

遍历指定目录下的所有 .dat 文件，为每个文件生成：
1. 多通道波形图 (Waveform)
2. 多通道功率谱密度图 (PSD)
3. 多通道 STFT 时频图 (Spectrogram)
4. 多通道 RMS 分布图

使用多进程并行处理以提高效率。
"""

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
    wn: float = 900
    byte_order: str = "little"

    # 分析范围 (0-indexed, inclusive on both ends)
    start_ch: int = 0
    end_ch: int = 45  # Channels 0-45 (46 channels total)

    # STFT 参数
    window_size: int = 2048
    hop: int = 256

    # I/O 配置
    data_dir: Path = Path("/Users/qianlong/Code/DASMatrix/examples/dat")
    output_root: Path = Path("./plots/batch_analysis_0112_hp900Hz")

    # LFM 分析配置
    f_start: float = 1000.0
    f_end: float = 2000.0
    T: float = 3.0
    signal_band: tuple = (1000, 2000)  # 用于绘图标题兼容
    lfm_gamma: float = 2.0  # 降低阈值以包含更多信号 (was 8.0)
    lfm_delta_f: float = 150.0  # 增加频率容差 (was 80.0)
    lfm_keep_best_run: bool = False  # 允许断续 (was True)

    # 性能配置
    dpi: int = 100
    skip_existing: bool = False
    max_workers: int = 1
    # 仅生成 RMS 和方差图，跳过其他图（用于重新绘制 RMS/方差图）
    only_rms: bool = False

    @property
    def output_cols(self):
        # Convert inclusive end_ch to exclusive slice end (end_ch + 1)
        # e.g., start_ch=0, end_ch=45 -> slice [0:46] includes channels 0-45
        return (self.start_ch, self.end_ch + 1)


# 全局配置实例
CFG = ProcessingConfig(only_rms=False)

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


def init_worker_rms_only():
    """Worker 进程初始化函数（仅 RMS 模式）：只初始化滤波器"""
    # 仅预计算滤波器系数 (SOS)，readDat 需要用到
    sos = butter(N=4, Wn=CFG.wn, btype="highpass", fs=CFG.fs, output="sos")
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
    # diffMat -= diffMat.mean(axis=0)

    return diffMat


def calculate_signal_power(
    freq_data: np.ndarray, psd_data: np.ndarray, f_start: float, f_end: float
) -> float:
    """
    计算特定频带内的信号功率（去除背景噪声）。

    参数:
    freq_data: 频率数组 (Hz)
    psd_data: PSD数组 (线性单位，如 rad^2/Hz 或 strain^2/Hz，不能是dB)
    f_start, f_end: 频带范围
    """
    # 1. 截取 1kHz - 2kHz 的数据片段
    # 找到对应的索引范围
    mask = (freq_data >= f_start) & (freq_data <= f_end)
    f_band = freq_data[mask]
    p_band = psd_data[mask]

    if len(f_band) == 0:
        return 0.0

    # 2. 自动构建噪声基底 (模拟图中的红线)
    # 取频带最左侧和最右侧的 PSD 值作为噪声的“锚点”
    # 为了防止单点波动误差，建议取边缘 3-5 个点的平均值作为锚点
    n_avg_points = 5
    # 如果数据点太少，就取全部
    n_avg_points = min(n_avg_points, len(p_band))

    noise_start = np.mean(p_band[:n_avg_points])
    noise_end = np.mean(p_band[-n_avg_points:])

    # 生成噪声基线数组 (线性插值)
    # 这对应了图中连接左右两端的“红线”
    noise_line = np.linspace(noise_start, noise_end, len(f_band))

    # 3. 逐点相减 (总功率 - 噪声功率)
    p_net = p_band - noise_line

    # 4. 修正负值 (低于噪声基底的部分置零)
    p_net[p_net < 0] = 0

    # 5. 积分计算功率
    # 使用梯形积分公式 (np.trapz) 比简单的求和更准确
    signal_power = np.trapz(p_net, f_band)

    return float(signal_power)


def estimate_lfm_power_stft_known_chirp(
    x,
    fs,
    *,
    f_start=1000.0,
    f_end=2000.0,
    T=3.0,
    do_bandpass=True,
    bp_order=4,
    stft_win="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    gamma=8.0,
    delta_f_hz=80.0,
    mask_bw_hz=120.0,
    t_margin=0.10,
    mad_mult=3.5,
    max_robust_iter=3,
    keep_best_run=True,
    return_debug=True,
):
    """
    STFT 脊线检测 + 掩膜重构（已知 chirp 斜率），估计截断 LFM 的功率。
    """
    x = np.asarray(x, dtype=float).ravel()
    # Basic checks
    if x.size < 4:
        return {"P_lfm_active": 0.0, "active_duration_s": 0.0}

    fmin = min(f_start, f_end)
    fmax = max(f_start, f_end)
    a = (f_end - f_start) / T

    if np.isclose(a, 0.0):
        return {"P_lfm_active": 0.0}

    N = x.size
    x0 = x - np.mean(x)

    # 可选：带通到 [fmin, fmax]
    y = x0
    if do_bandpass:
        nyq = 0.5 * fs
        lo = max(1.0, fmin) / nyq
        hi = min(0.999, fmax / nyq)
        sos = butter(bp_order, [lo, hi], btype="bandpass", output="sos")
        y = sosfiltfilt(sos, y)

    # STFT 参数默认
    if nperseg is None:
        nperseg = int(round(0.040 * fs))
        nperseg = max(256, nperseg)
    if noverlap is None:
        noverlap = int(round(0.75 * nperseg))
    hop = nperseg - noverlap
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(nperseg)))

    # 使用 ShortTimeFFT (Modern API)
    # scale_to='magnitude' 对应 legacy stft 的 scaling (approx behavior for inversion consistency if we stick to same logic)
    # explicitly: SFT handles window scaling.
    sft_obj = ShortTimeFFT.from_window(
        stft_win, fs, nperseg, noverlap, mfft=nfft, scale_to="magnitude"
    )

    # 计算 STFT
    # 注意: legacy stft 默认 boundary=None, padded=False (如果我们在上面代码里显式这么写了)
    # ShortTimeFFT 默认 padding='zeros'。为了保持一致性或解决 NOLA，我们可以依赖 SFT 的默认行为 (usually safer).
    # 如果要完全模仿之前的 boundary=None，可以在 call stft 时指定 padding=None (CHECK API).
    # 但用户为了优化代码，建议使用标准做法。

    Z = sft_obj.stft(
        y
    )  # Shape: (freqs, times) or (times, freqs)? -> (n_freqs, n_times) usually
    # ShortTimeFFT.stft returns (n_freq, n_time) if axis is default.
    # Check bounds:
    f = sft_obj.f
    t = sft_obj.t(len(y))

    # SFT output dimensions match legacy stft roughly, but let's ensure indices align.
    # legacy stft returns Z with shape (n_freq, n_time) where n_freq = nfft//2 + 1

    P = np.abs(Z) ** 2
    band_idx = np.where((f >= fmin) & (f <= fmax))[0]
    if band_idx.size < 3:
        return {"P_lfm_active": 0.0}

    # 每帧：找频带内峰值点
    # Z shape is (n_freq, n_time).
    # If t from sft_obj.t(N) is longer than Z.shape[1] (due to padding defaults), we need to align.
    # usually Z has same time dim as t.

    # 截取有效部分 (Clip to match dimensions if needed, though they should match)
    n_time_frames = Z.shape[1]
    if t.size > n_time_frames:
        t = t[:n_time_frames]
    elif t.size < n_time_frames:
        # unexpected but handle
        Z = Z[:, : t.size]
        P = P[:, : t.size]

    P_band = P[band_idx, :]
    k_rel = np.argmax(P_band, axis=0)
    k_abs = band_idx[k_rel]
    f_hat = f[k_abs]
    peak_power = P[k_abs, np.arange(t.size)]
    noise_floor = np.median(P_band, axis=0) + 1e-20

    cand = peak_power > (gamma * noise_floor)
    cand_count = int(np.count_nonzero(cand))

    if cand_count < 5:
        return {"P_lfm_active": 0.0, "active_duration_s": 0.0}

    # 关键：用候选点估计 chirp 的时间对齐 t0
    tt = t[cand]
    ff = f_hat[cand]
    pp = peak_power[cand]
    t0_i = tt - (ff - f_start) / a

    # 改进：仅使用能量较高的点进行 t0 投票 (防止锁定到弱噪声)
    # 阈值：取最大能量的 10% (约 -10dB)
    if pp.size > 0:
        p_max_cand = np.max(pp)
        # 如果是强信号，只用强点 voting
        strong_idx = pp > (p_max_cand * 0.1)
        if np.count_nonzero(strong_idx) >= 5:
            t0_i_voting = t0_i[strong_idx]
        else:
            t0_i_voting = t0_i
    else:
        t0_i_voting = t0_i

    # MAD 鲁棒估计
    inlier = np.ones_like(t0_i, dtype=bool)
    t0_hat = float(np.median(t0_i_voting))
    for _ in range(max_robust_iter):
        r = t0_i - t0_hat
        mad = np.median(np.abs(r - np.median(r))) + 1e-12
        thr = mad_mult * 1.4826 * mad
        new_inlier = np.abs(r) <= thr
        if np.count_nonzero(new_inlier) < 5:
            break
        if np.all(new_inlier == inlier):
            break
        inlier = new_inlier
        t0_hat = float(np.median(t0_i[inlier]))

    # 定义 chirp 时间窗
    tau = t - t0_hat
    in_time = (tau >= -t_margin) & (tau <= T + t_margin)

    # 频率脊线预测 + 残差筛选
    f_pred = f_start + a * tau
    active_frames = cand & in_time & (np.abs(f_hat - f_pred) <= float(delta_f_hz))

    # 保留能量最大的一段连续区间
    if keep_best_run and np.any(active_frames):
        idx = np.flatnonzero(active_frames)
        runs = []
        if len(idx) > 0:
            s = idx[0]
            prev = idx[0]
            for cur in idx[1:]:
                if cur == prev + 1:
                    prev = cur
                else:
                    runs.append((s, prev))
                    s = cur
                    prev = cur
            runs.append((s, prev))

            best = None
            best_score = -np.inf
            for s, e in runs:
                score = float(np.sum(peak_power[s : e + 1]))
                if score > best_score:
                    best_score = score
                    best = (s, e)

            keep = np.zeros_like(active_frames, dtype=bool)
            if best is not None:
                keep[best[0] : best[1] + 1] = True
            active_frames = keep

    # 构造掩膜
    bw = float(mask_bw_hz) / 2.0
    f_band_val = f[band_idx]

    # Fix dimensions for broadcasting
    mask_band = (
        active_frames[None, :]
        & in_time[None, :]
        & (np.abs(f_band_val[:, None] - f_pred[None, :]) <= bw)
    )

    Z_lfm = np.zeros_like(Z)
    Z_lfm[band_idx, :] = Z[band_idx, :] * mask_band

    # 逆 STFT 重构 (使用 ShortTimeFFT)
    x_lfm_rec = sft_obj.istft(
        Z_lfm, k1=len(y)
    )  # k1 specifies expected output length (cuts padding)

    # 对齐长度
    if x_lfm_rec.size < N:
        x_lfm = np.pad(x_lfm_rec, (0, N - x_lfm_rec.size))
    else:
        x_lfm = x_lfm_rec[:N]

    # 帧 -> 样本映射
    active_samples = np.zeros(N, dtype=bool)
    for m, is_on in enumerate(active_frames):
        if not is_on:
            continue
        # Use calc logic consistent with SFT
        # SFT: frame m corresponds to sample range?
        # SFT lower_border_end maps frames.
        # Simple approx: m*hop to m*hop + nperseg
        start = m * hop
        end = start + nperseg
        if start >= N:
            break
        active_samples[start : min(end, N)] = True

    Na = int(np.count_nonzero(active_samples))
    P_lfm_active = float(np.mean((x_lfm[active_samples]) ** 2)) if Na > 0 else 0.0

    return {
        "P_lfm_active": P_lfm_active,
        "active_duration_s": float(Na / fs),
        "debug": {
            "t": t,
            "f_pred": f_pred,
            "mask": active_frames,
        },
    }


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
    band_power: np.ndarray = None,  # Optional: Pass band power values
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

        # 显示频带能量
        if band_power is not None:
            bp_val = band_power[i]
            # format as scientific notation
            ax.text(
                0.95,
                0.95,
                f"BP: {bp_val:.2e}",
                transform=ax.transAxes,
                fontsize=6,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6, pad=0.2),
            )

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
    max_time_pixels: int = 2000,
    ridges: list = None,  # List of debug dicts from LFM estimator
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

        # 叠加 LFM 脊线
        if ridges is not None and i < len(ridges):
            r = ridges[i]
            if r and "debug" in r:
                dbg = r["debug"]
                t_debug = dbg["t"]
                f_debug = dbg["f_pred"]
                mask = dbg["mask"]

                # Plot active segments
                # 为了防止散点过于密集，可以只画 mask=True 的点
                if np.any(mask):
                    ax.plot(
                        t_debug[mask],
                        f_debug[mask],
                        color="red",
                        marker=".",
                        linestyle="None",
                        markersize=1,
                        alpha=0.6,
                        label="Detected",
                    )

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
    var_values: np.ndarray,
    output_path: Path,
    file_label: str,
) -> None:
    """绘制多通道 RMS 和方差分布图 - 双子图布局"""
    start_ch = CFG.start_ch
    end_ch = CFG.end_ch
    channel_numbers = np.arange(start_ch, end_ch + 1)  # Array [0, 1, 2, ..., 45]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")

    # 子图1: RMS 分布
    ax1.bar(channel_numbers, rms_values, color="#3498db", alpha=0.8)
    ax1.plot(
        channel_numbers, rms_values, "o-", color="#e74c3c", linewidth=1.5, markersize=3
    )
    ax1.set_xlabel("Channel", fontsize=10)
    ax1.set_ylabel("RMS", fontsize=10)
    ax1.set_title("RMS Distribution", fontsize=11, fontweight="bold")
    ax1.set_xticks(channel_numbers[::4])
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    rms_stats = f"Mean: {np.mean(rms_values):.4f}, Max: {np.max(rms_values):.4f}"
    ax1.text(
        0.02,
        0.98,
        rms_stats,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # 子图2: 方差分布
    ax2.bar(channel_numbers, var_values, color="#27ae60", alpha=0.8)
    ax2.plot(
        channel_numbers, var_values, "o-", color="#8e44ad", linewidth=1.5, markersize=3
    )
    ax2.set_xlabel("Channel", fontsize=10)
    ax2.set_ylabel("Variance", fontsize=10)
    ax2.set_title("Variance Distribution", fontsize=11, fontweight="bold")
    ax2.set_xticks(channel_numbers[::4])
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    var_stats = f"Mean: {np.mean(var_values):.4f}, Max: {np.max(var_values):.4f}"
    ax2.text(
        0.02,
        0.98,
        var_stats,
        transform=ax2.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    fig.suptitle(f"RMS & Variance - {file_label}", fontsize=12, fontweight="bold")
    plt.savefig(output_path, dpi=CFG.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_band_power_distribution(
    signal_power: np.ndarray,
    output_path: Path,
    file_label: str,
) -> None:
    """绘制频带能量分布图 - 单图布局"""
    start_ch = CFG.start_ch
    end_ch = CFG.end_ch
    channel_numbers = np.arange(start_ch, end_ch + 1)

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 4), layout="constrained")

    # 子图1: 信号频带能量 (绝对值)
    ax1.bar(channel_numbers, signal_power, color="#e67e22", alpha=0.8)
    ax1.plot(
        channel_numbers,
        signal_power,
        "o-",
        color="#d35400",
        linewidth=1.5,
        markersize=3,
    )
    ax1.set_xlabel("Channel", fontsize=10)
    ax1.set_xlabel("Channel", fontsize=10)
    ax1.set_ylabel("LFM Power", fontsize=10)
    ax1.set_title("LFM Signal Power (Ridge Detected)", fontsize=11, fontweight="bold")
    ax1.set_xticks(channel_numbers[::4])
    ax1.set_xticks(channel_numbers[::4])
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    stats1 = f"Mean: {np.mean(signal_power):.2e}, Max: {np.max(signal_power):.2e}"
    ax1.text(
        0.02,
        0.98,
        stats1,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    fig.suptitle(f"Band Power Analysis - {file_label}", fontsize=12, fontweight="bold")
    plt.savefig(output_path, dpi=CFG.dpi, bbox_inches="tight")
    plt.close(fig)


def process_single_file_rms_only(dat_path: Path) -> tuple[str, bool, str]:
    """
    仅处理 RMS 和方差图 (独立进程中运行)
    跳过其他图的生成，用于重新绘制 RMS/方差图

    Returns:
        (文件名, 是否成功, 消息)
    """
    try:
        file_label = dat_path.stem
        time_stamp = file_label
        output_dir = CFG.output_root / time_stamp

        # 如果输出目录不存在，跳过（说明其他图还没生成）
        if not output_dir.exists():
            return (dat_path.name, False, "输出目录不存在，请先运行完整处理")

        output_dir.mkdir(parents=True, exist_ok=True)

        # 读取数据
        data_np = readDat(
            dat_path, fs=CFG.fs, channel_num=CFG.channels, output_cols=CFG.output_cols
        )

        if data_np.size == 0:
            return (dat_path.name, False, "Empty data array")

        n_ch = data_np.shape[1]

        # 仅计算 RMS 和方差
        rms_values = np.zeros(n_ch)
        var_values = np.zeros(n_ch)

        for i in range(n_ch):
            ch_data = data_np[:, i]
            # RMS 计算
            rms_values[i] = np.sqrt(np.mean(ch_data**2))
            # 方差计算
            var_values[i] = np.var(ch_data)

        # 仅生成 RMS/方差图
        plot_rms_distribution(
            rms_values, var_values, output_dir / "rms.png", time_stamp
        )

        return (dat_path.name, True, "RMS/方差图已更新")

    except Exception as e:
        return (dat_path.name, False, str(e))


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
        var_values = np.zeros(n_ch)
        signal_power = np.zeros(n_ch)
        psd_list = []
        lfm_results = []  # Store LFM details

        # 获取频率轴
        freqs = sft.f

        # 频率分辨率 (用于积分)
        df = freqs[1] - freqs[0]

        for i in range(n_ch):
            ch_data = data_np[:, i]  # 已经是 float32

            # 1. 核心计算：STFT
            Sx = sft.stft(ch_data)
            Sx_mag = np.abs(Sx)
            all_Sx.append(Sx_mag)

            # 2. 派生 PSD：从 STFT 结果推导 (Periodogram 平均)
            psd = np.mean(Sx_mag**2, axis=1) * (2 / (CFG.fs * np.sum(win**2)))
            psd_list.append(psd)

            # 3. 极速 RMS：时域计算最快
            rms_values[i] = np.sqrt(np.mean(ch_data**2))

            # 4. 方差计算 (数据已去均值，var = mean(x^2))
            var_values[i] = np.var(ch_data)

            # 5. LFM 能量计算 (脊线检测法)
            res = estimate_lfm_power_stft_known_chirp(
                ch_data,
                fs=CFG.fs,
                f_start=CFG.f_start,
                f_end=CFG.f_end,
                T=CFG.T,
                gamma=CFG.lfm_gamma,
                delta_f_hz=CFG.lfm_delta_f,
                keep_best_run=CFG.lfm_keep_best_run,
            )
            signal_power[i] = res["P_lfm_active"]
            lfm_results.append(res)

        all_freqs = [freqs] * n_ch
        extent = sft.extent(n_samples)

        # ===== 生成图表 =====
        plot_multichannel_waveform(data_np, output_dir / "waveform.png", time_stamp)
        plot_multichannel_psd(
            all_freqs,
            psd_list,
            output_dir / "psd.png",
            time_stamp,
            band_power=signal_power,
        )
        plot_multichannel_stft(
            all_Sx, extent, output_dir / "stft.png", time_stamp, ridges=lfm_results
        )
        plot_rms_distribution(
            rms_values, var_values, output_dir / "rms.png", time_stamp
        )
        plot_band_power_distribution(
            signal_power, output_dir / "band_power.png", time_stamp
        )

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
    if CFG.only_rms:
        print("皮带机 DAS 信号分析 - 仅重绘 RMS/方差图")
    else:
        print("皮带机 DAS 信号批量分析 (极致优化版 Phase 2)")
    print("=" * 60)

    CFG.output_root.mkdir(parents=True, exist_ok=True)

    # 获取所有 .dat 文件
    dat_files = sorted(CFG.data_dir.glob("*.dat"))
    total_files = len(dat_files)
    print(f"\n找到 {total_files} 个 DAT 文件")
    print(f"使用 {CFG.max_workers} 个并行进程")
    if CFG.only_rms:
        print("模式: 仅生成 RMS/方差图")
    else:
        print(f"跳过已存在: {CFG.skip_existing}")

    # 选择处理函数
    process_func = process_single_file_rms_only if CFG.only_rms else process_single_file

    # 使用进程池并行处理
    completed = 0
    failed = 0
    skipped = 0

    # 根据模式选择不同的初始化函数
    executor_kwargs = {"max_workers": CFG.max_workers}
    if CFG.only_rms:
        executor_kwargs["initializer"] = init_worker_rms_only
    else:
        executor_kwargs["initializer"] = init_worker

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        futures = {executor.submit(process_func, f): f for f in dat_files}

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
