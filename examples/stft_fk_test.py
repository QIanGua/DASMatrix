"""
皮带机 DAS 信号 FK 滤波测试脚本

基于 stft_spectrogram.py 修改，增加：
1. FK 滤波 (视速度分离)
2. 带通滤波 (1000-2000Hz)
3. 滤波前后对比图
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT, butter, sosfiltfilt
from scipy.signal.windows import hann

from DASMatrix.config.sampling_config import SamplingConfig

# 引入 DASMatrix 处理器
from DASMatrix.processing.das_processor import DASProcessor

# Matplotlib 配置
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
    "Bitstream Vera Sans",
    "sans-serif",
]


@dataclass(frozen=True)
class ProcessingConfig:
    # 采样配置
    fs: int = 16000
    channels: int = 46
    wn: float = 0.1
    byte_order: str = "little"

    # 物理参数
    dx: float = 10.0  # 通道间距 (m) - 需确认实际值，暂定10m

    # 滤波配置
    # Bandpass: 1000-2000Hz
    bp_low: float = 1000.0
    bp_high: float = 2000.0

    # FK Filter: 这里的参数需要根据实际信号速度调整
    # 假设我们要保留某种速度的波，或者去除某种速度
    # 暂时设置为保留所有速度用于测试流程，或者设置一个宽范围
    # v_min 测试值: 1000 m/s (去除低速噪音)
    fk_v_min: float = 1000.0
    fk_v_max: Optional[float] = None  # None 表示不限制上限

    # 分析范围
    start_ch: int = 0
    end_ch: int = 45

    # STFT 参数
    window_size: int = 2048
    hop: int = 256

    # I/O 配置
    data_dir: Path = Path("/Users/qianlong/Downloads/data2/data")
    output_root: Path = Path("./plots/fk_test_1000_2000Hz")

    # 性能配置
    dpi: int = 100
    skip_existing: bool = False
    max_workers: int = max(1, mp.cpu_count() - 2)

    @property
    def output_cols(self):
        return (self.start_ch, self.end_ch + 1)


CFG = ProcessingConfig()
WORKER_CTX = {}


def init_worker():
    """Worker 进程初始化"""
    win = hann(CFG.window_size, sym=True)
    sft = ShortTimeFFT(
        win, hop=CFG.hop, fs=CFG.fs, mfft=CFG.window_size * 2, scale_to="magnitude"
    )

    # 预计算基础高通滤波器 (用于 readDat 中的预处理)
    sos = butter(N=4, Wn=CFG.wn, btype="highpass", fs=CFG.fs, output="sos")

    # 初始化 DASProcessor
    samp_config = SamplingConfig(fs=CFG.fs, channels=CFG.channels)
    processor = DASProcessor(samp_config)

    WORKER_CTX["win"] = win
    WORKER_CTX["sft"] = sft
    WORKER_CTX["sos"] = sos
    WORKER_CTX["processor"] = processor


def readDat(fpath, fs, channel_num, output_cols):
    """读取并预处理 DAT 文件 (保持原有的基础预处理)"""
    finfo = os.stat(fpath)
    byteCount = finfo.st_size
    rowNum = byteCount // (channel_num * 2)
    dtype = ">i2" if CFG.byte_order == "big" else "<i2"
    with open(fpath, "rb") as file:
        diff_data = np.fromfile(file, dtype=dtype, count=rowNum * channel_num)
    diff_data = diff_data.reshape((rowNum, channel_num))

    scale = np.pi / 8192.0
    diffMat = diff_data[:, output_cols[0] : output_cols[1]].astype(np.float32)
    diffMat *= scale

    # 基础高通滤波 (去漂移)
    sos = WORKER_CTX["sos"]
    diffMat = sosfiltfilt(sos, diffMat, axis=0)
    diffMat -= diffMat.mean(axis=0)

    return diffMat


def plot_comparison(
    orig_data: np.ndarray,
    filtered_data: np.ndarray,
    output_path: Path,
    file_label: str,
    title_suffix: str = "",
):
    """绘制原始数据与滤波后数据的对比图 (瀑布图/时空图)"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    # 统一色标范围
    vmax = np.std(orig_data) * 3
    vmin = -vmax

    # 原始
    im1 = axes[0].imshow(
        orig_data.T,
        aspect="auto",
        origin="lower",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axes[0].set_title("Original Data")
    axes[0].set_xlabel("Time Samples")
    axes[0].set_ylabel("Channel")
    plt.colorbar(im1, ax=axes[0], shrink=0.6)

    # 滤波后
    im2 = axes[1].imshow(
        filtered_data.T,
        aspect="auto",
        origin="lower",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axes[1].set_title(f"Filtered Data {title_suffix}")
    axes[1].set_xlabel("Time Samples")
    axes[1].set_ylabel("Channel")
    plt.colorbar(im2, ax=axes[1], shrink=0.6)

    fig.suptitle(f"Comparison - {file_label}", fontsize=14, fontweight="bold")
    plt.savefig(output_path, dpi=CFG.dpi)
    plt.close(fig)


def plot_stft_single(
    data: np.ndarray,
    output_path: Path,
    title: str,
    sft: ShortTimeFFT,
    channel_idx: int = 20,
):
    """绘制单通道 STFT"""
    if channel_idx >= data.shape[1]:
        channel_idx = 0

    Sx = sft.stft(data[:, channel_idx])
    Sx_mag = np.abs(Sx)
    n_samples = data.shape[0]
    extent = sft.extent(n_samples)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        Sx_mag,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmax=np.max(Sx_mag) * 0.5,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Magnitude")
    ax.set_title(f"{title} - Ch {CFG.start_ch + channel_idx}")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 4000)  # 重点关注低频段

    plt.savefig(output_path, dpi=CFG.dpi)
    plt.close(fig)


def process_single_file(dat_path: Path) -> tuple[str, bool, str]:
    try:
        processor: DASProcessor = WORKER_CTX["processor"]
        sft = WORKER_CTX["sft"]

        file_label = dat_path.stem
        output_dir = CFG.output_root / file_label

        if (
            CFG.skip_existing
            and output_dir.exists()
            and (output_dir / "comparison.png").exists()
        ):
            return (dat_path.name, True, "跳过")

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 读取原始数据
        data_orig = readDat(
            dat_path, fs=CFG.fs, channel_num=CFG.channels, output_cols=CFG.output_cols
        )
        if data_orig.size == 0:
            return (dat_path.name, False, "Empty data")

        # 2. 应用带通滤波 (1000-2000Hz)
        data_bp = processor.ApplyBandpassFilter(
            data_orig, low_freq=CFG.bp_low, high_freq=CFG.bp_high
        )

        # 3. 应用 FK 滤波
        # 注意：FKFilter 要求输入维度为 (Time, Channel)
        # data_orig 已经是 (n_samples, n_channels)
        data_fk = processor.FKFilter(
            data_bp,  # 对带通滤波后的数据再做 FK
            v_min=CFG.fk_v_min,
            v_max=CFG.fk_v_max,
            dx=CFG.dx,
        )

        # 4. 绘图对比

        # 4.1 原始 vs (BP + FK) 时空图对比
        plot_comparison(
            data_orig,
            data_fk,
            output_dir / "comparison_waveform.png",
            file_label,
            title_suffix=f"(BP {CFG.bp_low}-{CFG.bp_high}Hz + FK v>{CFG.fk_v_min})",
        )

        # 4.2 STFT 对比 (选一个中间通道)
        mid_ch = data_orig.shape[1] // 2
        plot_stft_single(
            data_orig, output_dir / "stft_orig.png", "Original STFT", sft, mid_ch
        )
        plot_stft_single(
            data_fk, output_dir / "stft_filtered.png", "Filtered STFT", sft, mid_ch
        )

        # 4.3 仅 FK (无 BP) 对比 - 为了 debug 单独看 FK 效果
        # data_only_fk = processor.FKFilter(
        #     data_orig, v_min=CFG.fk_v_min, v_max=CFG.fk_v_max, dx=CFG.dx
        # )
        # plot_comparison(
        #     data_orig,
        #     data_only_fk,
        #     output_dir / "comparison_only_fk.png",
        #     file_label,
        #     "(Only FK)",
        # )

        return (dat_path.name, True, "完成")

    except Exception as e:
        import traceback

        traceback.print_exc()
        return (dat_path.name, False, str(e))


def main():
    print("=" * 60)
    print("FK 滤波测试 (Bandpass 1000-2000Hz + FK)")
    print("=" * 60)

    # 找几个文件测试即可，不要全部跑
    dat_files = sorted(CFG.data_dir.glob("*.dat"))
    test_files = dat_files[:5]  # 只测前5个

    print(f"测试文件数: {len(test_files)}")
    print(f"FK v_min: {CFG.fk_v_min} m/s")
    print(f"BP Range: {CFG.bp_low}-{CFG.bp_high} Hz")

    with ProcessPoolExecutor(
        max_workers=CFG.max_workers, initializer=init_worker
    ) as executor:
        futures = {executor.submit(process_single_file, f): f for f in test_files}

        for future in as_completed(futures):
            fname = futures[future].name
            name, success, msg = future.result()
            print(f"Processed {fname}: {msg}")


if __name__ == "__main__":
    main()
