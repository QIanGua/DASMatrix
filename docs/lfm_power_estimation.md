# 基于 STFT 的 LFM 能量估计（已知 Chirp 参数）

本文档详细说明了 `estimate_lfm_power_stft_known_chirp` 函数的算法原理及使用方法。该函数旨在从噪声环境中鲁棒地估计线性调频（LFM/Chirp）信号的能量。算法假设 Chirp 信号的调频斜率 ($k$) 是已知的（由起始和终止频率及持续时间决定），但其到达时间（时间偏移量 $t_0$）是未知的。

## 概述

该算法包含以下核心步骤：

1.  **预处理 (Preprocessing)**：对信号进行零均值化，并可选地进行带通滤波以隔离感兴趣的频带。
2.  **时频分析 (Time-Frequency Analysis)**：计算信号的短时傅里叶变换（STFT），获取时频分布。
3.  **脊线检测 (Ridge Detection)**：在时频图上识别超出动态噪声底限的候选峰值点。
4.  **Chirp 对齐 (Chirp Alignment)**：利用鲁棒统计方法（MAD，中位数绝对偏差）基于候选点估计 Chirp 的起始时间偏移量 ($t_0$)。
5.  **掩膜生成 (Mask Generation)**：在预测的 Chirp 频率轨迹周围构建时频掩膜。
6.  **信号重构 (Signal Reconstruction)**：将掩膜应用于 STFT 数据，并通过逆 STFT (ISTFT) 重构时域 LFM 信号。
7.  **能量估计 (Power Estimation)**：计算重构信号在“激活”帧内的平均功率。

## 函数签名

```python
def estimate_lfm_power_stft_known_chirp(
    x: np.ndarray,
    fs: float,
    *,
    f_start: float = 1000.0,
    f_end: float = 2000.0,
    T: float = 3.0,
    do_bandpass: bool = True,
    bp_order: int = 4,
    stft_win: str = "hann",
    nperseg: int = None,
    noverlap: int = None,
    nfft: int = None,
    gamma: float = 8.0,
    delta_f_hz: float = 80.0,
    mask_bw_hz: float = 120.0,
    t_margin: float = 0.10,
    mad_mult: float = 3.5,
    max_robust_iter: int = 3,
    keep_best_run: bool = True,
    return_debug: bool = True,
) -> dict:
```

## 参数说明

### 输入信号
- **`x`** *(array-like)*: 输入的时域信号（1D 数组）。
- **`fs`** *(float)*: 采样率 (Hz)。

### Chirp 信号配置
- **`f_start`** *(float)*: LFM Chirp 的起始频率 (Hz)。
- **`f_end`** *(float)*: LFM Chirp 的终止频率 (Hz)。
- **`T`** *(float)*: Chirp 信号的持续时间 (秒)。
  - 调频斜率 $a$ (Hz/s) 计算为 $(f_{end} - f_{start}) / T$。

### 预处理参数
- **`do_bandpass`** *(bool)*: 是否在处理前应用带通滤波。默认为 `True`。
- **`bp_order`** *(int)*: 巴特沃斯带通滤波器的阶数。

### STFT 参数
- **`stft_win`** *(str)*: 窗函数名称（例如 "hann"）。
- **`nperseg`** *(int)*: 每个 STFT 窗口的长度。如果为 `None`，默认对应约 40ms。
- **`noverlap`** *(int)*:窗口重叠点数。默认为 `nperseg` 的 75%。
- **`nfft`** *(int)*: FFT 的长度。如果需要零填充可设置此值，默认通常为 `nperseg` 的下一个 2 的幂次方。

### 检测阈值
- **`gamma`** *(float)*: 峰值检测的信噪比倍数阈值。如果是 candidate 点，需满足 $P_{peak} > \gamma \times P_{noise}$。
- **`delta_f_hz`** *(float)*: 频率容差带 (±Hz)。在检测阶段，用于判断峰值点是否位于预测的 Chirp 轨迹附近。
- **`mask_bw_hz`** *(float)*: 重构掩膜的带宽 (±Hz)。在重构阶段，保留预测脊线周围多宽的频带。

### 鲁棒估计参数
- **`t_margin`** *(float)*: 时间裕量 (秒)。允许检测到的信号超出标称持续时间 $T$ 的范围。
- **`mad_mult`** *(float)*: MAD（中位数绝对偏差）乘数。用于在估计 $t_0$ 时剔除离群点（Outliers）。
- **`max_robust_iter`** *(int)*: $t_0$ 鲁棒估计迭代的最大次数。
- **`keep_best_run`** *(bool)*: 
  - `True`: 仅保留能量最大的一段连续信号段（处理断续信号时可能丢失部分能量）。
  - `False`: 保留所有符合模型的信号段（适合检测断续的 Chirp）。

## 算法详解

### 1. 预处理 (Preprocessing)
首先对信号 `x` 去除直流分量 (`x - mean(x)`)。如果 `do_bandpass` 为 True，则应用一个范围为 `[min(f_start, f_end), max(f_start, f_end)]` 的巴特沃斯带通滤波器。

### 2. 候选点检测 (Candidate Point Detection)
对 STFT 的每一帧：
1.  提取感兴趣频带内的功率谱。
2.  找到该频带内的峰值频率及其功率值。
3.  使用频带内功率的中位数估计噪声底限 (Noise Floor)。
4.  如果 `峰值功率 > gamma * 噪声底限`，则标记该点为 **候选点**。

### 3. $t_0$ 估计 (Time Alignment)
对于所有候选点 $(t_i, f_i)$，假设它们符合线性关系：
$$ f_i = f_{start} + a \cdot (t_i - t_0) $$
反解出 $t_0$：
$$ t_0^{(i)} = t_i - \frac{f_i - f_{start}}{a} $$
通过鲁棒统计方法估计真实的 $t_0$：
1.  计算所有 $t_0^{(i)}$ 的中位数。
2.  使用 MAD 阈值 (`mad_mult`) 迭代剔除离群点，直到收敛。

### 4. 脊线筛选与掩膜 (Ridge Filtering & Masking)
确定 $t_0$ 后，预测的频率轨迹为：
$$ f_{pred}(t) = f_{start} + a \cdot (t - t_{0_{hat}}) $$
筛选出“激活帧” (Active Frames)，需满足：
1.  该帧包含候选峰值。
2.  峰值频率在预测轨迹的 `delta_f_hz` 范围内。
3.  时间在有效 Chirp 窗口 $[-t_{margin}, T + t_{margin}]$ 内。

如果 `keep_best_run` 为 True，将仅保留总能量最高的那一段连续激活帧。

### 5. 重构与能量计算 (Reconstruction & Power)
在 STFT 域构建二进制掩膜：
- **时间维度**：仅保留激活帧的索引。
- **频率维度**：仅保留距离 $f_{pred}(t)$ 在 `mask_bw_hz`（半带宽）范围内的频率点。

应用掩膜后的 STFT 数据通过逆变换 (`istft`) 重构为时域信号 `x_lfm`。最终返回的功率 `P_lfm_active` 是 `x_lfm` 在所有激活帧对应的时间样本上的均方值。

## 返回值

函数返回一个字典：

```python
{
    "P_lfm_active": float,       # 估计的 LFM 信号功率
    "active_duration_s": float,  # 检测到的信号总有效时长 (秒)
    "debug": {                   # 用于绘图的调试信息
        "t": np.array,           # STFT 时间轴
        "f_pred": np.array,      # 基于估计 t0 的预测频率轨迹
        "mask": np.array         # 激活时间帧的布尔掩膜
    }
}
```

## 使用示例

```python
import numpy as np
from scipy.signal import chirp
from examples.stft_spectrogram import estimate_lfm_power_stft_known_chirp

# 1. 生成带噪声的 LFM 信号
fs = 16000
T = 3.0
t = np.arange(0, T, 1/fs)
# 信号: 1000Hz -> 2000Hz
clean_sig = chirp(t, f0=1000, t1=T, f1=2000, method='linear')
noise = np.random.normal(0, 0.5, size=len(t))
x = clean_sig + noise

# 2. 估计功率
result = estimate_lfm_power_stft_known_chirp(
    x,
    fs=fs,
    f_start=1000.0,
    f_end=2000.0,
    T=T,
    gamma=2.0,       # 灵敏度阈值
    delta_f_hz=150.0, # 频率搜索带宽
    keep_best_run=False
)

print(f"LFM 估计功率: {result['P_lfm_active']:.4f}")
print(f"有效时长: {result['active_duration_s']:.2f}s")
```
