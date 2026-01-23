"""DSP 精度测试。"""

import numpy as np
from scipy import signal

import DASMatrix as dm


class TestDSPAccuracy:
    """信号处理精度测试套件。"""

    def test_bandpass_frequency_response(self):
        """验证带通滤波器频率响应精度。"""
        fs = 1000.0
        duration = 10.0
        n_samples = int(fs * duration)
        t = np.arange(n_samples) / fs

        # 生成扫频信号 (Chirp) 0-100Hz
        # 10s 内从 0 扫到 100Hz
        chirp_sig = signal.chirp(t, f0=0, t1=duration, f1=100)
        data = chirp_sig.reshape(-1, 1).astype(np.float32)

        frame = dm.from_array(data, fs=fs, dx=1.0)

        # 滤波: 40-60Hz
        low, high = 40, 60
        filtered = frame.bandpass(low, high).collect()

        # 计算频谱
        spec = np.abs(np.fft.rfft(filtered[:, 0]))
        freqs = np.fft.rfftfreq(n_samples, 1 / fs)

        # 验证通带 (50Hz)
        idx_50 = np.argmin(np.abs(freqs - 50))
        # 验证阻带 (20Hz, 80Hz)
        idx_20 = np.argmin(np.abs(freqs - 20))
        idx_80 = np.argmin(np.abs(freqs - 80))

        amp_50 = spec[idx_50]
        amp_20 = spec[idx_20]
        amp_80 = spec[idx_80]

        # 简单的验收标准：通带比阻带高至少 20dB (10倍幅度)
        assert amp_50 > amp_20 * 10, "Low frequency stopband attenuation failed"
        assert amp_50 > amp_80 * 10, "High frequency stopband attenuation failed"

    def test_fft_peak_location(self):
        """验证 FFT 峰值位置精度。"""
        fs = 1000.0
        n_samples = 10000
        t = np.arange(n_samples) / fs

        # 信号：精确的 50.0 Hz
        target_freq = 50.0
        sig = np.sin(2 * np.pi * target_freq * t)
        data = sig.reshape(-1, 1).astype(np.float32)

        frame = dm.from_array(data, fs=fs, dx=1.0)

        # 执行 FFT
        # 注意：DASMatrix fft() 返回的是 complex spectrum
        fft_frame = frame.fft()
        spec = fft_frame.collect()  # shape (n_freq, n_ch)

        # 获取频率坐标
        # 使用 DASFrame 自动计算的坐标
        freqs = fft_frame.data.frequency.values

        # 找到峰值频率
        # spec shape is (n_freq, n_ch)
        peak_idx = np.argmax(np.abs(spec[:, 0]))
        peak_freq = freqs[peak_idx]

        # 误差应小于频率分辨率 (df = fs/N = 0.1 Hz)
        df = fs / n_samples
        error = np.abs(peak_freq - target_freq)

        assert error < df, f"FFT peak error {error} Hz exceeds resolution {df} Hz"

    def test_integration_precision(self):
        """验证积分精度。"""
        fs = 1000.0
        t = np.arange(1000) / fs
        # 速度 v = cos(2*pi*f*t), 位移 d = sin(2*pi*f*t) / (2*pi*f)
        f = 10.0
        v = np.cos(2 * np.pi * f * t)
        expected_d = np.sin(2 * np.pi * f * t) / (2 * np.pi * f)

        # 初始条件 d(0) = 0
        data = v.reshape(-1, 1).astype(np.float32)
        frame = dm.from_array(data, fs=fs, dx=1.0)

        # 积分前先去直流，积分后去趋势以消除漂移
        integrated = frame.demean().integrate().detrend().collect()

        # 验证 (排除开头几个点的瞬态误差)
        result = integrated[100:, 0]
        target = expected_d[100:]

        # 移除目标信号的直流分量以便对比
        target = signal.detrend(target)

        rmse = np.sqrt(np.mean((result - target) ** 2))
        # 相对误差
        rel_error = rmse / (np.max(np.abs(target)) + 1e-9)

        # 能够接受的误差 (数值积分 vs 解析解)
        # 放宽到 5% 因为累积误差和边界效应
        assert rel_error < 0.05, f"Integration relative error {rel_error} too high"
