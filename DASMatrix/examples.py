"""DASMatrix 示例数据模块。

提供内置示例数据生成函数，方便快速测试和演示。

Example:
    >>> import DASMatrix as dm
    >>> frame = dm.get_example_frame('sine_wave')
    >>> print(frame.shape)
    (10000, 100)
"""

from typing import List, Literal, Optional

import numpy as np

from .api.dasframe import DASFrame
from .api.spool import DASSpool

# 示例类型定义
ExampleFrameType = Literal["random_das", "sine_wave", "chirp", "impulse", "event"]
ExampleSpoolType = Literal["diverse_das", "continuous"]


def get_example_frame(
    name: ExampleFrameType = "random_das",
    n_samples: int = 10000,
    n_channels: int = 100,
    fs: float = 1000.0,
    dx: float = 1.0,
    seed: Optional[int] = 42,
) -> DASFrame:
    """生成示例 DASFrame 数据。

    Args:
        name: 示例类型
            - 'random_das': 随机高斯噪声
            - 'sine_wave': 多频正弦波信号
            - 'chirp': 线性调频信号
            - 'impulse': 脉冲响应信号
            - 'event': 模拟地震事件
        n_samples: 时间采样点数
        n_channels: 通道数
        fs: 采样频率 (Hz)
        dx: 通道间距 (m)
        seed: 随机种子，用于可重复性

    Returns:
        DASFrame: 生成的示例数据

    Example:
        >>> frame = get_example_frame('sine_wave')
        >>> frame.shape
        (10000, 100)
        >>> frame = get_example_frame('chirp', n_samples=5000, fs=2000)
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n_samples) / fs

    if name == "random_das":
        data = _generate_random_das(n_samples, n_channels)
    elif name == "sine_wave":
        data = _generate_sine_wave(t, n_channels)
    elif name == "chirp":
        data = _generate_chirp(t, n_channels, fs)
    elif name == "impulse":
        data = _generate_impulse(n_samples, n_channels)
    elif name == "event":
        data = _generate_event(t, n_channels)
    else:
        raise ValueError(f"Unknown example type: {name}. Available: random_das, sine_wave, chirp, impulse, event")

    return DASFrame(data, fs=fs, dx=dx)


def get_example_spool(
    name: ExampleSpoolType = "diverse_das",
    n_frames: int = 3,
    **kwargs,
) -> DASSpool:
    """生成示例 DASSpool。

    Args:
        name: 示例类型
            - 'diverse_das': 包含多种不同信号类型的帧
            - 'continuous': 连续时间序列的多段数据
        n_frames: 生成的帧数量
        **kwargs: 传递给 get_example_frame 的参数

    Returns:
        DASSpool: 生成的示例 Spool

    Example:
        >>> spool = get_example_spool('diverse_das')
        >>> len(spool)
        3
    """
    frames = _generate_spool_frames(name, n_frames, **kwargs)
    return _MemorySpool(frames)


def list_example_types() -> dict:
    """列出所有可用的示例类型。

    Returns:
        dict: 包含 frame 和 spool 可用类型的字典
    """
    return {
        "frame": ["random_das", "sine_wave", "chirp", "impulse", "event"],
        "spool": ["diverse_das", "continuous"],
    }


# === 内部生成函数 ===


def _generate_random_das(n_samples: int, n_channels: int) -> np.ndarray:
    """生成随机高斯噪声数据。"""
    return np.random.randn(n_samples, n_channels).astype(np.float32)


def _generate_sine_wave(t: np.ndarray, n_channels: int) -> np.ndarray:
    """生成多频正弦波信号。"""
    data = np.zeros((len(t), n_channels), dtype=np.float32)

    # 多频率成分
    freqs = [10.0, 25.0, 50.0, 100.0]
    amps = [1.0, 0.5, 0.3, 0.2]

    for ch in range(n_channels):
        # 每个通道有相位偏移
        phase_shift = ch * 0.1
        signal = np.zeros(len(t))
        for freq, amp in zip(freqs, amps):
            signal += amp * np.sin(2 * np.pi * freq * t + phase_shift)

        # 添加少量噪声
        signal += 0.05 * np.random.randn(len(t))
        data[:, ch] = signal.astype(np.float32)

    return data


def _generate_chirp(t: np.ndarray, n_channels: int, fs: float) -> np.ndarray:
    """生成线性调频信号。"""
    from scipy.signal import chirp

    data = np.zeros((len(t), n_channels), dtype=np.float32)

    f0 = 10.0  # 起始频率
    f1 = fs / 4  # 终止频率 (Nyquist/2)

    for ch in range(n_channels):
        # 每个通道有时间偏移
        t_shift = ch * 0.001
        signal = chirp(t + t_shift, f0, t[-1], f1)
        signal += 0.1 * np.random.randn(len(t))
        data[:, ch] = signal.astype(np.float32)

    return data


def _generate_impulse(n_samples: int, n_channels: int) -> np.ndarray:
    """生成脉冲响应信号。"""
    data = np.zeros((n_samples, n_channels), dtype=np.float32)

    # 在不同位置添加脉冲
    for ch in range(n_channels):
        # 脉冲位置随通道变化（模拟传播）
        pulse_pos = n_samples // 4 + ch * 2
        if pulse_pos < n_samples:
            # 高斯脉冲
            pulse_width = 10
            x = np.arange(n_samples)
            pulse = np.exp(-((x - pulse_pos) ** 2) / (2 * pulse_width**2))
            data[:, ch] = pulse.astype(np.float32)

    # 添加噪声
    data += 0.01 * np.random.randn(n_samples, n_channels).astype(np.float32)

    return data


def _generate_event(t: np.ndarray, n_channels: int) -> np.ndarray:
    """生成模拟地震事件信号。"""
    data = np.zeros((len(t), n_channels), dtype=np.float32)

    # 事件参数
    event_time = t[-1] / 3
    velocity = 50  # m/s (假设 dx=1m)

    for ch in range(n_channels):
        # 计算到达时间
        arrival = event_time + ch / velocity

        # 创建衰减正弦波
        env = np.exp(-2 * np.maximum(0, t - arrival))
        carrier = np.sin(2 * np.pi * 30 * (t - arrival))
        signal = env * carrier

        # 只保留到达后的信号
        signal[t < arrival] = 0

        data[:, ch] = signal.astype(np.float32)

    # 添加背景噪声
    data += 0.02 * np.random.randn(len(t), n_channels).astype(np.float32)

    return data


def _generate_spool_frames(name: str, n_frames: int, **kwargs) -> List[DASFrame]:
    """生成 Spool 的帧列表。"""
    frames = []

    if name == "diverse_das":
        # 生成不同类型的帧
        types = ["sine_wave", "chirp", "event", "random_das", "impulse"]
        from typing import cast

        for i in range(n_frames):
            frame_type = types[i % len(types)]
            frame = get_example_frame(cast(ExampleFrameType, frame_type), seed=42 + i, **kwargs)
            frames.append(frame)

    elif name == "continuous":
        # 生成连续的正弦波帧
        for i in range(n_frames):
            frame = get_example_frame("sine_wave", seed=42 + i, **kwargs)
            frames.append(frame)

    else:
        raise ValueError(f"Unknown spool type: {name}")

    return frames


class _MemorySpool(DASSpool):
    """内存中的 Spool 实现，用于示例数据。"""

    def __init__(self, frames: List[DASFrame]):
        """初始化内存 Spool。

        Args:
            frames: DASFrame 列表
        """
        # 不调用父类 __init__，因为我们不需要文件路径
        self._frames = frames
        self._files = []
        self._meta_cache = {}
        self._index = None
        self._format = None

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> DASFrame:
        return self._frames[idx]

    def __iter__(self):
        return iter(self._frames)

    def update(self, force: bool = False) -> "_MemorySpool":
        """内存 Spool 不需要更新。"""
        return self


__all__ = [
    "get_example_frame",
    "get_example_spool",
    "list_example_types",
    "ExampleFrameType",
    "ExampleSpoolType",
]
