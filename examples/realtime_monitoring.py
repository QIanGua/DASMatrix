"""DASMatrix 实时监测示例

演示如何使用 DASMatrix 进行流式数据处理和实时事件检测。
适用于 DAS 在线监控场景。
"""

import time
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from DASMatrix import from_array
from DASMatrix.visualization.styles import apply_nature_style


def simulate_das_stream(
    fs: float = 1000,
    n_channels: int = 64,
    chunk_duration: float = 0.5,
    event_probability: float = 0.3,
) -> np.ndarray:
    """模拟 DAS 数据流的一个数据块

    Args:
        fs: 采样频率 (Hz)
        n_channels: 通道数
        chunk_duration: 数据块时长 (s)
        event_probability: 事件发生概率

    Returns:
        数据块 (samples, channels)
    """
    n_samples = int(chunk_duration * fs)
    data = 0.2 * np.random.randn(n_samples, n_channels)

    # 随机生成事件
    if np.random.random() < event_probability:
        event_channel = np.random.randint(10, n_channels - 10)
        event_start = np.random.randint(0, n_samples - 100)
        event_duration = min(100, n_samples - event_start)

        # 事件信号
        t_event = np.linspace(0, 0.1, event_duration)
        pulse = 3.0 * np.sin(2 * np.pi * 50 * t_event) * np.hanning(event_duration)

        # 添加到邻近通道
        for ch_offset in range(-5, 6):
            ch = event_channel + ch_offset
            if 0 <= ch < n_channels:
                amplitude = 1 - abs(ch_offset) / 6
                end = event_start + event_duration
                data[event_start:end, ch] += amplitude * pulse

    return data


def process_chunk(
    chunk: np.ndarray,
    fs: float,
    sigma: float = 3.0,
) -> dict:
    """处理单个数据块

    Args:
        chunk: 输入数据块
        fs: 采样频率
        sigma: 检测阈值 (标准差倍数)

    Returns:
        处理结果字典
    """
    frame = from_array(chunk, fs=fs)

    # 处理流程
    processed = frame.bandpass(low=10, high=200).envelope()

    # 事件检测 - threshold_detect 返回 numpy 数组
    event_mask = processed.threshold_detect(sigma=sigma)

    # 收集包络数据
    env_data = processed.collect()

    # 统计
    n_events = np.sum(event_mask) if event_mask.any() else 0
    max_amplitude = np.max(env_data)

    return {
        "envelope": env_data,
        "events": event_mask,
        "n_events": int(n_events),
        "max_amplitude": float(max_amplitude),
    }


def realtime_monitoring_demo(
    duration: float = 10.0,
    callback: Optional[Callable] = None,
) -> None:
    """实时监测演示

    Args:
        duration: 模拟监测时长 (s)
        callback: 可选的回调函数，用于处理每个数据块的结果
    """
    print("=== DASMatrix 实时监测示例 ===\n")

    fs = 1000
    n_channels = 64
    chunk_duration = 0.5

    n_chunks = int(duration / chunk_duration)
    all_events = []

    print("监测参数:")
    print(f"  采样率: {fs} Hz")
    print(f"  通道数: {n_channels}")
    print(f"  数据块时长: {chunk_duration} s")
    print(f"  总时长: {duration} s")
    print()

    apply_nature_style()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    plt.ion()

    for i in range(n_chunks):
        # 模拟接收数据
        chunk = simulate_das_stream(
            fs=fs, n_channels=n_channels, chunk_duration=chunk_duration
        )

        # 处理
        start_time = time.time()
        result = process_chunk(chunk, fs=fs)
        proc_time = (time.time() - start_time) * 1000

        # 记录事件并保存事件图
        if result["n_events"] > 0:
            all_events.append({"time": i * chunk_duration, "count": result["n_events"]})
            
            # 保存事件图
            event_fig, event_ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            im = event_ax.imshow(
                result["envelope"].T,
                aspect="auto",
                origin="lower",
                cmap="hot",
                extent=[0, chunk_duration, 0, n_channels],
            )
            event_ax.set_title(
                f"事件检测 - 块 {i + 1} @ {i * chunk_duration:.1f}s\n"
                f"事件数: {result['n_events']}, 最大幅值: {result['max_amplitude']:.3f}"
            )
            event_ax.set_xlabel("Time (s)")
            event_ax.set_ylabel("Channel")
            event_fig.colorbar(im, ax=event_ax, label="Amplitude")
            
            event_path = f"output/events/event_block_{i + 1:03d}.png"
            event_fig.savefig(event_path, dpi=150)
            plt.close(event_fig)

        # 回调
        if callback:
            callback(i, result)

        # 更新可视化
        axes[0].clear()
        axes[0].imshow(
            result["envelope"].T,
            aspect="auto",
            origin="lower",
            cmap="hot",
            extent=[0, chunk_duration, 0, n_channels],
        )
        axes[0].set_title(
            f"实时包络图 - 块 {i + 1}/{n_chunks} (处理耗时: {proc_time:.1f} ms)"
        )
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Channel")

        axes[1].clear()
        if all_events:
            event_times = [e["time"] for e in all_events]
            event_counts = [e["count"] for e in all_events]
            axes[1].bar(event_times, event_counts, width=chunk_duration * 0.8)
        axes[1].set_title(f"累计事件: {len(all_events)}")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Event Count")
        axes[1].set_xlim(0, duration)

        plt.pause(0.1)

        # 状态输出
        status = "⚠️ 检测到事件!" if result["n_events"] > 0 else "✓ 正常"
        print(
            f"  块 {i + 1:3d}: 最大幅值={result['max_amplitude']:.3f}, "
            f"事件数={result['n_events']}, {status}"
        )

    plt.ioff()
    plt.savefig("output/realtime_monitoring.png", dpi=150)
    plt.close()

    print(f"\n监测完成！共检测到 {len(all_events)} 个事件块")
    print("结果已保存至 output/realtime_monitoring.png")


if __name__ == "__main__":
    import os

    os.makedirs("output", exist_ok=True)
    os.makedirs("output/events", exist_ok=True)
    realtime_monitoring_demo(duration=5.0)
