"""
DASMatrix 快速入门示例

展示DASFrame API的基本用法，包括数据创建、处理和可视化。
"""

import matplotlib.pyplot as plt
import numpy as np

# 导入DASMatrix API
from DASMatrix import from_array


def create_synthetic_data():
    """创建合成测试数据，模拟DAS信号。"""
    # 参数
    fs = 1000  # 采样率 (Hz)
    duration = 5  # 持续时间 (s)
    num_channels = 100  # 通道数

    # 时间轴
    t = np.arange(0, duration, 1 / fs)

    # 创建基本信号: 各通道具有不同频率的正弦波
    data = np.zeros((len(t), num_channels))
    for ch in range(num_channels):
        # 基础频率从5Hz到100Hz线性变化
        freq = 5 + ch * 95 / num_channels
        data[:, ch] = np.sin(2 * np.pi * freq * t)

    # 添加一个事件: 在中心位置附近的通道产生一个高幅振动
    event_time = int(duration * fs / 2)  # 事件发生时间，在中间
    event_duration = int(0.2 * fs)  # 事件持续200毫秒
    event_center = num_channels // 2  # 事件中心通道
    event_width = 10  # 事件影响的通道范围

    # 生成脉冲信号
    pulse = np.sin(2 * np.pi * 50 * t[:event_duration]) * np.hanning(event_duration)

    # 将脉冲添加到数据中
    for ch in range(event_center - event_width, event_center + event_width):
        if 0 <= ch < num_channels:
            # 脉冲幅度随着与中心通道的距离减弱
            amplitude = 5 * (1 - abs(ch - event_center) / event_width)
            data[event_time : event_time + event_duration, ch] += amplitude * pulse

    # 添加噪声
    noise_level = 0.1
    data += noise_level * np.random.randn(*data.shape)

    return data, fs


def main():
    """主程序：演示DASFrame API的基本使用方法。"""
    print("创建合成DAS数据...")
    data, fs = create_synthetic_data()

    # 从数组创建DASFrame
    print("从数组创建DASFrame对象...")
    D = from_array(data, fs=fs)

    # 演示基本处理流程
    print("执行处理流程: 带通滤波 -> 希尔伯特包络 -> 空间平滑...")
    processed = (
        D.bandpass(low=20, high=100)  # 带通滤波
        .hilbert_env()  # 希尔伯特包络
        .spatial_smooth(kernel=3)
    )  # 空间平滑

    # 生成热图可视化
    print("生成热图...")
    fig = processed.plot_heatmap()
    plt.savefig("das_heatmap.png")

    # 事件检测
    print("执行事件检测...")
    events = processed.threshold_detect(db=-20)

    # 获取结果数据
    print("获取结果数据...")
    result_data = events.collect()

    print(f"结果形状: {result_data.shape}")
    print(f"检测到的事件数量: {np.sum(result_data)}")

    print("处理完成，热图已保存为 'das_heatmap.png'")


if __name__ == "__main__":
    main()
