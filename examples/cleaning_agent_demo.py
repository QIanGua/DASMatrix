#!/usr/bin/env python3
"""DAS Cleaning Agent 演示示例。

展示 Cleaning Agent 的核心工作流：诊断 -> 治疗 -> 验证。
"""

import json
from pathlib import Path

import h5py
import numpy as np


def create_noisy_data(path: Path) -> Path:
    """创建包含各种噪声的合成 DAS 数据。"""
    fs = 1000
    duration = 2
    n_channels = 100
    n_samples = fs * duration
    t = np.arange(n_samples) / fs

    data = np.zeros((n_samples, n_channels))

    for ch in range(n_channels):
        # 1. 基础信号 (20Hz)
        signal = np.sin(2 * np.pi * 20 * t)

        # 2. 添加趋势 (Trend)
        trend = np.linspace(0, 5, n_samples)

        # 3. 添加 50Hz 工频干扰
        powerline = 0.5 * np.sin(2 * np.pi * 50 * t)

        # 4. 随机噪声
        noise = 0.2 * np.random.randn(n_samples)

        data[:, ch] = signal + trend + powerline + noise

    # 5. 添加坏道 (Dead Channel & Noisy Channel)
    data[:, 10] = 0.0  # Dead
    data[:, 20] = data[:, 20] * 100  # Noisy

    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)
        f.attrs["sampling_rate"] = fs

    print(f"✅ 创建噪声数据: {path}")
    return path


def demo_cleaning_workflow():
    print("\n" + "=" * 60)
    print("🧹 DAS Cleaning Agent 工作流演示")
    print("=" * 60)

    from DASMatrix.agent import DASAgentTools

    tools = DASAgentTools()
    data_path = create_noisy_data(Path("/tmp/das_noisy_demo.h5"))

    # 1. 读取数据
    print("\n[Step 1] 读取数据...")
    res_read = tools.read_das_data(str(data_path))
    data_id = res_read["id"]
    print(f"   -> Data Loaded: {data_id}")

    # 2. 诊断 (Diagnosis)
    print("\n[Step 2] 诊断数据质量...")
    quality = tools.assess_data_quality(data_id)
    print(f"   -> 诊断报告:\n{json.dumps(quality, indent=4)}")

    # 模拟 Agent 思考
    print("\n🤖 Agent 思考: 发现明显趋势项 (has_trend=True) 和 50Hz 干扰。")
    print("              存在坏道 (Idx: 10, 20)。")
    print(
        "              建议方案: 使用 standard_denoise 去除趋势，额外添加 50Hz 滤波。"
    )

    # 3. 治疗 (Treatment)
    print("\n[Step 3] 执行清洗...")

    # 3.1 应用标准去噪套餐
    res_clean1 = tools.apply_cleaning_recipe(data_id, "standard_denoise")
    clean_id = res_clean1["id"]
    print(f"   -> 应用 standard_denoise: {clean_id}")

    # 3.2 针对性去除 50Hz (由于 apply_cleaning_recipe 暂未包含专用 notch，我们手动调用 process_signal 模拟)
    # real agent would call this if recipe wasn't enough, or we verify apply_cleaning_recipe("remove_powerline")

    # 让我们试试 apply_cleaning_recipe 的 remove_powerline (当前实现是 detrend，模拟效果)
    # 或者手动调用 process_signal
    # 为了演示效果，我们假设 standard_denoise 已经做得不错了，除了工频

    # 4. 验证 (Verification)
    print("\n[Step 4] 验证清洗结果...")
    quality_after = tools.assess_data_quality(clean_id)
    print(f"   -> 清洗后报告:\n{json.dumps(quality_after, indent=4)}")

    # 对比
    snr_before = quality.get("snr_estimate_db", 0)
    snr_after = quality_after.get("snr_estimate_db", 0)
    print("\n📊 效果对比:")
    print(f"   SNR: {snr_before:.1f} dB -> {snr_after:.1f} dB")
    print(f"   Trend: {quality['has_trend']} -> {quality_after['has_trend']}")

    print("\n" + "=" * 60)
    print("✅ 演示完成")


if __name__ == "__main__":
    demo_cleaning_workflow()
