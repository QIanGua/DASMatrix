#!/usr/bin/env python3
"""Update DASMatrix_Tutorial.ipynb to reflect the latest API."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "nb" / "DASMatrix_Tutorial.ipynb"

def create_updated_notebook():
    """Create an updated tutorial notebook with the latest API examples."""
    
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    def add_markdown(source):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source if isinstance(source, list) else [source]
        })
    
    def add_code(source, outputs=None):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": outputs or [],
            "source": source if isinstance(source, list) else [source]
        })
    
    # Title
    add_markdown([
        "# DASMatrix 完整教程\n",
        "\n",
        "**分布式声学传感（DAS）数据处理与分析框架**\n",
        "\n",
        "本教程将全面介绍 DASMatrix 的核心功能，包括：\n",
        "\n",
        "1. 数据创建与加载\n",
        "2. **链式信号处理 API**（新特性）\n",
        "3. 频域分析（FFT/STFT）\n",
        "4. 统计分析与事件检测\n",
        "5. Nature/Science 级别可视化\n",
        "\n",
        "---"
    ])
    
    # Section 1: Environment Setup
    add_markdown(["## 1. 环境准备"])
    add_code([
        "# 导入必要的库\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "# 导入 DASMatrix 模块\n",
        "from DASMatrix.api import df\n",
        "from DASMatrix.api.dasframe import DASFrame\n",
        "from DASMatrix.visualization import apply_nature_style\n",
        "\n",
        "# 应用 Nature 风格（全局）\n",
        "apply_nature_style()\n",
        "\n",
        "print(\"DASMatrix 加载成功！\")"
    ])
    
    # Section 2: Create Synthetic Data
    add_markdown([
        "## 2. 创建合成 DAS 数据\n",
        "\n",
        "在没有真实 DAS 数据的情况下，我们创建一些模拟数据用于演示。"
    ])
    add_code([
        "def create_synthetic_das_data(\n",
        "    duration=2.0,  # 持续时间（秒）\n",
        "    fs=1000,  # 采样率（Hz）\n",
        "    n_channels=64,  # 通道数\n",
        "    signal_freq=50,  # 主信号频率（Hz）\n",
        "    noise_level=0.2,  # 噪声水平\n",
        "):\n",
        "    \"\"\"创建模拟 DAS 数据\"\"\"\n",
        "    t = np.linspace(0, duration, int(duration * fs))\n",
        "    n_samples = len(t)\n",
        "    data = np.zeros((n_samples, n_channels))\n",
        "\n",
        "    for ch in range(n_channels):\n",
        "        # 基础正弦波（带相位延迟模拟波传播）\n",
        "        phase_delay = ch * 0.02\n",
        "        signal = np.sin(2 * np.pi * signal_freq * (t - phase_delay))\n",
        "\n",
        "        # 添加谐波\n",
        "        signal += 0.3 * np.sin(2 * np.pi * 2 * signal_freq * (t - phase_delay))\n",
        "\n",
        "        # 模拟事件（如地震或入侵）\n",
        "        if 20 < ch < 45:\n",
        "            event_time = 0.8 + ch * 0.01\n",
        "            event = 3.0 * np.exp(-((t - event_time) ** 2) / 0.005)\n",
        "            signal += event\n",
        "\n",
        "        # 添加噪声\n",
        "        noise = noise_level * np.random.randn(n_samples)\n",
        "        data[:, ch] = signal + noise\n",
        "\n",
        "    return data\n",
        "\n",
        "# 创建数据\n",
        "raw_data = create_synthetic_das_data()\n",
        "print(f\"数据形状: {raw_data.shape} (样本数 × 通道数)\")\n",
        "print(\"采样率: 1000 Hz\")\n",
        "print(\"持续时间: 2.0 秒\")"
    ])
    
    # Section 3: Create DASFrame
    add_markdown([
        "## 3. 创建 DASFrame 对象\n",
        "\n",
        "DASFrame 是 DASMatrix 的核心数据结构，提供**链式 API** 进行信号处理。\n",
        "\n",
        "它基于 Xarray 和 Dask 实现，支持延迟计算以处理大规模数据。"
    ])
    add_code([
        "# 使用 df.from_array 创建 DASFrame（推荐方式）\n",
        "frame = df.from_array(raw_data, fs=1000, dx=1.0)\n",
        "\n",
        "print(\"DASFrame 创建成功！\")\n",
        "print(f\"  - 数据形状: {frame.shape}\")\n",
        "print(f\"  - 采样率: {frame.fs} Hz\")"
    ])
    
    # Section 4: Basic Visualization
    add_markdown([
        "## 4. 基础可视化\n",
        "\n",
        "DASFrame 提供内置的 Nature/Science 级别可视化方法。\n",
        "\n",
        "`plot_heatmap` 方法支持以下参数：\n",
        "- `channels`: 通道切片，如 `slice(10, 50)`\n",
        "- `t_range`: 时间样本切片，如 `slice(0, 1000)`\n",
        "- `title`, `cmap`: 标题和颜色映射"
    ])
    add_code([
        "# 绘制原始数据热图\n",
        "fig = frame.plot_heatmap(\n",
        "    title=\"原始 DAS 数据\",\n",
        "    cmap=\"RdBu_r\"\n",
        ")\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ])
    
    # Section 5: Chainable Signal Processing (NEW!)
    add_markdown([
        "## 5. 链式信号处理（核心特性）\n",
        "\n",
        "DASMatrix 的核心优势是**链式 API**，允许将多个处理步骤连接起来：\n",
        "\n",
        "```python\n",
        "processed = frame.detrend().bandpass(1, 100).normalize()\n",
        "```\n",
        "\n",
        "### 5.1 可用的链式方法\n",
        "\n",
        "| 类别 | 方法 |\n",
        "|------|------|\n",
        "| 预处理 | `detrend()`, `demean()`, `normalize()` |\n",
        "| 滤波器 | `bandpass()`, `lowpass()`, `highpass()`, `notch()` |\n",
        "| 变换 | `fft()`, `stft()`, `hilbert()`, `envelope()` |\n",
        "| F-K 滤波 | `fk_filter()` |"
    ])
    add_code([
        "# 链式信号处理示例\n",
        "# 去趋势 → 带通滤波 (1-100 Hz) → 归一化\n",
        "processed = frame.detrend().bandpass(1, 100).normalize()\n",
        "\n",
        "print(f\"处理后数据形状: {processed.shape}\")\n",
        "print(\"链式处理完成！\")"
    ])
    add_code([
        "# 可视化处理后的数据\n",
        "fig = processed.plot_heatmap(\n",
        "    title=\"处理后的 DAS 数据（去趋势 + 带通滤波 + 归一化）\",\n",
        "    cmap=\"seismic\"\n",
        ")\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ])
    
    # Section 5.2: Filter Examples
    add_markdown([
        "### 5.2 滤波器示例\n",
        "\n",
        "DASFrame 支持多种滤波器，均返回 DASFrame 以支持链式调用。"
    ])
    add_code([
        "# 低通滤波器\n",
        "lowpassed = frame.lowpass(cutoff=50, order=4)\n",
        "print(f\"低通滤波后形状: {lowpassed.shape}\")\n",
        "\n",
        "# 高通滤波器\n",
        "highpassed = frame.highpass(cutoff=5, order=4)\n",
        "print(f\"高通滤波后形状: {highpassed.shape}\")\n",
        "\n",
        "# 陷波滤波器（移除 50Hz 工频干扰）\n",
        "notched = frame.notch(freq=50, Q=30)\n",
        "print(f\"陷波滤波后形状: {notched.shape}\")"
    ])
    
    # Section 6: Frequency Domain Analysis
    add_markdown([
        "## 6. 频域分析\n",
        "\n",
        "DASFrame 提供 FFT 和 STFT 方法进行频域分析。\n",
        "\n",
        "**注意**：`fft()` 和 `stft()` 方法现在返回 `DASFrame` 对象，支持继续链式处理。"
    ])
    add_code([
        "# FFT 频谱分析\n",
        "spectrum = frame.fft()\n",
        "\n",
        "print(f\"频谱数据类型: {type(spectrum).__name__}\")\n",
        "print(f\"频谱形状: {spectrum.shape}\")\n",
        "\n",
        "# 获取 numpy 数据进行自定义绘图\n",
        "spectrum_data = spectrum.collect()\n",
        "\n",
        "# 绘制平均频谱\n",
        "plt.figure(figsize=(10, 4))\n",
        "freqs = np.fft.rfftfreq(frame.shape[0], 1/frame.fs)\n",
        "mean_spectrum = np.mean(spectrum_data, axis=1)\n",
        "plt.semilogy(freqs[:len(mean_spectrum)], mean_spectrum)\n",
        "plt.xlabel('Frequency (Hz)')\n",
        "plt.ylabel('Amplitude')\n",
        "plt.title('平均频谱')\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ])
    
    # Section 6.2: STFT
    add_markdown(["### 6.2 短时傅里叶变换 (STFT)"])
    add_code([
        "# STFT 时频分析（返回 DASFrame）\n",
        "stft_result = frame.stft(nperseg=256, noverlap=128)\n",
        "\n",
        "print(f\"STFT 结果类型: {type(stft_result).__name__}\")\n",
        "print(f\"STFT 结果形状: {stft_result.shape}\")"
    ])
    
    # Section 6.3: Envelope
    add_markdown(["### 6.3 希尔伯特变换与包络提取"])
    add_code([
        "# 提取信号包络\n",
        "env = frame.envelope()\n",
        "\n",
        "print(f\"包络形状: {env.shape}\")\n",
        "\n",
        "# 可视化包络\n",
        "fig = env.plot_heatmap(\n",
        "    title=\"信号包络\",\n",
        "    cmap=\"hot\"\n",
        ")\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ])
    
    # Section 7: Statistics
    add_markdown([
        "## 7. 统计分析\n",
        "\n",
        "DASFrame 提供多种统计方法：\n",
        "- `mean(axis)`: 均值\n",
        "- `std(axis)`: 标准差\n",
        "- `max(axis)` / `min(axis)`: 最值\n",
        "- `rms(window)`: 均方根"
    ])
    add_code([
        "# 统计分析示例\n",
        "print(\"统计分析：\")\n",
        "print(f\"  每通道均值形状: {frame.mean(axis=0).shape}\")\n",
        "print(f\"  每通道标准差形状: {frame.std(axis=0).shape}\")\n",
        "print(f\"  每通道最大值形状: {frame.max(axis=0).shape}\")\n",
        "print(f\"  每通道最小值形状: {frame.min(axis=0).shape}\")\n",
        "print(f\"  RMS 形状: {frame.rms().shape}\")"
    ])
    add_code([
        "# 绘制统计结果\n",
        "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n",
        "\n",
        "# 均值\n",
        "axes[0, 0].plot(frame.mean(axis=0).flatten())\n",
        "axes[0, 0].set_title('各通道均值')\n",
        "axes[0, 0].set_xlabel('Channel')\n",
        "axes[0, 0].set_ylabel('Mean')\n",
        "\n",
        "# 标准差\n",
        "axes[0, 1].plot(frame.std(axis=0).flatten())\n",
        "axes[0, 1].set_title('各通道标准差')\n",
        "axes[0, 1].set_xlabel('Channel')\n",
        "axes[0, 1].set_ylabel('Std')\n",
        "\n",
        "# RMS\n",
        "axes[1, 0].plot(frame.rms().flatten())\n",
        "axes[1, 0].set_title('各通道 RMS')\n",
        "axes[1, 0].set_xlabel('Channel')\n",
        "axes[1, 0].set_ylabel('RMS')\n",
        "\n",
        "# 最大值\n",
        "axes[1, 1].plot(frame.max(axis=0).flatten(), label='Max')\n",
        "axes[1, 1].plot(frame.min(axis=0).flatten(), label='Min')\n",
        "axes[1, 1].set_title('各通道最值')\n",
        "axes[1, 1].set_xlabel('Channel')\n",
        "axes[1, 1].set_ylabel('Value')\n",
        "axes[1, 1].legend()\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ])
    
    # Section 8: Event Detection
    add_markdown([
        "## 8. 事件检测\n",
        "\n",
        "`threshold_detect` 方法用于检测超过阈值的事件。\n",
        "\n",
        "参数：\n",
        "- `threshold`: 绝对阈值（可选）\n",
        "- `sigma`: 如果未指定阈值，使用 `mean + sigma * std` 计算"
    ])
    add_code([
        "# 阈值检测\n",
        "detections = frame.threshold_detect(sigma=3.0)\n",
        "\n",
        "print(f\"检测结果形状: {detections.shape}\")\n",
        "print(f\"检测到的事件总数: {np.sum(detections)}\")"
    ])
    add_code([
        "# 可视化检测结果\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# 原始数据\n",
        "frame.plot_heatmap(title=\"原始数据\", ax=axes[0])\n",
        "\n",
        "# 检测结果叠加\n",
        "from DASMatrix.api.dasframe import DASFrame as DF\n",
        "detection_frame = DF(detections.astype(float), fs=frame.fs)\n",
        "detection_frame.plot_heatmap(title=\"阈值检测结果\", ax=axes[1], cmap=\"binary\")\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ])
    
    # Section 9: F-K Filtering
    add_markdown([
        "## 9. F-K 滤波\n",
        "\n",
        "F-K 滤波用于根据表观速度滤除特定波场成分。\n",
        "\n",
        "参数：\n",
        "- `v_min` / `v_max`: 最小/最大表观速度\n",
        "- `dx`: 空间采样间隔"
    ])
    add_code([
        "# F-K 滤波示例\n",
        "fk_filtered = frame.fk_filter(v_min=100, v_max=2000, dx=1.0)\n",
        "\n",
        "print(f\"F-K 滤波后形状: {fk_filtered.shape}\")"
    ])
    add_code([
        "# 对比 F-K 滤波前后\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "frame.plot_heatmap(title=\"原始数据\", ax=axes[0])\n",
        "fk_filtered.plot_heatmap(title=\"F-K 滤波后\", ax=axes[1])\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ])
    
    # Section 10: Complete Pipeline
    add_markdown([
        "## 10. 完整处理流程示例\n",
        "\n",
        "下面展示一个典型的 DAS 数据处理流程。"
    ])
    add_code([
        "# 完整的链式处理流程\n",
        "result = (\n",
        "    frame\n",
        "    .detrend()              # 去趋势\n",
        "    .bandpass(5, 100)       # 带通滤波 5-100 Hz\n",
        "    .normalize()            # 归一化\n",
        ")\n",
        "\n",
        "print(\"完整处理流程完成！\")\n",
        "print(f\"最终形状: {result.shape}\")\n",
        "\n",
        "# 可视化最终结果\n",
        "fig = result.plot_heatmap(\n",
        "    title=\"完整处理流程结果\",\n",
        "    channels=slice(10, 55),  # 只显示 10-55 通道\n",
        "    cmap=\"RdBu_r\"\n",
        ")\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ])
    
    # Summary
    add_markdown([
        "## 总结\n",
        "\n",
        "本教程介绍了 DASMatrix 的核心功能：\n",
        "\n",
        "1. **DASFrame**：基于 Xarray/Dask 的核心数据结构\n",
        "2. **链式 API**：流式处理，代码简洁高效\n",
        "3. **信号处理**：丰富的滤波和预处理方法\n",
        "4. **频域分析**：FFT、STFT、希尔伯特变换\n",
        "5. **统计分析**：均值、标准差、RMS 等\n",
        "6. **事件检测**：阈值检测\n",
        "7. **F-K 滤波**：速度域滤波\n",
        "8. **可视化**：Nature/Science 级别图表\n",
        "\n",
        "更多信息请参阅 [DASMatrix 文档](https://github.com/your-repo/DASMatrix)。"
    ])
    
    return notebook


def main():
    """Main entry point."""
    notebook = create_updated_notebook()
    
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Notebook updated: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
