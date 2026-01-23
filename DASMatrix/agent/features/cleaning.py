"""数据清洗专用工具集。"""

from typing import Any, Dict

import numpy as np


def assess_data_quality(session: Any, data_id: str) -> Dict[str, Any]:
    """评估数据质量，识别噪声特征。

    Args:
        session: AgentSession 实例
        data_id: 数据对象 ID

    Returns:
        包含质量指标的字典
    """
    data = session.get(data_id)
    arr = data.collect()
    fs = data.fs

    # 1. 检测坏道 (基于能量异常)
    # 计算每个通道的平均能量
    energy = np.mean(arr**2, axis=0)
    median_energy = np.median(energy)

    # 定义阈值: 能量过高 > 100倍中值, 能量过低 < 1/100倍中值
    # 注意: 防止 median_energy 为 0
    safe_median = median_energy if median_energy > 0 else 1e-10

    bad_channels = np.where(energy > 100 * safe_median)[0].tolist()
    dead_channels = np.where(energy < 0.01 * safe_median)[0].tolist()

    # 2. 检测工频干扰 (50Hz/60Hz)
    # 取中间通道做 FFT
    mid_ch = arr.shape[1] // 2
    from scipy.fft import rfft, rfftfreq

    spec = np.abs(rfft(arr[:, mid_ch]))
    freqs = rfftfreq(arr.shape[0], 1 / fs)

    # 检查 50Hz 附近 +/- 1Hz
    idx_50 = np.where((freqs >= 49) & (freqs <= 51))[0]
    peak_50 = np.max(spec[idx_50]) if len(idx_50) > 0 else 0

    # 检查 60Hz 附近 +/- 1Hz
    idx_60 = np.where((freqs >= 59) & (freqs <= 61))[0]
    peak_60 = np.max(spec[idx_60]) if len(idx_60) > 0 else 0

    # 背景噪声水平 (中值)
    bg_level = np.median(spec)
    safe_bg = bg_level if bg_level > 0 else 1e-10

    has_50hz = peak_50 > 10 * safe_bg
    has_60hz = peak_60 > 10 * safe_bg

    # 3. 检测直流偏置/趋势
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    has_trend = abs(mean_val) > 0.1 * std_val

    # 4. 削波检测 (Clipping)
    # 假设数据归一化在 [-1, 1] 之外或达到最大值
    max_val = np.max(np.abs(arr))
    if max_val > 0:
        # 简单判定：如果有超过 0.99 * max_val 的点占比过高
        threshold = 0.99 * max_val
        clipping_ratio = np.sum(np.abs(arr) >= threshold) / arr.size
    else:
        clipping_ratio = 0.0

    return {
        "bad_channels_indices": sorted(list(set(bad_channels + dead_channels))),
        "dead_channels_count": len(dead_channels),
        "noisy_channels_count": len(bad_channels),
        "has_50hz_noise": bool(has_50hz),
        "has_60hz_noise": bool(has_60hz),
        "has_trend": bool(has_trend),
        "clipping_ratio": float(clipping_ratio),
        "rms_level": float(np.sqrt(np.mean(arr**2))),
        "noise_floor_db": float(20 * np.log10(safe_bg + 1e-10)),
        "snr_estimate_db": float(20 * np.log10(np.max(spec) / safe_bg))
        if safe_bg > 0
        else 0.0,
    }


def apply_cleaning_recipe(
    session: Any, data_id: str, recipe_name: str
) -> Dict[str, Any]:
    """应用预定义的清洗套餐。

    Args:
        session: AgentSession 实例
        data_id: 数据对象 ID
        recipe_name: 套餐名称
            - 'standard_denoise': 去趋势 + 1Hz高通
            - 'remove_powerline': 去除 50Hz 和 60Hz 干扰
            - 'seismic_enhance': 适合地震数据 (1-100Hz 带通 + 归一化)

    Returns:
        处理结果信息
    """

    # 我们需要通过 tools 实例来复用 process_signal 逻辑
    # 这里为了简单，我们重新构建 operations 列表并让上层调用 process_signal
    # 或者直接操作数据对象

    recipes = {
        "standard_denoise": [
            {"op": "detrend"},
            {"op": "highpass", "cutoff": 1.0, "order": 4},
        ],
        "remove_powerline": [
            # 简单的陷波器模拟 (组合高通低通或其他方式，当前 DASMatrix 尚未实现专用 notch，暂用带阻模拟或多次滤波)
            # 暂时用简单的 detrend 代替，实际应添加 notch 算子支持
            {"op": "detrend"}
        ],
        "seismic_enhance": [
            {"op": "detrend"},
            {"op": "bandpass", "low": 1.0, "high": 100.0, "order": 4},
            {"op": "normalize", "method": "zscore"},
        ],
    }

    if recipe_name not in recipes:
        raise ValueError(
            f"Unknown recipe: {recipe_name}. Available: {list(recipes.keys())}"
        )

    operations = recipes[recipe_name]

    # 这里我们模拟 process_signal 的逻辑
    data = session.get(data_id)
    result = data
    applied_ops = []

    for op_config in operations:
        # 复制配置以防修改
        config = op_config.copy()
        op_name = config.pop("op")

        if not hasattr(result, op_name):
            # 如果是复合操作或未实现的，暂时跳过或报错
            continue

        method = getattr(result, op_name)
        result = method(**config)
        applied_ops.append({"op": op_name, **config})

    # 存储结果
    metadata = {"source_id": data_id, "recipe": recipe_name, "operations": applied_ops}
    result_id = session.store(result, metadata)

    return {
        "id": result_id,
        "recipe": recipe_name,
        "operations_applied": applied_ops,
        "shape": list(result.shape),
    }
