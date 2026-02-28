"""DASMatrix 时间工具函数。

提供时间类型转换的便捷函数，简化 numpy datetime64 操作。

Example:
    >>> import DASMatrix as dm
    >>> t = dm.to_datetime64('2024-01-01T12:00:00')
    >>> delta = dm.to_timedelta64(3600)  # 1小时
"""

from datetime import datetime, timedelta
from typing import Literal, Optional, Union

import numpy as np

# 支持的时间单位
TimeUnit = Literal["s", "ms", "us", "ns", "m", "h", "D"]


def to_datetime64(
    value: Union[str, int, float, datetime, np.datetime64, None],
    unit: str = "ns",
) -> np.datetime64:
    """将各种输入转换为 numpy datetime64。

    Args:
        value: 输入值，支持:
            - str: ISO 8601 格式字符串，如 '2024-01-01T12:00:00'
            - int/float: Unix 时间戳（秒，从 1970-01-01 起）
            - datetime: Python datetime 对象
            - np.datetime64: 直接返回
            - None: 返回 NaT
        unit: 时间戳的单位（当 value 为数值时），默认 'ns'

    Returns:
        np.datetime64: 转换后的 datetime64 对象

    Example:
        >>> to_datetime64('2024-01-01T12:00:00')
        numpy.datetime64('2024-01-01T12:00:00')
        >>> to_datetime64(0)  # Unix epoch
        numpy.datetime64('1970-01-01T00:00:00')
        >>> to_datetime64(1704067200)  # 2024-01-01 00:00:00 UTC
        numpy.datetime64('2024-01-01T00:00:00')
    """
    if value is None:
        return np.datetime64("NaT")

    if isinstance(value, np.datetime64):
        return value

    if isinstance(value, str):
        return np.datetime64(value)

    if isinstance(value, datetime):
        return np.datetime64(value)

    if isinstance(value, (int, float)):
        # 解释为 Unix 时间戳（秒）
        # 转换为纳秒级整数
        ns_value = int(value * 1e9)
        return np.datetime64(ns_value, "ns")

    raise TypeError(
        f"Cannot convert {type(value).__name__} to datetime64. Expected str, int, float, datetime, or np.datetime64."
    )


def to_timedelta64(
    value: Union[str, int, float, timedelta, np.timedelta64, None],
    unit: TimeUnit = "s",
) -> np.timedelta64:
    """将各种输入转换为 numpy timedelta64。

    Args:
        value: 输入值，支持:
            - str: 带单位的字符串，如 '10s', '1h', '500ms'
            - int/float: 数值，单位由 unit 参数指定
            - timedelta: Python timedelta 对象
            - np.timedelta64: 直接返回
            - None: 返回 NaT
        unit: 当 value 为数值时的时间单位，默认 's'（秒）
            支持: 's', 'ms', 'us', 'ns', 'm', 'h', 'D'

    Returns:
        np.timedelta64: 转换后的 timedelta64 对象

    Example:
        >>> to_timedelta64(3600)  # 3600秒 = 1小时
        numpy.timedelta64(3600, 's')
        >>> to_timedelta64(1.5, 'h')  # 1.5小时
        numpy.timedelta64(5400000000000, 'ns')
        >>> to_timedelta64('10s')
        numpy.timedelta64(10, 's')
    """
    if value is None:
        return np.timedelta64("NaT")

    if isinstance(value, np.timedelta64):
        return value

    if isinstance(value, timedelta):
        # 转换为纳秒
        ns_value = int(value.total_seconds() * 1e9)
        return np.timedelta64(ns_value, "ns")

    if isinstance(value, str):
        return _parse_timedelta_string(value)

    if isinstance(value, (int, float)):
        # 根据单位创建 timedelta64
        return _create_timedelta(value, unit)

    raise TypeError(
        f"Cannot convert {type(value).__name__} to timedelta64. Expected str, int, float, timedelta, or np.timedelta64."
    )


def to_float(
    value: Union[np.datetime64, np.timedelta64],
    unit: TimeUnit = "s",
    reference: Optional[np.datetime64] = None,
) -> float:
    """将 datetime64 或 timedelta64 转换为浮点数。

    Args:
        value: datetime64 或 timedelta64 值
        unit: 输出单位，默认 's'（秒）
        reference: 当 value 为 datetime64 时的参考时间点
            如果为 None，使用 Unix epoch (1970-01-01)

    Returns:
        float: 转换后的浮点数

    Example:
        >>> delta = np.timedelta64(1500, 'ms')
        >>> to_float(delta)
        1.5
        >>> to_float(delta, 'ms')
        1500.0
    """
    unit_factors = {
        "ns": 1e-9,
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1.0,
        "m": 60.0,
        "h": 3600.0,
        "D": 86400.0,
    }

    if unit not in unit_factors:
        raise ValueError(f"Unknown unit: {unit}. Expected one of {list(unit_factors)}")

    if isinstance(value, np.datetime64):
        # 转换为相对于参考点的时间差
        if reference is None:
            reference = np.datetime64(0, "ns")
        value = value - reference

    if isinstance(value, np.timedelta64):
        # 转换为纳秒，再转换为指定单位
        ns_value = value / np.timedelta64(1, "ns")
        seconds = float(ns_value) * 1e-9
        return seconds / unit_factors[unit] * unit_factors["s"]

    raise TypeError(f"Cannot convert {type(value).__name__} to float. Expected np.datetime64 or np.timedelta64.")


def _parse_timedelta_string(s: str) -> np.timedelta64:
    """解析时间间隔字符串。

    支持格式: '10s', '1.5h', '500ms', '1D' 等
    """
    s = s.strip()

    # 单位映射
    unit_map = {
        "ns": "ns",
        "us": "us",
        "μs": "us",
        "ms": "ms",
        "s": "s",
        "sec": "s",
        "m": "m",
        "min": "m",
        "h": "h",
        "hr": "h",
        "hour": "h",
        "D": "D",
        "d": "D",
        "day": "D",
    }

    # 尝试解析
    for suffix, unit in sorted(unit_map.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            value_str = s[: -len(suffix)].strip()
            try:
                value = float(value_str)
                return _create_timedelta(value, unit)
            except ValueError:
                continue

    # 尝试直接解析为 numpy 格式
    try:
        return np.timedelta64(s)
    except ValueError as e:
        raise ValueError(f"Cannot parse '{s}' as timedelta. Expected format like '10s', '1.5h', '500ms'.") from e


def _create_timedelta(value: float, unit: str) -> np.timedelta64:
    """从数值和单位创建 timedelta64。"""
    # 对于整数值，直接使用 numpy 的单位
    if value == int(value):
        return np.timedelta64(int(value), unit)

    # 对于小数，转换为纳秒以保持精度
    unit_to_ns = {
        "ns": 1,
        "us": 1_000,
        "ms": 1_000_000,
        "s": 1_000_000_000,
        "m": 60_000_000_000,
        "h": 3_600_000_000_000,
        "D": 86_400_000_000_000,
    }

    if unit not in unit_to_ns:
        raise ValueError(f"Unknown unit: {unit}")

    ns_value = int(value * unit_to_ns[unit])
    return np.timedelta64(ns_value, "ns")


__all__ = [
    "to_datetime64",
    "to_timedelta64",
    "to_float",
    "TimeUnit",
]
