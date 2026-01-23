"""DASMatrix 单位系统模块。

基于 Pint 库提供物理单位支持，避免单位混淆错误。

Example:
    >>> from DASMatrix.units import m, Hz, get_quantity
    >>> distance = 100 * m
    >>> freq = get_quantity("10 Hz")
    >>> print(freq.to("kHz"))
    0.01 kHz
"""

from typing import Union

import pint

# 创建全局单位注册表
ureg = pint.UnitRegistry()
UnitRegistry = pint.UnitRegistry
Quantity = ureg.Quantity

# === 常用长度单位 ===
m: pint.Unit = ureg.meter
km: pint.Unit = ureg.kilometer
mm: pint.Unit = ureg.millimeter
um: pint.Unit = ureg.micrometer
nm: pint.Unit = ureg.nanometer
ft: pint.Unit = ureg.foot

# === 常用时间单位 ===
s: pint.Unit = ureg.second
ms: pint.Unit = ureg.millisecond
us: pint.Unit = ureg.microsecond
ns: pint.Unit = ureg.nanosecond
minute: pint.Unit = ureg.minute
hour: pint.Unit = ureg.hour

# === 常用频率单位 ===
Hz: pint.Unit = ureg.hertz
kHz: pint.Unit = ureg.kilohertz
MHz: pint.Unit = ureg.megahertz

# === DAS 特定单位 ===
# 应变率 (strain rate)
strain: pint.Unit = ureg.dimensionless
strain_rate: pint.Quantity = ureg.dimensionless / ureg.second

# === 速度单位 ===
m_per_s: pint.Quantity = ureg.meter / ureg.second
km_per_s: pint.Quantity = ureg.kilometer / ureg.second


def get_quantity(expr: str) -> pint.Quantity:
    """解析字符串表达式为带单位的数值。

    Args:
        expr: 单位表达式字符串，如 "10 Hz", "100 m", "5 km/s"

    Returns:
        pint.Quantity: 带单位的数值对象

    Example:
        >>> get_quantity("10 Hz")
        <Quantity(10, 'hertz')>
        >>> get_quantity("100 m")
        <Quantity(100, 'meter')>
        >>> get_quantity("5 km/s")
        <Quantity(5, 'kilometer / second')>
    """
    return ureg(expr)


def get_unit(name: str) -> pint.Unit:
    """获取单位对象。

    Args:
        name: 单位名称，如 "meter", "hertz", "second"

    Returns:
        pint.Unit: 单位对象

    Example:
        >>> get_unit("meter")
        <Unit('meter')>
        >>> get_unit("Hz")
        <Unit('hertz')>
    """
    return getattr(ureg, name)


def to_base_units(quantity: pint.Quantity) -> pint.Quantity:
    """将数值转换为基本单位。

    Args:
        quantity: 带单位的数值

    Returns:
        pint.Quantity: 转换为 SI 基本单位后的数值

    Example:
        >>> to_base_units(1 * km)
        <Quantity(1000, 'meter')>
    """
    return quantity.to_base_units()


def magnitude(quantity: Union[pint.Quantity, float]) -> float:
    """获取数值的量值（去除单位）。

    Args:
        quantity: 带单位的数值或普通浮点数

    Returns:
        float: 数值的量值

    Example:
        >>> magnitude(10 * Hz)
        10.0
    """
    if isinstance(quantity, pint.Quantity):
        return float(quantity.magnitude)
    return float(quantity)


__all__ = [
    # 注册表
    "ureg",
    "Quantity",
    # 长度单位
    "m",
    "km",
    "mm",
    "um",
    "nm",
    "ft",
    # 时间单位
    "s",
    "ms",
    "us",
    "ns",
    "minute",
    "hour",
    # 频率单位
    "Hz",
    "kHz",
    "MHz",
    # DAS 特定
    "strain",
    "strain_rate",
    # 速度单位
    "m_per_s",
    "km_per_s",
    # 函数
    "get_quantity",
    "get_unit",
    "to_base_units",
    "magnitude",
    "UnitRegistry",
]
