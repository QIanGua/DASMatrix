"""DASMatrix 数据获取模块

提供多种 DAS 数据格式的读取功能。

推荐使用新的便捷函数:
    >>> from DASMatrix.acquisition import read, scan, detect_format
    >>> data = read("data.h5")

或使用 FormatRegistry:
    >>> from DASMatrix.acquisition.formats import FormatRegistry
    >>> data = FormatRegistry.read("data.h5")

旧的 DASReader API 保持向后兼容:
    >>> from DASMatrix.acquisition import DASReader, DataType
    >>> reader = DASReader(config, DataType.H5)
"""

# 新的便捷函数 (推荐使用)
# 向后兼容的旧 API
from .das_reader import (
    DASReader,
    DataReader,
    DataType,
    DATReader,
    H5Reader,
    MiniSEEDReader,
    SEGYReader,
    detect_format,
    list_formats,
    read,
    scan,
)

__all__ = [
    # 新 API
    "read",
    "scan",
    "detect_format",
    "list_formats",
    # 旧 API (向后兼容)
    "DASReader",
    "DataType",
    "DataReader",
    "DATReader",
    "H5Reader",
    "SEGYReader",
    "MiniSEEDReader",
]
