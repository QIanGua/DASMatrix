"""DAS 数据读取器

该模块提供统一的数据读取接口, 支持多种 DAS 文件格式。

推荐使用新的 FormatRegistry API:
    >>> from DASMatrix.acquisition.formats import FormatRegistry
    >>> data = FormatRegistry.read("data.h5")

旧的 DASReader API 保持向后兼容:
    >>> reader = DASReader(config, DataType.H5)
    >>> data = reader.ReadRawData(file_path)
"""

import logging
import os
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Union

import dask.array as da
import h5py
import numpy as np
import obspy
import xarray as xr

from ..config.sampling_config import SamplingConfig

logger = logging.getLogger(__name__)


class DataType(Enum):
    """数据类型枚举类 (向后兼容)"""

    DAT = auto()
    H5 = auto()
    SEGY = auto()
    MINISEED = auto()


class DataReader(ABC):
    """抽象数据读取器基类 (向后兼容)"""

    def __init__(self, sampling_config: SamplingConfig):
        """初始化数据读取器

        Args:
            sampling_config: 采样配置对象, 包含采样频率、通道数等信息
        """
        self.sampling_config = sampling_config
        self.channels = sampling_config.channels
        self.fs = sampling_config.fs
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def ReadRawData(self, file_path: Path, target_col: Optional[List[int]] = None) -> Union[np.ndarray, "da.Array"]:
        """读取原始数据"""
        pass

    def ValidateFile(self, file_path: Path) -> None:
        """验证文件是否存在"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")


class DATReader(DataReader):
    """DAT 格式数据读取器 (向后兼容)"""

    def ReadRawData(self, file_path: Path, target_col: Optional[List[int]] = None) -> Union[np.ndarray, "da.Array"]:
        """读取 DAT 格式的原始数据 (使用内存映射以支持大文件)"""
        self.ValidateFile(file_path)

        try:
            # 计算总样本数
            file_size = os.path.getsize(file_path)
            item_size = np.dtype(np.float32).itemsize
            total_items = file_size // item_size
            n_rows = total_items // self.channels

            data = np.memmap(file_path, dtype=np.float32, mode="r", shape=(n_rows, self.channels))

            if target_col:
                data = data[:, target_col]

            return data
        except Exception as e:
            self.logger.error(f"读取 DAT 文件失败: {e}")
            raise


class H5Reader(DataReader):
    """HDF5 格式数据读取器 (向后兼容)"""

    def ReadRawData(self, file_path: Path, target_col: Optional[List[int]] = None) -> Union[np.ndarray, "da.Array"]:
        """读取 HDF5 格式的原始数据"""
        self.ValidateFile(file_path)

        try:
            with h5py.File(file_path, "r") as f:
                # 假设数据在 "Data" 或 "DAS" 组下
                data_key = "Data" if "Data" in f else "DAS"
                if data_key not in f:
                    # 尝试查找第一个数据集
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            data_key = key
                            break

                obj = f[data_key]
                if isinstance(obj, h5py.Dataset):
                    if target_col:
                        data = obj[:, target_col]
                    else:
                        data = obj[:]
                    return data
                else:
                    raise TypeError(f"Expected Dataset at {data_key}, found {type(obj)}")
        except Exception as e:
            self.logger.error(f"读取 H5 文件时发生错误: {e}")
            raise IOError(f"无法读取 H5 文件: {file_path}") from e


class SEGYReader(DataReader):
    """SEGY 格式数据读取器 (向后兼容)"""

    def ReadRawData(self, file_path: Path, target_col: Optional[List[int]] = None) -> Union[np.ndarray, "da.Array"]:
        """读取 SEGY 格式的原始数据"""
        self.ValidateFile(file_path)

        try:
            # 使用 obspy 读取 SEGY
            st = obspy.read(str(file_path), format="SEGY")
            data = np.stack([tr.data for tr in st], axis=1)
            if target_col:
                data = data[:, target_col]
            return data
        except Exception as e:
            self.logger.error(f"读取 SEGY 文件时发生错误: {e}")
            raise IOError(f"无法读取 SEGY 文件: {file_path}") from e


class MiniSEEDReader(DataReader):
    """MiniSEED 格式数据读取器 (向后兼容)"""

    def ReadRawData(self, file_path: Path, target_col: Optional[List[int]] = None) -> Union[np.ndarray, "da.Array"]:
        """读取 MiniSEED 格式的原始数据"""
        self.ValidateFile(file_path)

        try:
            st = obspy.read(str(file_path))
            data = np.stack([tr.data for tr in st], axis=1)
            if target_col:
                data = data[:, target_col]
            return data
        except Exception as e:
            self.logger.error(f"读取 MiniSEED 文件时发生错误: {e}")
            raise IOError(f"无法读取 MiniSEED 文件: {file_path}") from e


class DASReader:
    """数据读取器类 (向后兼容)

    推荐使用新的 FormatRegistry API:
        >>> from DASMatrix.acquisition.formats import FormatRegistry
        >>> data = FormatRegistry.read("data.h5")

    旧的 DASReader API 保持可用:
        >>> reader = DASReader(config, DataType.H5)
        >>> data = reader.ReadRawData(file_path)
    """

    def __init__(self, sampling_config: SamplingConfig, data_type: DataType = DataType.DAT):
        """初始化数据读取器

        Args:
            sampling_config: 采样配置对象
            data_type: 数据类型, 默认为 DAT 格式
        """
        self.sampling_config = sampling_config
        self.data_type = data_type
        self.logger = logging.getLogger(__name__)

        # 根据数据类型选择合适的读取器
        reader_map = {
            DataType.DAT: DATReader,
            DataType.H5: H5Reader,
            DataType.SEGY: SEGYReader,
            DataType.MINISEED: MiniSEEDReader,
        }

        reader_class = reader_map.get(data_type)
        if reader_class is None:
            raise ValueError(f"不支持的数据类型: {data_type}")

        self.reader: DataReader = reader_class(sampling_config)

    def ReadRawData(self, file_path: Path, target_col: Optional[List[int]] = None) -> Union[np.ndarray, "da.Array"]:
        """读取指定路径的数据文件

        Args:
            file_path: 数据文件路径
            target_col: 目标列索引列表

        Returns:
            np.ndarray | da.Array: 读取的原始数据
        """
        try:
            return self.reader.ReadRawData(file_path, target_col)
        except Exception as e:
            self.logger.error(f"读取数据时发生错误: {e}, 文件路径: {file_path}")
            raise


# ============================================================================
# 新的便捷函数 (推荐使用)
# ============================================================================


def read(
    path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs,
) -> Union[xr.DataArray, "da.Array", np.ndarray]:
    """统一读取函数

    自动检测文件格式并读取数据。这是推荐的读取方式。

    Args:
        path: 文件路径
        format: 强制指定格式, None 表示自动检测
        **kwargs: 传递给格式插件的参数

    Returns:
        da.Array: 读取的数据 (延迟加载)

    Example:
        >>> from DASMatrix.acquisition import read
        >>> data = read("data.h5")
        >>> data = read("data.dat", n_channels=800, sampling_rate=5000)
    """
    from .formats import FormatRegistry

    return FormatRegistry.read(path, format_name=format, **kwargs)


def scan(
    path: Union[str, Path],
    format: Optional[str] = None,
):
    """扫描文件元数据

    快速获取文件元数据, 不加载数据。

    Args:
        path: 文件路径
        format: 强制指定格式, None 表示自动检测

    Returns:
        FormatMetadata: 文件元数据

    Example:
        >>> from DASMatrix.acquisition import scan
        >>> meta = scan("data.h5")
        >>> print(f"采样率: {meta.sampling_rate} Hz")
    """
    from .formats import FormatRegistry

    return FormatRegistry.scan(path, format_name=format)


def detect_format(path: Union[str, Path]) -> Optional[str]:
    """检测文件格式

    Args:
        path: 文件路径

    Returns:
        str | None: 格式名称, 未知格式返回 None

    Example:
        >>> from DASMatrix.acquisition import detect_format
        >>> fmt = detect_format("data.h5")
        >>> print(f"格式: {fmt}")  # 输出: 格式: H5
    """
    from .formats import FormatRegistry

    return FormatRegistry.detect_format(path)


def list_formats() -> list[str]:
    """列出所有支持的格式

    Returns:
        list[str]: 格式名称列表

    Example:
        >>> from DASMatrix.acquisition import list_formats
        >>> print(list_formats())  # ['DAT', 'H5', 'SEGY', 'MINISEED']
    """
    from .formats import FormatRegistry

    return FormatRegistry.list_formats()
