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
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def ReadRawData(self, file_path: Path, target_col: Optional[List[int]] = None) -> Union[np.ndarray, "da.Array"]:
        """读取原始数据

        Args:
            file_path: 数据文件路径
            target_col: 目标列索引列表

        Returns:
            Union[np.ndarray, da.Array]: 读取的原始数据
        """
        pass

    def ValidateFile(self, file_path: Path) -> None:
        """验证文件是否存在且可读"""
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"路径不是文件: {file_path}")

        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"无法读取文件: {file_path}")


class DATReader(DataReader):
    """DAT 格式数据读取器 (向后兼容)"""

    def ReadRawData(self, file_path: Path, target_col: Optional[List[int]] = None) -> Union[np.ndarray, "da.Array"]:
        """读取 DAT 格式的原始数据"""
        self.ValidateFile(file_path)

        file_info = file_path.stat()
        byte_count = file_info.st_size
        row_count = byte_count // (self.sampling_config.channels * 2)

        if row_count <= 0:
            raise ValueError(f"文件大小异常, 无法读取有效数据: {file_path}")

        dtype = ">i2" if self.sampling_config.byte_order == "big" else "<i2"

        try:
            mmap = np.memmap(
                str(file_path),
                dtype=dtype,
                mode="r",
                shape=(int(row_count), int(self.sampling_config.channels)),
            )

            data = da.from_array(mmap, chunks="auto")
            data = (data * np.pi) / 2**13

            if target_col is not None:
                data = data[:, target_col]

            return data

        except Exception as e:
            self.logger.error(f"读取 DAT 文件时发生错误: {e}")
            raise IOError(f"无法读取 DAT 文件: {file_path}") from e


class H5Reader(DataReader):
    """HDF5 格式数据读取器 (向后兼容)"""

    def ReadRawData(self, file_path: Path, target_col: Optional[List[int]] = None) -> Union[np.ndarray, "da.Array"]:
        """读取 HDF5 格式的原始数据"""
        self.ValidateFile(file_path)

        try:
            f = h5py.File(file_path, "r")
            if "Acquisition/Raw[0]" not in f:
                f.close()
                raise KeyError("HDF5 文件中未找到 'Acquisition/Raw[0]' 数据集")

            dataset = f["Acquisition/Raw[0]"]
            data = da.from_array(dataset, chunks="auto")
            data = (data / 4) * (np.pi / 2**13)

            if target_col is not None:
                data = data[:, target_col]

            return data

        except Exception as e:
            self.logger.error(f"读取 H5 文件时发生错误: {e}")
            raise IOError(f"无法读取 H5 文件: {file_path}") from e


class SEGYReader(DataReader):
    """SEGY 格式数据读取器 (向后兼容)"""

    def ReadRawData(self, file_path: Path, target_col: Optional[List[int]] = None) -> Union[np.ndarray, "da.Array"]:
        """读取 SEGY 格式的原始数据"""
        self.ValidateFile(file_path)

        try:
            st = obspy.read(str(file_path), format="SEGY")
            data = np.stack([tr.data for tr in st], axis=1)

            if target_col is not None:
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
            st = obspy.read(str(file_path), format="MSEED")
            st.merge(method=1, fill_value="interpolate")

            lens = [len(tr) for tr in st]
            if len(set(lens)) > 1:
                min_len = min(lens)
                self.logger.warning(f"Trace 长度不一致, 裁剪到 {min_len}")
                data = np.stack([tr.data[:min_len] for tr in st], axis=1)
            else:
                data = np.stack([tr.data for tr in st], axis=1)

            if target_col is not None:
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
