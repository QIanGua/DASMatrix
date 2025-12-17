import logging
import os
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

# 从DASMatrix.config导入配置类
from ..config.sampling_config import SamplingConfig


class DataType(Enum):
    """数据类型枚举类"""

    DAT = auto()  # DAT文件格式
    H5 = auto()  # HDF5文件格式


class DataReader(ABC):
    """抽象数据读取器基类"""

    def __init__(self, sampling_config: SamplingConfig):
        """初始化数据读取器

        Args:
            sampling_config: 采样配置对象，包含采样频率、通道数等信息
        """
        self.sampling_config = sampling_config
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def ReadRawData(
        self, file_path: Path, target_col: Optional[list] = None
    ) -> np.ndarray:
        """读取原始数据

        Args:
            file_path: 数据文件路径
            target_col: 目标列索引列表，如 [21,22,249,251]，指定要处理的非连续列

        Returns:
            np.ndarray: 读取的原始数据
        """
        pass

    def ValidateFile(self, file_path: Path) -> None:
        """验证文件是否存在且可读

        Args:
            file_path: 数据文件路径

        Raises:
            FileNotFoundError: 文件不存在
            PermissionError: 无法读取文件
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
            self.logger.debug(f"将文件路径转换为 Path 对象: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"路径不是文件: {file_path}")

        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"无法读取文件: {file_path}")


class DATReader(DataReader):
    """DAT格式数据读取器"""

    def ReadRawData(
        self, file_path: Path, target_col: Optional[list] = None
    ) -> np.ndarray:
        """读取DAT格式的原始数据

        Args:
            file_path: 数据文件路径
            target_col: 目标列索引列表，如 [21,22,249,251]，指定要处理的非连续列

        Returns:
            np.ndarray: 读取的原始数据
        """
        self.ValidateFile(file_path)

        # 计算行数
        file_info = file_path.stat()
        byte_count = file_info.st_size
        row_count = byte_count // (self.sampling_config.channels * 2)

        if row_count <= 0:
            raise ValueError(f"文件大小异常，无法读取有效数据: {file_path}")

        # 根据byte_order指定字节序
        dtype = ">i2" if self.sampling_config.byte_order == "big" else "<i2"

        # 直接读取并重塑数据
        try:
            with open(file_path, "rb") as file:
                raw_data = np.fromfile(
                    file, dtype=dtype, count=row_count * self.sampling_config.channels
                ).reshape((row_count, self.sampling_config.channels))

            # 转换为物理量
            raw_data = (raw_data * np.pi) / 2**13  # type: ignore

            # 如果提供了target_col，只选择指定列
            if target_col is not None:
                raw_data = raw_data[:, target_col]  # type: ignore

            return raw_data
        except Exception as e:
            self.logger.error(f"读取DAT文件时发生错误: {e}")
            raise IOError(f"无法读取DAT文件: {file_path}") from e


class H5Reader(DataReader):
    """HDF5格式数据读取器"""

    def ReadRawData(
        self, file_path: Path, target_col: Optional[list] = None
    ) -> np.ndarray:
        """读取HDF5格式的原始数据

        Args:
            file_path: 数据文件路径
            target_col: 目标列索引列表，如 [21,22,249,251]，指定要处理的非连续列

        Returns:
            np.ndarray: 读取的原始数据
        """
        self.ValidateFile(file_path)

        try:
            with h5py.File(file_path, "r") as f:
                if "Acquisition/Raw[0]" not in f:
                    raise KeyError("HDF5文件中未找到 'Acquisition/Raw[0]' 数据集")

                raw_data = f["Acquisition/Raw[0]"][:] / 4

            # 转换为物理量
            raw_data = (raw_data * np.pi) / 2**13  # type: ignore

            # 如果提供了target_col，只选择指定列
            if target_col is not None:
                raw_data = raw_data[:, target_col]

            return raw_data
        except Exception as e:
            self.logger.error(f"读取H5文件时发生错误: {e}")
            raise IOError(f"无法读取H5文件: {file_path}") from e


class DASReader:
    """数据读取器类, 用于读取不同格式的数据文件"""

    def __init__(
        self, sampling_config: SamplingConfig, data_type: DataType = DataType.DAT
    ):
        """初始化数据读取器

        Args:
            sampling_config: 采样配置对象
            data_type: 数据类型，默认为DAT格式
        """
        self.sampling_config = sampling_config
        self.reader: DataReader
        self.data_type = data_type
        self.logger = logging.getLogger(__name__)

        # 根据数据类型选择合适的读取器
        if data_type == DataType.DAT:
            self.reader = DATReader(sampling_config)
        elif data_type == DataType.H5:
            self.reader = H5Reader(sampling_config)
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")

    def ReadRawData(
        self, file_path: Path, target_col: Optional[list] = None
    ) -> np.ndarray:
        """读取指定路径的数据文件

        Args:
            file_path: 数据文件路径
            target_col: 目标列索引列表

        Returns:
            np.ndarray: 包含原始数据的 NumPy 数组
        """
        try:
            # 读取原始数据
            raw_data = self.reader.ReadRawData(file_path, target_col)
            return raw_data
        except Exception as e:
            self.logger.error(f"读取数据时发生错误: {e}，文件路径: {file_path}")
            raise
