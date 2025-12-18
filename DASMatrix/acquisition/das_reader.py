import logging
import os
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Union, List

import dask.array as da
import h5py
import numpy as np
import obspy
from ..config.sampling_config import SamplingConfig

# 从DASMatrix.config导入配置类
from ..config.sampling_config import SamplingConfig


class DataType(Enum):
    """数据类型枚举类"""

    DAT = auto()  # DAT文件格式
    H5 = auto()  # HDF5文件格式
    SEGY = auto()  # SEG-Y地震数据格式
    MINISEED = auto()  # MiniSEED地震数据格式


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
            Union[np.ndarray, da.Array]: 读取的原始数据
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

        # 如果是 lazy 模式 (chunks 不是 None)
        chunks = getattr(self.sampling_config, "chunks", None) # 假设 config 有 chunks
        # 或者增加参数 lazy=False
        
        # 统一策略：默认 lazy 如果文件很大？或者总是 lazy，用户可以 compute()
        # 为了兼容性，我们可以让 ReadRawData 返回 da.Array，它是 np.ndarray 的鸭子类型
        # 但是旧代码可能期待 pure numpy
        
        # 这里我们修改为：使用 memmap + dask
        try:
            # Memory map the file
            mmap_mode = "r"
            raw_data_mmap = np.memmap(
                file_path,
                dtype=dtype,
                mode=mmap_mode,
                shape=(row_count, self.sampling_config.channels)
            )
            
            # Wrap with dask array
            # Chunking strategy: 
            # If chunks not specified, default to auto or specific size
            # Typically DAS data is time-heavy. 
            # Let's chunk by time (e.g. 10000 samples) and full channels?
            # Or use 'auto'
            chunks = "auto"
            
            dask_data = da.from_array(raw_data_mmap, chunks=chunks)
            
            # Convert to physical units dealing with dask graph
            # This is lazy
            data = (dask_data * np.pi) / 2**13
            
            # Slicing columns (channels) is also lazy and efficient in dask
            if target_col is not None:
                data = data[:, target_col]
                
            # For backward compatibility, if caller expects numpy, they might be surprised.
            # But the task is "Refactor DASReader to support lazy loading".
            # We should probably return dask array. 
            # However, existing tests expect numpy shape immediately. Dask array has shape.
            # Operations will be lazy.
            
            # Wait, if we return dask array, existing code doing things like `data[0]` works.
            # But `type(data)` is `dask.array.core.Array`.
            # Let's return dask array.
            
            return data

        except Exception as e:
            self.logger.error(f"读取DAT文件时发生错误: {e}")
            raise IOError(f"无法读取DAT文件: {file_path}") from e
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
            # Open file in read mode. Note: We cannot use 'with' block here if we want to return lazy array
            # because closing file invalidates the h5 dataset.
            # Dask handles this if we don't close immediately? 
            # Actually standard practice for lazy h5 in typical scripts is tricky.
            # But xarray/dask usually keeps file open or reopens.
            # For dask.from_array(dataset), the file must remain open.
            
            # Compromise: We behave like xarray.open_dataset.
            # We attach the file object to the array or reader? 
            # Or we simply don't support context manager closing in strict sense inside this method.
            # Let's assume user manages lifecycle or we leave it to GC (h5py does this reasonably).
            
            f = h5py.File(file_path, "r")
            if "Acquisition/Raw[0]" not in f:
                f.close()
                raise KeyError("HDF5文件中未找到 'Acquisition/Raw[0]' 数据集")

            dataset = f["Acquisition/Raw[0]"]
            
            # Wrap with dask
            # chunks=None means use h5 chunking if available, else auto?
            # Safe to specify chunks="auto"
            dask_data = da.from_array(dataset, chunks="auto") 
            
            # Divide by 4 (lazy)
            raw_data = dask_data / 4

            # 转换为物理量 (lazy)
            raw_data = (raw_data * np.pi) / 2**13

            # 如果提供了target_col，只选择指定列 (lazy)
            if target_col is not None:
                raw_data = raw_data[:, target_col]

            return raw_data
        except Exception as e:
            self.logger.error(f"读取H5文件时发生错误: {e}")
            raise IOError(f"无法读取H5文件: {file_path}") from e


class SEGYReader(DataReader):
    """SEGY格式数据读取器"""

    def ReadRawData(
        self, file_path: Path, target_col: Optional[list] = None
    ) -> np.ndarray:
        """读取SEGY格式的原始数据

        Args:
            file_path: 数据文件路径
            target_col: 目标列索引列表

        Returns:
            np.ndarray: 读取的原始数据 (n_samples, n_channels)
        """
        self.ValidateFile(file_path)

        try:
            # 读取SEGY文件
            # unpack_trace_headers=True 可以读取头部信息，但会增加内存消耗
            # headonly=True 只读取头部用于校验，这里我们需要数据
            st = obspy.read(str(file_path), format="SEGY")
            
            # 将Stream转换为numpy数组
            # ObsPy的习惯是 (n_traces, n_samples)，我们需要转置为 (n_samples, n_channels)
            # stack=True 确保所有trace长度一致并堆叠成二维数组
            data = np.stack([tr.data for tr in st], axis=1)

            # 如果提供了target_col，只选择指定列
            if target_col is not None:
                data = data[:, target_col]

            return data
        except Exception as e:
            self.logger.error(f"读取SEGY文件时发生错误: {e}")
            raise IOError(f"无法读取SEGY文件: {file_path}") from e


class MiniSEEDReader(DataReader):
    """MiniSEED格式数据读取器"""

    def ReadRawData(
        self, file_path: Path, target_col: Optional[list] = None
    ) -> np.ndarray:
        """读取MiniSEED格式的原始数据

        Args:
            file_path: 数据文件路径
            target_col: 目标列索引列表

        Returns:
            np.ndarray: 读取的原始数据 (n_samples, n_channels)
        """
        self.ValidateFile(file_path)

        try:
            st = obspy.read(str(file_path), format="MSEED")
            
            # 确保所有trace对齐（MiniSEED可能有间隙）
            st.merge(method=1, fill_value='interpolate')
            
            # 转换为numpy数组 (n_channels, n_samples) -> (n_samples, n_channels)
            #由于MiniSEED不保证所有trace长度绝对一致（除非merge后trim），这里做个安全检查
            lens = [len(tr) for tr in st]
            if len(set(lens)) > 1:
                # 裁剪到最小长度
                min_len = min(lens)
                self.logger.warning(f"Trace lengths inconsistent, trimming to {min_len}")
                data = np.stack([tr.data[:min_len] for tr in st], axis=1)
            else:
                data = np.stack([tr.data for tr in st], axis=1)

            # 如果提供了target_col，只选择指定列
            if target_col is not None:
                data = data[:, target_col]

            return data
        except Exception as e:
            self.logger.error(f"读取MiniSEED文件时发生错误: {e}")
            raise IOError(f"无法读取MiniSEED文件: {file_path}") from e


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
        elif data_type == DataType.SEGY:
            self.reader = SEGYReader(sampling_config)
        elif data_type == DataType.MINISEED:
            self.reader = MiniSEEDReader(sampling_config)
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
