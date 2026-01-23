"""函数式API入口，提供便捷的DASFrame创建方法。"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .dasframe import DASFrame

import h5py
import numpy as np

from ..acquisition.das_reader import DASReader, DataType
from ..config.sampling_config import SamplingConfig


# 避免循环导入
def _create_dasframe(data: Any, fs: float, **kwargs: Any) -> "DASFrame":
    """创建 DASFrame 对象（延迟导入避免循环依赖）"""
    from .dasframe import DASFrame

    return DASFrame(data, fs=fs, **kwargs)


def read(
    path: str,
    fs: Optional[float] = None,
    channels: Optional[int] = None,
    byte_order: str = "little",
    **kwargs: Any,
) -> "DASFrame":
    """从文件读取DAS数据，返回DASFrame对象。

    Args:
        path: 数据文件路径
        fs: 采样频率(Hz)，H5文件可自动检测，DAT文件必须指定
        channels: 通道数，H5文件可自动检测，DAT文件必须指定
        byte_order: 字节序，'big' 或 'little'，仅用于 DAT 文件
        **kwargs: 传递给 DASFrame 的其他元数据

    Returns:
        DASFrame: 包含DAS数据的DASFrame对象

    Examples:
        >>> # 读取 H5 文件（自动检测参数）
        >>> frame = df.read("data.h5")

        >>> # 读取 DAT 文件（需要指定参数）
        >>> frame = df.read("data.dat", fs=10000, channels=512)
    """
    from ..acquisition.formats import FormatRegistry

    # 使用 FormatRegistry 统一读取
    # 这将返回带有 attrs['inventory'] 的 xr.DataArray
    daa = FormatRegistry.read(path, fs=fs, **kwargs)

    # 提取关键参数
    attrs = getattr(daa, "attrs", {})
    fs_val = attrs.get("fs", fs)
    if fs_val is None:
        # Fallback if not found
        fs_val = 1.0

    return _create_dasframe(daa, fs=fs_val, **attrs)


def _read_h5(path: str, fs: Optional[float] = None, **kwargs):
    """从HDF5文件读取DAS数据。

    自动从 H5 文件中提取采样频率和通道数。
    """
    file_path = Path(path)

    with h5py.File(file_path, "r") as f:
        # 尝试自动检测采样频率
        detected_fs = fs
        if detected_fs is None:
            # 常见的 H5 DAS 文件采样率存储位置
            fs_paths = [
                "Acquisition/Raw[0]/RawDataTime",
                "Acquisition/SamplingFrequency",
                "metadata/fs",
                "SamplingFrequency",
            ]
            for fs_path in fs_paths:
                if fs_path in f:
                    try:
                        # 尝试读取采样率属性或数据
                        fs_data = f[fs_path]
                        if hasattr(fs_data, "attrs") and "SamplingFrequency" in fs_data.attrs:
                            detected_fs = float(fs_data.attrs["SamplingFrequency"])
                            break
                        elif isinstance(fs_data, h5py.Dataset):
                            # 从时间数据推算采样率
                            if len(fs_data) > 1:
                                dt = fs_data[1] - fs_data[0]
                                if dt > 0:
                                    detected_fs = 1.0 / dt
                                    break
                    except (KeyError, TypeError, IndexError):
                        continue

            # 尝试从根属性读取
            if detected_fs is None and "SamplingFrequency" in f.attrs:
                detected_fs = float(f.attrs["SamplingFrequency"])

        # 如果仍未检测到，使用默认值
        if detected_fs is None:
            detected_fs = 10000.0  # 默认 10kHz

        # 检测数据路径和通道数
        data_paths = [
            "Acquisition/Raw[0]/RawData",
            "Acquisition/Raw[0]",
            "data",
            "Data",
        ]

        raw_data = None
        for data_path in data_paths:
            if data_path in f:
                raw_data = f[data_path][:]
                break

        if raw_data is None:
            raise KeyError(f"无法在 H5 文件中找到数据。尝试过的路径: {data_paths}")

        # 确保数据形状正确 (time, channels)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(-1, 1)

        # 转换为物理量（如果需要）
        if raw_data.dtype in (np.int16, np.int32):
            raw_data = (raw_data.astype(np.float64) * np.pi) / 2**13

    return _create_dasframe(raw_data, fs=detected_fs, source_file=str(path), **kwargs)


def _read_dat(
    path: str,
    fs: float,
    channels: int,
    byte_order: str = "little",
    **kwargs,
):
    """从DAT文件读取DAS数据。"""
    config = SamplingConfig(fs=fs, channels=channels, byte_order=byte_order)
    reader = DASReader(config, data_type=DataType.DAT)
    raw_data = reader.ReadRawData(Path(path))

    return _create_dasframe(raw_data, fs=fs, source_file=str(path), **kwargs)


def from_array(data: np.ndarray, fs: float, **kwargs: Any) -> "DASFrame":
    """从NumPy数组创建DASFrame对象。

    Args:
        data: DAS数据数组，形状为(时间点数, 通道数)
        fs: 采样频率(Hz)
        **kwargs: 其他元数据

    Returns:
        DASFrame: 包含数据的DASFrame对象

    Examples:
        >>> import numpy as np
        >>> from DASMatrix.api import df
        >>> data = np.random.randn(10000, 64)
        >>> frame = df.from_array(data, fs=10000)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return _create_dasframe(data, fs=fs, **kwargs)


def stream(url: str, chunk: int = 1024, fs: float = 10000.0, **kwargs: Any) -> "DASFrame":
    """创建用于流式处理的DASFrame。

    Args:
        url: 数据流URL，例如 "tcp://0.0.0.0:9000"
        chunk: 每次处理的数据块大小
        fs: 预期的采样频率(Hz)
        **kwargs: 其他参数

    Returns:
        DASFrame: 用于流处理的DASFrame对象

    Note:
        当前为占位实现，流处理功能将在后续版本中完善。
    """
    # TODO: 实现真正的流处理
    return _create_dasframe(
        np.array([]).reshape(0, 1),
        fs=fs,
        is_stream=True,
        stream_url=url,
        chunk_size=chunk,
        **kwargs,
    )
