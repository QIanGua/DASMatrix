"""DAT 格式插件

DAT 格式是一种简单的二进制格式, 通常由 DAS 询问器直接输出。
数据按行存储, 每行包含所有通道的采样值。
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr

from .base import FormatMetadata, FormatPlugin

logger = logging.getLogger(__name__)


class DATFormatPlugin(FormatPlugin):
    """DAT 格式插件

    支持常见的二进制 DAT 文件格式, 数据按 int16 存储。

    Attributes:
        format_name: "DAT"
        file_extensions: (".dat", ".bin")
        priority: 40 (较高优先级)
    """

    format_name = "DAT"
    version = "1.0.0"
    file_extensions = (".dat", ".bin")
    priority = 40

    def __init__(
        self,
        channels: int = 800,
        sampling_rate: float = 5000.0,
        byte_order: str = "little",
    ):
        """初始化 DAT 格式插件

        Args:
            channels: 默认通道数
            sampling_rate: 默认采样率 (Hz)
            byte_order: 字节序 ("little" 或 "big")
        """
        self.default_channels = channels
        self.default_sampling_rate = sampling_rate
        self.byte_order = byte_order

    def can_read(self, path: Path) -> bool:
        """检测文件是否为 DAT 格式"""
        if not path.exists():
            return False

        # 检查扩展名
        if path.suffix.lower() in self.file_extensions:
            return True

        # 可以添加魔数检测, 但 DAT 通常没有固定的魔数
        return False

    def scan(self, path: Path) -> FormatMetadata:
        """快速扫描 DAT 文件元数据"""
        file_size = path.stat().st_size
        bytes_per_sample = 2  # int16

        # 计算样本数
        row_bytes = self.default_channels * bytes_per_sample
        n_samples = file_size // row_bytes

        return FormatMetadata(
            n_samples=n_samples,
            n_channels=self.default_channels,
            sampling_rate=self.default_sampling_rate,
            file_size=file_size,
            format_name=self.format_name,
            format_version=self.version,
            data_unit="strain_rate",
        )

    def read(
        self,
        path: Path,
        channels: Optional[list[int]] = None,
        time_slice: Optional[tuple[int, int]] = None,
        lazy: bool = True,
        n_channels: Optional[int] = None,
        sampling_rate: Optional[float] = None,
        byte_order: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[xr.DataArray, da.Array, np.ndarray]:
        """读取 DAT 格式数据

        Args:
            path: 文件路径
            channels: 要读取的通道索引列表
            time_slice: 时间切片 (start, end)
            lazy: 是否延迟加载
            n_channels: 通道数 (覆盖默认值)
            sampling_rate: 采样率 (覆盖默认值)
            byte_order: 字节序 (覆盖默认值)

        Returns:
            xr.DataArray: 带标签的数据数组
        """
        n_ch = n_channels or self.default_channels
        fs = sampling_rate or self.default_sampling_rate
        bo = byte_order or self.byte_order

        # 确定字节序和数据类型
        dtype = ">i2" if bo == "big" else "<i2"

        # 计算行数
        file_size = path.stat().st_size
        bytes_per_row = n_ch * 2
        n_samples = file_size // bytes_per_row

        if n_samples <= 0:
            raise ValueError(f"文件大小异常, 无法读取有效数据: {path}")

        # 使用内存映射
        mmap = np.memmap(
            str(path),
            dtype=dtype,
            mode="r",
            shape=(n_samples, n_ch),
        )

        if lazy:
            # 包装为 dask 数组
            data = da.from_array(mmap, chunks="auto")
        else:
            # 直接读取为 numpy 数组
            data = np.array(mmap)

        # 转换为物理量 (应变率)
        data = (data * np.pi) / (2**13)

        # 应用通道选择
        if channels is not None:
            data = data[:, channels]
            n_ch = len(channels)

        # 应用时间切片
        if time_slice is not None:
            start, end = time_slice
            data = data[start:end]
            n_samples = end - start

        # 创建坐标
        time_coord = np.arange(data.shape[0]) / fs
        channel_coord = list(range(data.shape[1])) if channels is None else list(channels)

        # 返回 xr.DataArray
        return xr.DataArray(
            data,
            dims=["time", "channel"],
            coords={
                "time": time_coord,
                "channel": channel_coord,
            },
            attrs={
                "sampling_rate": fs,
                "format": self.format_name,
                "units": "rad/s",  # 应变率单位
                "source_file": str(path),
            },
        )
