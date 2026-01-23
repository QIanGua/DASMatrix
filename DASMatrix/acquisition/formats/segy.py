"""SEG-Y 格式插件

SEG-Y 是地震勘探行业的标准数据交换格式。
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr

from .base import FormatMetadata, FormatPlugin

logger = logging.getLogger(__name__)


class SEGYFormatPlugin(FormatPlugin):
    """SEG-Y 格式插件

    使用 ObsPy 读取 SEG-Y 格式文件。

    Attributes:
        format_name: "SEGY"
        file_extensions: (".segy", ".sgy", ".seg")
        priority: 50 (默认优先级)
    """

    format_name = "SEGY"
    version = "1.0.0"
    file_extensions = (".segy", ".sgy", ".seg")
    priority = 50

    def __init__(self, sampling_rate: float = 5000.0):
        """初始化 SEG-Y 格式插件

        Args:
            sampling_rate: 默认采样率 (Hz), 如果文件中未指定
        """
        self.default_sampling_rate = sampling_rate

    def can_read(self, path: Path) -> bool:
        """检测文件是否为 SEG-Y 格式"""
        if not path.exists():
            return False

        # 检查扩展名
        if path.suffix.lower() in self.file_extensions:
            return True

        # 可以添加 SEG-Y 魔数检测 (文本头部等)
        return False

    def scan(self, path: Path) -> FormatMetadata:
        """快速扫描 SEG-Y 文件元数据"""
        try:
            import obspy

            # headonly=True 只读取头部
            st = obspy.read(str(path), format="SEGY", headonly=True)

            n_traces = len(st)
            n_samples = st[0].stats.npts if n_traces > 0 else 0
            if n_traces > 0:
                fs = st[0].stats.sampling_rate
            else:
                fs = self.default_sampling_rate

            return FormatMetadata(
                n_samples=n_samples,
                n_channels=n_traces,
                sampling_rate=fs,
                file_size=path.stat().st_size,
                format_name=self.format_name,
                format_version=self.version,
            )
        except Exception as e:
            logger.warning(f"扫描 SEG-Y 文件失败: {e}")
            # 返回基本信息
            return FormatMetadata(
                n_samples=0,
                n_channels=0,
                sampling_rate=self.default_sampling_rate,
                file_size=path.stat().st_size,
                format_name=self.format_name,
            )

    def read(
        self,
        path: Path,
        channels: Optional[list[int]] = None,
        time_slice: Optional[tuple[int, int]] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> Union[xr.DataArray, da.Array, np.ndarray]:
        """读取 SEG-Y 格式数据

        Args:
            path: 文件路径
            channels: 要读取的通道索引列表
            time_slice: 时间切片 (start, end)
            lazy: 是否延迟加载 (SEG-Y 暂不支持真正的延迟加载)

        Returns:
            xr.DataArray: 带标签的数据数组

        Note:
            SEG-Y 格式目前使用 ObsPy 读取, 暂不支持真正的延迟加载。
            数据会立即加载到内存中。
        """
        import obspy

        st = obspy.read(str(path), format="SEGY")

        # 获取采样率
        fs = st[0].stats.sampling_rate if len(st) > 0 else self.default_sampling_rate

        # 转换为 numpy 数组 (n_samples, n_channels)
        data = np.stack([tr.data for tr in st], axis=1)

        # 应用通道选择
        n_ch = data.shape[1]
        if channels is not None:
            data = data[:, channels]
            channel_coord = list(channels)
        else:
            channel_coord = list(range(n_ch))

        # 应用时间切片
        if time_slice is not None:
            start, end = time_slice
            data = data[start:end]
            time_coord = np.arange(end - start) / fs
        else:
            time_coord = np.arange(data.shape[0]) / fs

        # 包装为 dask 数组 (如果需要延迟计算)
        if lazy:
            data = da.from_array(data, chunks="auto")

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
                "source_file": str(path),
            },
        )
