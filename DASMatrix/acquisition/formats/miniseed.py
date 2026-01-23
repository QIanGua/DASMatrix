"""MiniSEED 格式插件

MiniSEED 是地震学领域的标准数据交换格式, 是 SEED 格式的精简版。
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr

from .base import FormatMetadata, FormatPlugin

logger = logging.getLogger(__name__)


class MiniSEEDFormatPlugin(FormatPlugin):
    """MiniSEED 格式插件

    使用 ObsPy 读取 MiniSEED 格式文件。

    Attributes:
        format_name: "MINISEED"
        file_extensions: (".mseed", ".miniseed", ".ms")
        priority: 50 (默认优先级)
    """

    format_name = "MINISEED"
    version = "1.0.0"
    file_extensions = (".mseed", ".miniseed", ".ms")
    priority = 50

    def __init__(self, sampling_rate: float = 5000.0):
        """初始化 MiniSEED 格式插件

        Args:
            sampling_rate: 默认采样率 (Hz), 如果文件中未指定
        """
        self.default_sampling_rate = sampling_rate

    def can_read(self, path: Path) -> bool:
        """检测文件是否为 MiniSEED 格式"""
        if not path.exists():
            return False

        # 检查扩展名
        if path.suffix.lower() in self.file_extensions:
            return True

        # 检查 MiniSEED 魔数 (通过检查记录头)
        try:
            with open(path, "rb") as f:
                # MiniSEED 记录以固定的序列号开头
                header = f.read(20)
                # 检查是否包含有效的 SEED 时间戳
                return len(header) >= 20 and header[6:8] in [b"D ", b"R ", b"Q ", b"M "]
        except Exception:
            return False

    def scan(self, path: Path) -> FormatMetadata:
        """快速扫描 MiniSEED 文件元数据"""
        try:
            import obspy

            st = obspy.read(str(path), format="MSEED", headonly=True)

            n_traces = len(st)
            n_samples = st[0].stats.npts if n_traces > 0 else 0
            fs = (
                st[0].stats.sampling_rate
                if n_traces > 0
                else self.default_sampling_rate
            )

            # 获取时间信息
            start_time = str(st[0].stats.starttime) if n_traces > 0 else None

            return FormatMetadata(
                n_samples=n_samples,
                n_channels=n_traces,
                sampling_rate=fs,
                start_time=start_time,
                file_size=path.stat().st_size,
                format_name=self.format_name,
                format_version=self.version,
            )
        except Exception as e:
            logger.warning(f"扫描 MiniSEED 文件失败: {e}")
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
        merge_gaps: bool = True,
        **kwargs: Any,
    ) -> Union[xr.DataArray, da.Array, np.ndarray]:
        """读取 MiniSEED 格式数据

        Args:
            path: 文件路径
            channels: 要读取的通道索引列表
            time_slice: 时间切片 (start, end)
            lazy: 是否延迟加载 (MiniSEED 暂不支持真正的延迟加载)
            merge_gaps: 是否合并数据间隙

        Returns:
            xr.DataArray: 带标签的数据数组

        Note:
            MiniSEED 格式目前使用 ObsPy 读取, 暂不支持真正的延迟加载。
            数据会立即加载到内存中。
        """
        import obspy

        st = obspy.read(str(path), format="MSEED")

        # 合并数据间隙
        if merge_gaps:
            st.merge(method=1, fill_value="interpolate")

        # 获取采样率和时间信息
        fs = st[0].stats.sampling_rate if len(st) > 0 else self.default_sampling_rate
        start_time = st[0].stats.starttime if len(st) > 0 else None

        # 确保所有 trace 长度一致
        lens = [len(tr) for tr in st]
        if len(set(lens)) > 1:
            min_len = min(lens)
            logger.warning(f"Trace 长度不一致, 裁剪到 {min_len} 个采样点")
            data = np.stack([tr.data[:min_len] for tr in st], axis=1)
        else:
            data = np.stack([tr.data for tr in st], axis=1)

        # 应用通道选择
        n_ch = data.shape[1]
        if channels is not None:
            data = data[:, channels]
            channel_coord = channels
        else:
            channel_coord = np.arange(n_ch)

        # 应用时间切片
        if time_slice is not None:
            start, end = time_slice
            data = data[start:end]
            time_coord = np.arange(start, end) / fs
        else:
            time_coord = np.arange(data.shape[0]) / fs

        # 包装为 dask 数组 (如果需要延迟计算)
        if lazy:
            data = da.from_array(data, chunks="auto")

        # 构建属性
        attrs = {
            "sampling_rate": fs,
            "format": self.format_name,
            "source_file": str(path),
        }
        if start_time is not None:
            attrs["start_time"] = str(start_time)

        # 返回 xr.DataArray
        return xr.DataArray(
            data,
            dims=["time", "channel"],
            coords={
                "time": time_coord,
                "channel": channel_coord,
            },
            attrs=attrs,
        )
