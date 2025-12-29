"""TDMS 格式插件

TDMS (Technical Data Management Streaming) 是 National Instruments 开发的格式。
常用于 LabVIEW 采集系统输出的 DAS 数据。

需要安装 nptdms 库: pip install nptdms
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr

from .base import FormatMetadata, FormatPlugin

logger = logging.getLogger(__name__)


class TDMSFormatPlugin(FormatPlugin):
    """TDMS 格式插件

    读取 National Instruments TDMS 格式文件。
    需要安装 nptdms 库。

    Attributes:
        format_name: "TDMS"
        file_extensions: (".tdms",)
        priority: 45
    """

    format_name = "TDMS"
    version = "1.0.0"
    file_extensions = (".tdms",)
    priority = 45

    def __init__(self, sampling_rate: float = 5000.0):
        """初始化 TDMS 格式插件

        Args:
            sampling_rate: 默认采样率 (Hz)
        """
        self.default_sampling_rate = sampling_rate

    def _check_nptdms(self) -> bool:
        """检查 nptdms 是否可用"""
        try:
            import nptdms  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False

    def can_read(self, path: Path) -> bool:
        """检测文件是否为 TDMS 格式"""
        if not path.exists():
            return False

        if path.suffix.lower() != ".tdms":
            return False

        # 检查 TDMS 魔数
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
                return magic == b"TDSm"
        except Exception:
            return False

    def scan(self, path: Path) -> FormatMetadata:
        """快速扫描 TDMS 文件元数据"""
        if not self._check_nptdms():
            raise ImportError("需要安装 nptdms: pip install nptdms")

        from nptdms import TdmsFile  # type: ignore

        with TdmsFile.open(path) as f:
            # 获取第一个组的第一个通道
            groups = f.groups()
            if not groups:
                raise ValueError(f"TDMS 文件中没有数据组: {path}")

            group = groups[0]
            channels = group.channels()
            if not channels:
                raise ValueError(f"TDMS 文件中没有通道: {path}")

            n_channels = len(channels)
            n_samples = len(channels[0]) if n_channels > 0 else 0

            # 尝试获取采样率
            fs = self.default_sampling_rate
            for ch in channels:
                if hasattr(ch, "properties"):
                    if "wf_increment" in ch.properties:
                        dt = ch.properties["wf_increment"]
                        if dt > 0:
                            fs = 1.0 / dt
                            break
                    if "SamplingFrequency" in ch.properties:
                        fs = ch.properties["SamplingFrequency"]
                        break

            return FormatMetadata(
                n_samples=n_samples,
                n_channels=n_channels,
                sampling_rate=fs,
                file_size=path.stat().st_size,
                format_name=self.format_name,
                format_version=self.version,
            )

    def read(
        self,
        path: Path,
        channels: Optional[list[int]] = None,
        time_slice: Optional[tuple[int, int]] = None,
        lazy: bool = True,
        group_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[xr.DataArray, da.Array, np.ndarray]:
        """读取 TDMS 格式数据

        Args:
            path: 文件路径
            channels: 要读取的通道索引列表
            time_slice: 时间切片 (start, end)
            lazy: 是否延迟加载
            group_name: TDMS 组名, None 表示使用第一个组
        """
        if not self._check_nptdms():
            raise ImportError("需要安装 nptdms: pip install nptdms")

        from nptdms import TdmsFile  # type: ignore

        f = TdmsFile.open(path)

        try:
            groups = f.groups()
            if not groups:
                raise ValueError(f"TDMS 文件中没有数据组: {path}")

            if group_name:
                group = f[group_name]
            else:
                group = groups[0]

            all_channels = group.channels()
            if not all_channels:
                raise ValueError(f"TDMS 文件中没有通道: {path}")

            # 获取采样率
            fs = self.default_sampling_rate
            for ch in all_channels:
                if hasattr(ch, "properties"):
                    if "wf_increment" in ch.properties:
                        dt = ch.properties["wf_increment"]
                        if dt > 0:
                            fs = 1.0 / dt
                            break

            # 读取数据
            if channels is not None:
                selected_channels = [all_channels[i] for i in channels]
            else:
                selected_channels = all_channels

            # 堆叠为 2D 数组 (n_samples, n_channels)
            data_list = [np.array(ch[:]) for ch in selected_channels]
            data = np.column_stack(data_list)

            n_samples, n_ch = data.shape

            # 应用时间切片
            if time_slice is not None:
                start, end = time_slice
                data = data[start:end]
                time_coord = np.arange(end - start) / fs
            else:
                time_coord = np.arange(n_samples) / fs

            if channels is not None:
                channel_coord = list(channels)
            else:
                channel_coord = list(range(n_ch))

            if lazy:
                data = da.from_array(data, chunks="auto")

            return xr.DataArray(
                data,
                dims=["time", "channel"],
                coords={"time": time_coord, "channel": channel_coord},
                attrs={
                    "sampling_rate": fs,
                    "format": self.format_name,
                    "source_file": str(path),
                    "group_name": group.name,
                },
            )

        finally:
            f.close()
