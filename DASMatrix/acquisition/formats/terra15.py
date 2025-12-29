"""Terra15 格式插件

Terra15 是澳大利亚 Terra15 公司的 DAS 询问器输出格式, 基于 HDF5。
数据存储在 /data_product/data 数据集中。

参考: https://github.com/DASDAE/dascore (MIT License)
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import dask.array as da
import h5py
import numpy as np
import xarray as xr

from .base import FormatMetadata, FormatPlugin

logger = logging.getLogger(__name__)


class Terra15FormatPlugin(FormatPlugin):
    """Terra15 格式插件

    Terra15 询问器使用 HDF5 格式存储数据, 数据集路径为:
    - /data_product/data: 主数据
    - /data_product/gps_time: GPS 时间戳
    - /_metadata: 元数据属性

    Attributes:
        format_name: "TERRA15"
        file_extensions: (".hdf5", ".h5")
        priority: 25 (高优先级, 在通用 H5 之前检测)
    """

    format_name = "TERRA15"
    version = "1.0.0"
    file_extensions = (".hdf5", ".h5")
    priority = 25  # 高于通用 H5 格式

    # Terra15 特有的数据集路径
    DATA_PATH = "data_product/data"
    GPS_TIME_PATH = "data_product/gps_time"

    def can_read(self, path: Path) -> bool:
        """检测文件是否为 Terra15 格式"""
        if not path.exists():
            return False

        if path.suffix.lower() not in self.file_extensions:
            return False

        # 检查 HDF5 内部结构
        try:
            with h5py.File(path, "r") as f:
                # Terra15 特有: data_product 组
                return self.DATA_PATH in f
        except Exception:
            return False

    def _read_attrs(self, f: h5py.File) -> dict:
        """读取 Terra15 元数据"""
        attrs = {}

        # 尝试从 _metadata 组读取
        if "_metadata" in f:
            meta = f["_metadata"]
            if hasattr(meta, "attrs"):
                for key in meta.attrs:
                    attrs[key] = meta.attrs[key]

        # 尝试从根属性读取
        for key in ["SamplingFrequency", "SpatialSamplingInterval", "GaugeLength"]:
            if key in f.attrs:
                attrs[key] = f.attrs[key]

        return attrs

    def scan(self, path: Path) -> FormatMetadata:
        """快速扫描 Terra15 文件元数据"""
        with h5py.File(path, "r") as f:
            if self.DATA_PATH not in f:
                raise ValueError(f"无效的 Terra15 文件: {path}")

            dataset = f[self.DATA_PATH]
            shape = dataset.shape
            attrs = self._read_attrs(f)

            # 获取采样率
            fs = attrs.get("SamplingFrequency", 1000.0)
            if isinstance(fs, (list, np.ndarray)):
                fs = float(fs[0]) if len(fs) > 0 else 1000.0
            else:
                fs = float(fs)

            # 获取通道间距
            dx = attrs.get("SpatialSamplingInterval", 1.0)
            if isinstance(dx, (list, np.ndarray)):
                dx = float(dx[0]) if len(dx) > 0 else 1.0
            else:
                dx = float(dx)

            return FormatMetadata(
                n_samples=shape[0],
                n_channels=shape[1] if len(shape) > 1 else 1,
                sampling_rate=fs,
                channel_spacing=dx,
                gauge_length=attrs.get("GaugeLength"),
                file_size=path.stat().st_size,
                format_name=self.format_name,
                format_version=self.version,
                attrs=attrs,
            )

    def read(
        self,
        path: Path,
        channels: Optional[list[int]] = None,
        time_slice: Optional[tuple[int, int]] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> Union[xr.DataArray, da.Array, np.ndarray]:
        """读取 Terra15 格式数据"""
        f = h5py.File(path, "r")

        try:
            if self.DATA_PATH not in f:
                f.close()
                raise ValueError(f"无效的 Terra15 文件: {path}")

            dataset = f[self.DATA_PATH]
            attrs = self._read_attrs(f)

            # 获取采样率
            fs = attrs.get("SamplingFrequency", 1000.0)
            if isinstance(fs, (list, np.ndarray)):
                fs = float(fs[0]) if len(fs) > 0 else 1000.0
            else:
                fs = float(fs)

            # 获取通道间距
            dx = attrs.get("SpatialSamplingInterval", 1.0)
            if isinstance(dx, (list, np.ndarray)):
                dx = float(dx[0]) if len(dx) > 0 else 1.0
            else:
                dx = float(dx)

            if lazy:
                data = da.from_array(dataset, chunks="auto")
            else:
                data = np.array(dataset)
                f.close()

            # 应用选择
            n_samples, n_ch = dataset.shape[0], dataset.shape[1]
            if channels is not None:
                data = data[:, channels]
                channel_coord = channels
            else:
                channel_coord = np.arange(n_ch) * dx

            if time_slice is not None:
                start, end = time_slice
                data = data[start:end]
                time_coord = np.arange(start, end) / fs
            else:
                time_coord = np.arange(n_samples) / fs

            return xr.DataArray(
                data,
                dims=["time", "channel"],
                coords={"time": time_coord, "channel": channel_coord},
                attrs={
                    "sampling_rate": fs,
                    "channel_spacing": dx,
                    "format": self.format_name,
                    "source_file": str(path),
                },
            )

        except Exception as e:
            f.close()
            raise e
