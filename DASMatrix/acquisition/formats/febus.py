"""Febus 格式插件

Febus Optics 是法国的 DAS 设备制造商。
数据使用 HDF5 格式存储, 结构与 PRODML 类似但有自定义扩展。

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


class FebusFormatPlugin(FormatPlugin):
    """Febus 格式插件

    Febus Optics 询问器的 HDF5 格式。

    Attributes:
        format_name: "FEBUS"
        file_extensions: (".h5", ".hdf5")
        priority: 22
    """

    format_name = "FEBUS"
    version = "1.0.0"
    file_extensions = (".h5", ".hdf5")
    priority = 22

    # Febus 特有的数据集路径
    DATA_PATH = "Source1/Zone1/Trace"
    ALT_DATA_PATH = "data"

    def can_read(self, path: Path) -> bool:
        """检测文件是否为 Febus 格式"""
        if not path.exists():
            return False

        if path.suffix.lower() not in self.file_extensions:
            return False

        try:
            with h5py.File(path, "r") as f:
                # Febus 特有: Source1/Zone1 结构
                if "Source1" in f and "Zone1" in f["Source1"]:
                    return True

                # 检查 Febus 标识属性
                if "manufacturer" in f.attrs:
                    mfr = f.attrs["manufacturer"]
                    if isinstance(mfr, bytes):
                        mfr = mfr.decode()
                    if "febus" in mfr.lower():
                        return True

                return False
        except Exception:
            return False

    def _find_data_path(self, f: h5py.File) -> Optional[str]:
        """查找数据集路径"""
        if self.DATA_PATH in f:
            return self.DATA_PATH

        # 查找 Source*/Zone*/Trace 模式
        for key in f.keys():
            if key.startswith("Source"):
                source = f[key]
                for zone_key in source.keys():
                    if zone_key.startswith("Zone"):
                        zone_path = f"{key}/{zone_key}/Trace"
                        if zone_path in f:
                            return zone_path

        if self.ALT_DATA_PATH in f:
            return self.ALT_DATA_PATH

        return None

    def _read_attrs(self, f: h5py.File) -> dict:
        """读取 Febus 元数据"""
        attrs = {}

        # 读取根属性
        for key in f.attrs:
            val = f.attrs[key]
            if isinstance(val, bytes):
                val = val.decode()
            attrs[key] = val

        # 尝试从 Zone1 读取采样参数
        if "Source1/Zone1" in f:
            zone = f["Source1/Zone1"]
            if hasattr(zone, "attrs"):
                for key in zone.attrs:
                    val = zone.attrs[key]
                    if isinstance(val, bytes):
                        val = val.decode()
                    attrs[key] = val

        return attrs

    def scan(self, path: Path) -> FormatMetadata:
        """快速扫描 Febus 文件元数据"""
        with h5py.File(path, "r") as f:
            dp = self._find_data_path(f)
            if dp is None:
                raise ValueError(f"无效的 Febus 文件: {path}")

            dataset = f[dp]
            shape = dataset.shape
            attrs = self._read_attrs(f)

            fs = float(attrs.get("SamplingFrequency", attrs.get("fs", 1000.0)))
            dx = float(attrs.get("SpatialResolution", attrs.get("dx", 1.0)))

            return FormatMetadata(
                n_samples=shape[0],
                n_channels=shape[1] if len(shape) > 1 else 1,
                sampling_rate=fs,
                channel_spacing=dx,
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
        """读取 Febus 格式数据"""
        f = h5py.File(path, "r")

        try:
            dp = self._find_data_path(f)
            if dp is None:
                f.close()
                raise ValueError(f"无效的 Febus 文件: {path}")

            dataset = f[dp]
            attrs = self._read_attrs(f)

            fs = float(attrs.get("SamplingFrequency", attrs.get("fs", 1000.0)))
            dx = float(attrs.get("SpatialResolution", attrs.get("dx", 1.0)))

            if lazy:
                data = da.from_array(dataset, chunks="auto")
            else:
                data = np.array(dataset)
                f.close()

            n_samples, n_ch = dataset.shape[0], dataset.shape[1]

            if channels is not None:
                data = data[:, channels]
                channel_coord = np.array(channels) * dx
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
                dims=["time", "distance"],
                coords={"time": time_coord, "distance": channel_coord},
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
