"""Silixa 格式插件

Silixa 是英国的 DAS 设备制造商, 产品包括 iDAS 和 cDAS。
数据使用 HDF5 格式存储, 结构类似 PRODML。

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


class SilixaFormatPlugin(FormatPlugin):
    """Silixa 格式插件

    Silixa iDAS/cDAS 询问器的 HDF5 格式。
    数据结构类似 PRODML: Acquisition/Raw[0]/RawData

    Attributes:
        format_name: "SILIXA"
        file_extensions: (".h5", ".hdf5", ".tdms")
        priority: 21
    """

    format_name = "SILIXA"
    version = "1.0.0"
    file_extensions = (".h5", ".hdf5", ".tdms")
    priority = 21

    def can_read(self, path: Path) -> bool:
        """检测文件是否为 Silixa 格式"""
        if not path.exists():
            return False

        if path.suffix.lower() not in self.file_extensions:
            return False

        try:
            with h5py.File(path, "r") as f:
                # Silixa 特有标识
                for key in ["Manufacturer", "manufacturer", "SystemInfomation"]:
                    if key in f.attrs:
                        val = f.attrs[key]
                        if isinstance(val, bytes):
                            val = val.decode()
                        if "silixa" in str(val).lower():
                            return True

                # 检查 Silixa 特有属性
                if "MeasureLength[m]" in f.attrs or "SpatialResolution[m]" in f.attrs:
                    return True

                return False
        except Exception:
            return False

    def _read_attrs(self, f: h5py.File) -> dict:
        """读取 Silixa 元数据"""
        attrs = {}

        # Silixa 特有属性名
        attr_map = {
            "SamplingFrequency[Hz]": "SamplingFrequency",
            "SpatialResolution[m]": "SpatialResolution",
            "GaugeLength[m]": "GaugeLength",
            "MeasureLength[m]": "MeasureLength",
            "StartDistance[m]": "StartDistance",
            "StopDistance[m]": "StopDistance",
        }

        for key in f.attrs:
            val = f.attrs[key]
            if isinstance(val, bytes):
                val = val.decode()
            # 标准化属性名
            std_key = attr_map.get(key, key)
            attrs[std_key] = val

        return attrs

    def _find_data_path(self, f: h5py.File) -> Optional[str]:
        """查找数据集路径"""
        paths = [
            "Acquisition/Raw[0]/RawData",
            "Acquisition/Raw[0]",
            "RawData",
            "data",
        ]
        for dp in paths:
            if dp in f:
                item = f[dp]
                if isinstance(item, h5py.Dataset):
                    return dp
        return None

    def scan(self, path: Path) -> FormatMetadata:
        """快速扫描 Silixa 文件元数据"""
        with h5py.File(path, "r") as f:
            dp = self._find_data_path(f)
            if dp is None:
                raise ValueError(f"无效的 Silixa 文件: {path}")

            dataset = f[dp]
            shape = dataset.shape
            attrs = self._read_attrs(f)

            fs = float(attrs.get("SamplingFrequency", 1000.0))
            dx = float(attrs.get("SpatialResolution", 1.0))
            gl = attrs.get("GaugeLength")

            return FormatMetadata(
                n_samples=shape[0],
                n_channels=shape[1] if len(shape) > 1 else 1,
                sampling_rate=fs,
                channel_spacing=dx,
                gauge_length=float(gl) if gl is not None else None,
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
        """读取 Silixa 格式数据"""
        f = h5py.File(path, "r")

        try:
            dp = self._find_data_path(f)
            if dp is None:
                f.close()
                raise ValueError(f"无效的 Silixa 文件: {path}")

            dataset = f[dp]
            attrs = self._read_attrs(f)

            fs = float(attrs.get("SamplingFrequency", 1000.0))
            dx = float(attrs.get("SpatialResolution", 1.0))

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
                time_coord = np.arange(end - start) / fs
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
