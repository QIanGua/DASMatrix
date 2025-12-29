"""AP Sensing 格式插件

AP Sensing 是德国的 DAS 设备制造商。
数据使用 HDF5 格式存储。

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


class APSensingFormatPlugin(FormatPlugin):
    """AP Sensing 格式插件

    AP Sensing 询问器的 HDF5 格式。

    Attributes:
        format_name: "APSENSING"
        file_extensions: (".h5", ".hdf5")
        priority: 23
    """

    format_name = "APSENSING"
    version = "1.0.0"
    file_extensions = (".h5", ".hdf5")
    priority = 23

    def can_read(self, path: Path) -> bool:
        """检测文件是否为 AP Sensing 格式"""
        if not path.exists():
            return False

        if path.suffix.lower() not in self.file_extensions:
            return False

        try:
            with h5py.File(path, "r") as f:
                # AP Sensing 特有标识
                if "manufacturer" in f.attrs:
                    mfr = f.attrs["manufacturer"]
                    if isinstance(mfr, bytes):
                        mfr = mfr.decode()
                    if "ap sensing" in mfr.lower() or "apsensing" in mfr.lower():
                        return True

                # 检查 AP Sensing 特有的数据结构
                if "DAS" in f and "data" in f["DAS"]:
                    return True

                return False
        except Exception:
            return False

    def _find_data_path(self, f: h5py.File) -> Optional[str]:
        """查找数据集路径"""
        paths = ["DAS/data", "data", "Acquisition/Raw[0]"]
        for dp in paths:
            if dp in f:
                return dp
        return None

    def _read_attrs(self, f: h5py.File) -> dict:
        """读取 AP Sensing 元数据"""
        attrs = {}

        for key in f.attrs:
            val = f.attrs[key]
            if isinstance(val, bytes):
                val = val.decode()
            attrs[key] = val

        if "DAS" in f and hasattr(f["DAS"], "attrs"):
            for key in f["DAS"].attrs:
                val = f["DAS"].attrs[key]
                if isinstance(val, bytes):
                    val = val.decode()
                attrs[key] = val

        return attrs

    def scan(self, path: Path) -> FormatMetadata:
        """快速扫描 AP Sensing 文件元数据"""
        with h5py.File(path, "r") as f:
            dp = self._find_data_path(f)
            if dp is None:
                raise ValueError(f"无效的 AP Sensing 文件: {path}")

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
        """读取 AP Sensing 格式数据"""
        f = h5py.File(path, "r")

        try:
            dp = self._find_data_path(f)
            if dp is None:
                f.close()
                raise ValueError(f"无效的 AP Sensing 文件: {path}")

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
