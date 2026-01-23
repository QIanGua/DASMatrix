"""PRODML 格式插件

PRODML (Production Markup Language) 是石油天然气工业的数据交换标准。
PRODML v2.0/v2.1 定义了 DAS 数据的 HDF5 存储格式。

参考:
- https://www.energistics.org/prodml
- https://github.com/DASDAE/dascore (MIT License)
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


class PRODMLFormatPlugin(FormatPlugin):
    """PRODML 格式插件

    PRODML 是石油天然气工业标准, v2.0/v2.1 定义了 DAS 数据格式。
    数据存储结构:
    - /Acquisition/Raw[0]/RawData
    - /Acquisition/Raw[0]/RawDataTime

    Attributes:
        format_name: "PRODML"
        file_extensions: (".h5", ".hdf5")
        priority: 20 (最高优先级)
    """

    format_name = "PRODML"
    version = "2.1"
    file_extensions = (".h5", ".hdf5")
    priority = 20

    # PRODML 特有的数据集路径
    DATA_PATHS = [
        "Acquisition/Raw[0]/RawData",
        "Acquisition/Raw[0]/RawData[0]",
        "Acquisition/Custom/Raw[0]/RawData",
    ]

    def can_read(self, path: Path) -> bool:
        """检测文件是否为 PRODML 格式"""
        if not path.exists():
            return False

        if path.suffix.lower() not in self.file_extensions:
            return False

        try:
            with h5py.File(path, "r") as f:
                # PRODML 特有: Acquisition 组
                if "Acquisition" not in f:
                    return False

                # 检查是否有 PRODML 标识
                if "SchemaVersion" in f.attrs:
                    return True

                # 检查常见的 PRODML 数据路径
                for dp in self.DATA_PATHS:
                    if dp in f:
                        return True

                return False
        except Exception:
            return False

    def _find_data_path(self, f: h5py.File) -> Optional[str]:
        """查找数据集路径"""
        for dp in self.DATA_PATHS:
            if dp in f:
                return dp
        return None

    def _read_attrs(self, f: h5py.File) -> dict:
        """读取 PRODML 元数据"""
        attrs = {}

        # PRODML 标准属性
        for key in [
            "SamplingFrequency",
            "SpatialSamplingInterval",
            "GaugeLength",
            "PulseRate",
            "PulseWidth",
            "StartTime",
            "EndTime",
            "NumberOfLoci",
            "ProjectName",
            "DataUnit",
            "IUModel",
            "IUManufacturer",
            "SchemaVersion",
        ]:
            if key in f.attrs:
                attrs[key] = f.attrs[key]

        # 尝试从 Acquisition 组读取
        if "Acquisition" in f:
            acq = f["Acquisition"]
            if hasattr(acq, "attrs"):
                for key in acq.attrs:
                    if key not in attrs:
                        attrs[key] = acq.attrs[key]

        return attrs

    def scan(self, path: Path) -> FormatMetadata:
        """快速扫描 PRODML 文件元数据并构建 DASInventory"""
        from datetime import datetime

        from ...core.inventory import Acquisition, DASInventory, FiberGeometry, Interrogator

        with h5py.File(path, "r") as f:
            dp = self._find_data_path(f)
            if dp is None:
                raise ValueError(f"无效的 PRODML 文件: {path}")

            dataset = f[dp]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"Expected Dataset at {dp}")
            shape = dataset.shape
            attrs = self._read_attrs(f)

            fs = float(attrs.get("SamplingFrequency", 1000.0))
            dx = float(attrs.get("SpatialSamplingInterval", 1.0))
            gl = attrs.get("GaugeLength")

            # 构建标准的 DASInventory
            try:
                start_time_str = str(attrs.get("StartTime", "1970-01-01T00:00:00"))
                # Remove Z or handle common ISO formats if needed
                start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
            except Exception:
                from datetime import timezone

                start_time = datetime.fromtimestamp(0, tz=timezone.utc)

            inventory = DASInventory(
                project_name=str(attrs.get("ProjectName", "PRODML Project")),
                acquisition=Acquisition(
                    start_time=start_time,
                    n_channels=shape[1] if len(shape) > 1 else 1,
                    n_samples=shape[0],
                    data_unit=str(attrs.get("DataUnit", "strain_rate")),
                ),
                fiber=FiberGeometry(
                    channel_spacing=dx,
                    gauge_length=float(gl) if gl is not None else 1.0,
                ),
                interrogator=Interrogator(
                    model=str(attrs.get("IUModel", "Unknown")),
                    manufacturer=str(attrs.get("IUManufacturer", "Unknown")),
                    sampling_rate=fs,
                ),
                custom_attrs={k: v for k, v in attrs.items() if isinstance(v, (int, float, str))},
            )

            return FormatMetadata(
                n_samples=shape[0],
                n_channels=shape[1] if len(shape) > 1 else 1,
                sampling_rate=fs,
                channel_spacing=dx,
                gauge_length=float(gl) if gl is not None else None,
                start_time=str(attrs.get("StartTime", "")),
                file_size=path.stat().st_size,
                format_name=self.format_name,
                format_version=str(attrs.get("SchemaVersion", self.version)),
                inventory=inventory,
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
        """读取 PRODML 格式数据"""
        f = h5py.File(path, "r")

        try:
            dp = self._find_data_path(f)
            if dp is None:
                f.close()
                raise ValueError(f"无效的 PRODML 文件: {path}")

            dataset = f[dp]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"Expected Dataset at {dp}")
            attrs = self._read_attrs(f)

            fs = float(attrs.get("SamplingFrequency", 1000.0))
            dx = float(attrs.get("SpatialSamplingInterval", 1.0))

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
                    **{k: v for k, v in attrs.items() if isinstance(v, (int, float, str))},
                },
            )

        except Exception as e:
            f.close()
            raise e
