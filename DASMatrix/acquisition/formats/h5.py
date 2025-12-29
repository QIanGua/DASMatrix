"""HDF5 格式插件

支持标准 HDF5 格式的 DAS 数据文件, 包括常见的数据集路径约定。
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

# 常见的 HDF5 数据集路径
COMMON_DATASET_PATHS = [
    "Acquisition/Raw[0]",
    "Acquisition/Raw[0]/RawData",
    "data",
    "Data",
    "raw_data",
    "DAS/data",
]


class H5FormatPlugin(FormatPlugin):
    """HDF5 格式插件

    支持标准 HDF5 格式的 DAS 数据文件。

    Attributes:
        format_name: "H5"
        file_extensions: (".h5", ".hdf5", ".hdf")
        priority: 30 (高优先级)
    """

    format_name = "H5"
    version = "1.0.0"
    file_extensions = (".h5", ".hdf5", ".hdf")
    priority = 30

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        sampling_rate: float = 5000.0,
    ):
        """初始化 HDF5 格式插件

        Args:
            dataset_path: 数据集路径, None 表示自动检测
            sampling_rate: 默认采样率 (Hz)
        """
        self.dataset_path = dataset_path
        self.default_sampling_rate = sampling_rate

    def can_read(self, path: Path) -> bool:
        """检测文件是否为 HDF5 格式"""
        if not path.exists():
            return False

        # 检查扩展名
        if path.suffix.lower() in self.file_extensions:
            return True

        # 检查 HDF5 魔数 (前 8 字节)
        try:
            with open(path, "rb") as f:
                magic = f.read(8)
                # HDF5 魔数: \x89HDF\r\n\x1a\n
                return magic[:4] == b"\x89HDF"
        except Exception:
            return False

    def _find_dataset(self, f: h5py.File) -> Optional[str]:
        """在 HDF5 文件中查找数据集"""
        if self.dataset_path and self.dataset_path in f:
            return self.dataset_path

        for ds_path in COMMON_DATASET_PATHS:
            if ds_path in f:
                return ds_path

        # 递归查找第一个 2D 数据集
        def find_2d_dataset(group: h5py.Group, prefix: str = "") -> Optional[str]:
            for key in group.keys():
                item = group[key]
                full_path = f"{prefix}/{key}" if prefix else key
                if isinstance(item, h5py.Dataset) and len(item.shape) == 2:
                    return full_path
                elif isinstance(item, h5py.Group):
                    result = find_2d_dataset(item, full_path)
                    if result:
                        return result
            return None

        return find_2d_dataset(f)

    def _read_attrs(self, f: h5py.File) -> dict:
        """读取 HDF5 属性"""
        attrs = {}

        # 尝试读取采样率
        for key in ["SamplingFrequency", "sampling_rate", "fs", "sample_rate"]:
            if key in f.attrs:
                attrs["sampling_rate"] = float(f.attrs[key])
                break

        # 尝试读取通道间距
        for key in ["SpatialSamplingInterval", "dx", "channel_spacing"]:
            if key in f.attrs:
                attrs["channel_spacing"] = float(f.attrs[key])
                break

        # 尝试读取标距长度
        for key in ["GaugeLength", "gauge_length"]:
            if key in f.attrs:
                attrs["gauge_length"] = float(f.attrs[key])
                break

        return attrs

    def scan(self, path: Path) -> FormatMetadata:
        """快速扫描 HDF5 文件元数据"""
        with h5py.File(path, "r") as f:
            ds_path = self._find_dataset(f)
            if ds_path is None:
                raise ValueError(f"未找到有效的数据集: {path}")

            dataset = f[ds_path]
            shape = dataset.shape

            # 读取属性
            attrs = self._read_attrs(f)
            fs = attrs.get("sampling_rate", self.default_sampling_rate)

            return FormatMetadata(
                n_samples=shape[0],
                n_channels=shape[1] if len(shape) > 1 else 1,
                sampling_rate=fs,
                channel_spacing=attrs.get("channel_spacing"),
                gauge_length=attrs.get("gauge_length"),
                file_size=path.stat().st_size,
                format_name=self.format_name,
                format_version=self.version,
                attrs={"dataset_path": ds_path},
            )

    def read(
        self,
        path: Path,
        channels: Optional[list[int]] = None,
        time_slice: Optional[tuple[int, int]] = None,
        lazy: bool = True,
        dataset_path: Optional[str] = None,
        sampling_rate: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[xr.DataArray, da.Array, np.ndarray]:
        """读取 HDF5 格式数据

        Args:
            path: 文件路径
            channels: 要读取的通道索引列表
            time_slice: 时间切片 (start, end)
            lazy: 是否延迟加载
            dataset_path: 数据集路径 (覆盖默认值)
            sampling_rate: 采样率 (覆盖默认值)

        Returns:
            xr.DataArray: 带标签的数据数组
        """
        f = h5py.File(path, "r")

        try:
            # 查找数据集
            ds_path = dataset_path or self._find_dataset(f)
            if ds_path is None:
                f.close()
                raise ValueError(f"未找到有效的数据集: {path}")

            dataset = f[ds_path]

            # 读取属性
            attrs = self._read_attrs(f)
            fs = sampling_rate or attrs.get("sampling_rate", self.default_sampling_rate)

            if lazy:
                # 延迟加载
                data = da.from_array(dataset, chunks="auto")
            else:
                # 立即加载
                data = np.array(dataset)
                f.close()

            # 转换为物理量 (与原始代码一致)
            data = (data / 4) * (np.pi / (2**13))

            # 应用通道选择
            n_ch = dataset.shape[1] if len(dataset.shape) > 1 else 1
            if channels is not None:
                data = data[:, channels]
                n_ch = len(channels)
                channel_coord = channels
            else:
                channel_coord = np.arange(n_ch)

            # 应用时间切片
            n_samples = dataset.shape[0]
            if time_slice is not None:
                start, end = time_slice
                data = data[start:end]
                n_samples = end - start
                time_coord = np.arange(start, end) / fs
            else:
                time_coord = np.arange(n_samples) / fs

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
                    "dataset_path": ds_path,
                    "units": "rad/s",
                    "source_file": str(path),
                    **{k: v for k, v in attrs.items() if k != "sampling_rate"},
                },
            )

        except Exception as e:
            f.close()
            raise e
