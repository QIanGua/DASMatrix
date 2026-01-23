"""NetCDF 格式插件

NetCDF (Network Common Data Form) 是广泛使用的科学数据格式。
支持自描述、可移植的数组数据存储。

参考: https://www.unidata.ucar.edu/software/netcdf/
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr

from .base import FormatMetadata, FormatPlugin

logger = logging.getLogger(__name__)


class NetCDFFormatPlugin(FormatPlugin):
    """NetCDF 格式插件

    支持 NetCDF-3 和 NetCDF-4 (基于 HDF5) 格式。

    Attributes:
        format_name: "NETCDF"
        file_extensions: (".nc", ".nc4", ".netcdf")
        priority: 40
    """

    format_name = "NETCDF"
    version = "4.0"
    file_extensions = (".nc", ".nc4", ".netcdf")
    priority = 40

    def __init__(self, sampling_rate: float = 5000.0):
        """初始化 NetCDF 格式插件

        Args:
            sampling_rate: 默认采样率 (Hz)
        """
        self.default_sampling_rate = sampling_rate

    def can_read(self, path: Path) -> bool:
        """检测文件是否为 NetCDF 格式"""
        if not path.exists():
            return False

        if path.suffix.lower() in self.file_extensions:
            return True

        # 检查 NetCDF 魔数
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
                # CDF\x01 或 CDF\x02 (NetCDF-3)
                # \x89HDF (NetCDF-4/HDF5)
                return magic[:3] == b"CDF" or magic[:4] == b"\x89HDF"
        except Exception:
            return False

    def _find_data_var(self, ds: xr.Dataset) -> str:
        """查找主数据变量"""
        # 常见的变量名
        common_names = ["data", "strain", "strain_rate", "velocity", "amplitude"]
        for name in common_names:
            if name in ds.data_vars:
                return name

        # 返回第一个 2D 变量
        for name in ds.data_vars:
            if len(ds[name].dims) >= 2:
                return str(name)

        # 返回第一个变量
        if ds.data_vars:
            return list(ds.data_vars)[0]

        raise ValueError("NetCDF 文件中没有数据变量")

    def scan(self, path: Path) -> FormatMetadata:
        """快速扫描 NetCDF 文件元数据"""
        # 使用 xarray 打开但不加载数据
        ds = xr.open_dataset(path)

        try:
            var_name = self._find_data_var(ds)
            var = ds[var_name]
            shape = var.shape

            # 从属性获取采样率
            default_fs = self.default_sampling_rate
            fs = float(
                ds.attrs.get(
                    "sampling_rate",
                    ds.attrs.get("fs", var.attrs.get("sampling_rate", default_fs)),
                )
            )

            dx = float(ds.attrs.get("channel_spacing", ds.attrs.get("dx", 1.0)))

            return FormatMetadata(
                n_samples=shape[0],
                n_channels=shape[1] if len(shape) > 1 else 1,
                sampling_rate=fs,
                channel_spacing=dx,
                file_size=path.stat().st_size,
                format_name=self.format_name,
                format_version=self.version,
                attrs=dict(ds.attrs),
            )
        finally:
            ds.close()

    def read(
        self,
        path: Path,
        channels: Optional[list[int]] = None,
        time_slice: Optional[tuple[int, int]] = None,
        lazy: bool = True,
        var_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[xr.DataArray, da.Array, np.ndarray]:
        """读取 NetCDF 格式数据

        Args:
            path: 文件路径
            channels: 要读取的通道索引列表
            time_slice: 时间切片 (start, end)
            lazy: 是否延迟加载
            var_name: 变量名称, None 表示自动检测
        """
        # 使用 xarray 打开
        if lazy:
            ds = xr.open_dataset(path, chunks="auto")
        else:
            ds = xr.open_dataset(path)

        try:
            var_name = var_name or self._find_data_var(ds)
            data = ds[var_name]

            # 获取采样率
            default_fs = self.default_sampling_rate
            fs = float(
                ds.attrs.get(
                    "sampling_rate",
                    ds.attrs.get("fs", data.attrs.get("sampling_rate", default_fs)),
                )
            )

            dx = float(ds.attrs.get("channel_spacing", ds.attrs.get("dx", 1.0)))

            # 确定维度名称
            time_dim = None
            channel_dim = None
            for dim in data.dims:
                dim_lower = str(dim).lower()
                if "time" in dim_lower or dim_lower == "t":
                    time_dim = dim
                elif "channel" in dim_lower or "distance" in dim_lower or dim_lower in ("x", "d"):
                    channel_dim = dim

            # 应用选择
            if time_slice is not None and time_dim:
                start, end = time_slice
                data = data.isel({time_dim: slice(start, end)})

            if channels is not None and channel_dim:
                data = data.isel({channel_dim: channels})

            # 重命名维度为标准名称
            rename_dict = {}
            if time_dim and time_dim != "time":
                rename_dict[time_dim] = "time"
            if channel_dim and channel_dim != "distance":
                rename_dict[channel_dim] = "distance"

            if rename_dict:
                data = data.rename(rename_dict)

            # 更新属性
            data.attrs["sampling_rate"] = fs
            data.attrs["channel_spacing"] = dx
            data.attrs["format"] = self.format_name
            data.attrs["source_file"] = str(path)

            if not lazy:
                data = data.compute()

            return data

        except Exception as e:
            ds.close()
            raise e

    def write(
        self,
        data: Union[xr.DataArray, np.ndarray],
        path: Path,
        format: str = "NETCDF4",
        **kwargs: Any,
    ) -> None:
        """写入 NetCDF 格式数据

        Args:
            data: 要写入的数据
            path: 输出路径
            format: NetCDF 格式 ("NETCDF3_CLASSIC", "NETCDF4", etc.)
        """
        if isinstance(data, np.ndarray):
            data = xr.DataArray(
                data,
                dims=["time", "distance"],
                attrs={"format": self.format_name},
            )

        # 创建 Dataset
        ds = xr.Dataset({"data": data})
        ds.attrs.update(data.attrs)

        # 写入文件
        ds.to_netcdf(str(path), format=format)  # type: ignore[call-overload]
