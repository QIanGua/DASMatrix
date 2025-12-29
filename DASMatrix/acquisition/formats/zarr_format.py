"""Zarr 格式插件

Zarr 是云原生的分块数组存储格式, 支持并行读写和多种压缩算法。
非常适合大规模 DAS 数据的存储和分析。

参考: https://zarr.readthedocs.io/
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr

from .base import FormatMetadata, FormatPlugin

logger = logging.getLogger(__name__)


class ZarrFormatPlugin(FormatPlugin):
    """Zarr 格式插件

    云原生分块数组格式, 支持:
    - 高效的分块存储和访问
    - 多种压缩算法 (Blosc, LZ4, Zstd 等)
    - 并行读写
    - S3/GCS 等云存储

    Attributes:
        format_name: "ZARR"
        file_extensions: (".zarr", ".zr")
        priority: 35
    """

    format_name = "ZARR"
    version = "3.0"
    file_extensions = (".zarr", ".zr")
    priority = 35

    def __init__(self, sampling_rate: float = 5000.0):
        """初始化 Zarr 格式插件

        Args:
            sampling_rate: 默认采样率 (Hz)
        """
        self.default_sampling_rate = sampling_rate

    def _check_zarr(self) -> bool:
        """检查 zarr 是否可用"""
        try:
            import zarr  # noqa: F401

            return True
        except ImportError:
            return False

    def can_read(self, path: Path) -> bool:
        """检测文件是否为 Zarr 格式"""
        if not path.exists():
            return False

        # Zarr 可以是目录或 zip 文件
        if path.is_dir():
            # 检查 .zarray 或 .zattrs 文件
            return (path / ".zarray").exists() or (path / ".zattrs").exists()

        # 检查 zip 格式的 zarr
        if path.suffix.lower() in (".zarr", ".zr", ".zip"):
            return True

        return False

    def scan(self, path: Path) -> FormatMetadata:
        """快速扫描 Zarr 文件元数据"""
        if not self._check_zarr():
            raise ImportError("需要安装 zarr: pip install zarr")

        import zarr

        z = zarr.open(path, mode="r")

        # 如果是 group, 查找数据数组
        if isinstance(z, zarr.Group):
            if "data" in z:
                arr = z["data"]
            elif "strain" in z:
                arr = z["strain"]
            else:
                # 使用第一个数组
                arrays = [k for k in z.keys() if isinstance(z[k], zarr.Array)]
                if arrays:
                    arr = z[arrays[0]]
                else:
                    raise ValueError(f"Zarr 文件中没有数据数组: {path}")
        else:
            arr = z

        shape = arr.shape  # type: ignore[union-attr]
        attrs = dict(z.attrs) if hasattr(z, "attrs") else {}

        default_fs = self.default_sampling_rate
        fs = float(attrs.get("sampling_rate", attrs.get("fs", default_fs)))
        dx = float(attrs.get("channel_spacing", attrs.get("dx", 1.0)))

        return FormatMetadata(
            n_samples=shape[0],
            n_channels=shape[1] if len(shape) > 1 else 1,
            sampling_rate=fs,
            channel_spacing=dx,
            file_size=0,  # Zarr 目录大小难以快速计算
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
        array_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[xr.DataArray, da.Array, np.ndarray]:
        """读取 Zarr 格式数据

        Args:
            path: 文件路径
            channels: 要读取的通道索引列表
            time_slice: 时间切片 (start, end)
            lazy: 是否延迟加载
            array_name: 数组名称 (用于 Zarr Group)
        """
        if not self._check_zarr():
            raise ImportError("需要安装 zarr: pip install zarr")

        import zarr

        z = zarr.open(path, mode="r")

        # 查找数据数组
        if isinstance(z, zarr.Group):
            if array_name and array_name in z:
                arr = z[array_name]
            elif "data" in z:
                arr = z["data"]
            elif "strain" in z:
                arr = z["strain"]
            else:
                arrays = [k for k in z.keys() if isinstance(z[k], zarr.Array)]
                if arrays:
                    arr = z[arrays[0]]
                else:
                    raise ValueError(f"Zarr 文件中没有数据数组: {path}")
        else:
            arr = z

        attrs = dict(z.attrs) if hasattr(z, "attrs") else {}
        default_fs = self.default_sampling_rate
        fs = float(attrs.get("sampling_rate", attrs.get("fs", default_fs)))
        dx = float(attrs.get("channel_spacing", attrs.get("dx", 1.0)))

        # 使用 dask 读取
        if lazy:
            data = da.from_zarr(arr)
        else:
            data = np.array(arr)

        arr_shape = arr.shape  # type: ignore[union-attr]
        n_samples, n_ch = arr_shape[0], arr_shape[1] if len(arr_shape) > 1 else 1

        # 应用选择
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

    def write(
        self,
        data: Union[xr.DataArray, np.ndarray],
        path: Path,
        chunks: Optional[tuple[int, ...]] = None,
        compressor: Optional[str] = "blosc",
        **kwargs: Any,
    ) -> None:
        """写入 Zarr 格式数据

        Args:
            data: 要写入的数据
            path: 输出路径
            chunks: 分块大小
            compressor: 压缩器 ("blosc", "lz4", "zstd", None)
        """
        if not self._check_zarr():
            raise ImportError("需要安装 zarr: pip install zarr")

        import zarr
        from numcodecs import Blosc

        # 获取数据数组
        if isinstance(data, xr.DataArray):
            arr = data.values
            attrs = dict(data.attrs)
        else:
            arr = data
            attrs = {}

        # 设置压缩器
        if compressor == "blosc":
            comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        elif compressor:
            comp = Blosc(cname=compressor, clevel=3)
        else:
            comp = None

        # 创建 zarr 存储
        z = zarr.open(path, mode="w")
        z.create_dataset(
            "data",
            data=arr,
            chunks=chunks or "auto",
            compressor=comp,
        )

        # 保存属性
        z.attrs.update(attrs)
        z.attrs["format"] = self.format_name
