"""DAS 文件格式插件协议定义

该模块定义了格式插件的基础协议和抽象类, 所有格式插件都需要实现这些接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from ...core.inventory import DASInventory


@dataclass
class FormatMetadata:
    """格式元数据, 用于快速扫描时返回"""

    # 基础信息
    n_samples: int
    n_channels: int
    sampling_rate: float  # Hz

    # 可选信息
    start_time: Optional[str] = None
    channel_spacing: Optional[float] = None  # 通道间距 (m)
    gauge_length: Optional[float] = None  # 标距长度 (m)
    data_unit: str = "unknown"  # strain, strain_rate, velocity

    # 文件信息
    file_size: int = 0
    format_name: str = ""
    format_version: str = ""

    # 扩展属性
    attrs: dict = field(default_factory=dict)

    def to_inventory(self) -> "DASInventory":
        """Convert to DASInventory object."""
        from ...core.inventory import Acquisition, DASInventory, FiberGeometry

        return DASInventory(
            project_name=self.attrs.get("project_name", "Unknown"),
            acquisition=Acquisition(
                start_time=self.start_time or "1970-01-01T00:00:00",  # type: ignore
                n_channels=self.n_channels,
                n_samples=self.n_samples,
                data_unit=self.data_unit,
            ),
            fiber=FiberGeometry(
                channel_spacing=self.channel_spacing or 1.0,
                gauge_length=self.gauge_length or 1.0,
            ),
            custom_attrs=self.attrs,
        )


class FormatPlugin(ABC):
    """DAS 文件格式插件抽象基类

    所有格式插件都需要继承此类并实现必要的方法。

    Attributes:
        format_name: 格式名称, 如 "DAT", "H5", "TERRA15"
        version: 插件版本
        file_extensions: 支持的文件扩展名元组, 如 (".dat", ".bin")
        priority: 格式检测优先级, 数值越小优先级越高

    Example:
        >>> class MyFormatPlugin(FormatPlugin):
        ...     format_name = "MYFORMAT"
        ...     version = "1.0.0"
        ...     file_extensions = (".mfmt",)
        ...     priority = 50
        ...
        ...     def can_read(self, path: Path) -> bool:
        ...         return path.suffix.lower() == ".mfmt"
        ...
        ...     def scan(self, path: Path) -> FormatMetadata:
        ...         # 快速扫描元数据
        ...         ...
        ...
        ...     def read(self, path: Path, **kwargs) -> xr.DataArray:
        ...         # 读取数据
        ...         ...
    """

    # 格式标识 (子类必须覆盖)
    format_name: str = ""
    version: str = "1.0.0"
    file_extensions: tuple[str, ...] = ()
    priority: int = 50  # 默认优先级

    @abstractmethod
    def can_read(self, path: Path) -> bool:
        """快速检测文件是否为该格式

        仅基于文件扩展名和/或魔数进行快速判断, 不读取完整数据。

        Args:
            path: 文件路径

        Returns:
            bool: 是否可以读取该文件
        """
        ...

    @abstractmethod
    def scan(self, path: Path) -> FormatMetadata:
        """快速扫描元数据, 不加载数据

        用于多文件索引构建, 需要尽可能快速返回。

        Args:
            path: 文件路径

        Returns:
            FormatMetadata: 格式元数据
        """
        ...

    @abstractmethod
    def read(
        self,
        path: Path,
        channels: Optional[list[int]] = None,
        time_slice: Optional[tuple[int, int]] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> Union[xr.DataArray, da.Array, np.ndarray]:
        """读取数据

        Args:
            path: 文件路径
            channels: 要读取的通道索引列表, None 表示全部
            time_slice: 时间切片 (start, end), None 表示全部
            lazy: 是否延迟加载 (返回 dask 数组)
            **kwargs: 格式特定参数

        Returns:
            xr.DataArray | da.Array | np.ndarray: 读取的数据
        """
        ...

    def write(
        self,
        data: Union[xr.DataArray, np.ndarray],
        path: Path,
        **kwargs: Any,
    ) -> None:
        """写入数据 (可选实现)

        Args:
            data: 要写入的数据
            path: 输出路径
            **kwargs: 格式特定参数

        Raises:
            NotImplementedError: 如果格式不支持写入
        """
        raise NotImplementedError(f"{self.format_name} 格式不支持写入")

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"format={self.format_name!r} "
            f"extensions={self.file_extensions}>"
        )
