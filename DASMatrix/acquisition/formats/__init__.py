"""DAS 文件格式注册表与插件管理

该模块提供格式注册表, 支持动态发现和注册格式插件。

Example:
    >>> from DASMatrix.acquisition.formats import FormatRegistry
    >>> # 自动检测并读取文件
    >>> data = FormatRegistry.read("/path/to/data.h5")
    >>> # 扫描元数据
    >>> meta = FormatRegistry.scan("/path/to/data.h5")
    >>> # 列出所有支持的格式
    >>> FormatRegistry.list_formats()
"""

import logging
from pathlib import Path
from typing import Optional, Union

import dask.array as da
import numpy as np
import xarray as xr

from .base import FormatMetadata, FormatPlugin

logger = logging.getLogger(__name__)


class FormatRegistry:
    """格式注册表, 支持动态发现和注册格式插件

    这是一个单例类, 通过类方法访问。
    """

    _formats: dict[str, FormatPlugin] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, plugin: FormatPlugin) -> None:
        """注册格式插件

        Args:
            plugin: 格式插件实例

        Raises:
            ValueError: 如果格式名称为空
        """
        if not plugin.format_name:
            raise ValueError("格式插件必须定义 format_name")

        name = plugin.format_name.upper()
        if name in cls._formats:
            logger.warning(f"覆盖已注册的格式: {name}")
        cls._formats[name] = plugin
        logger.debug(f"注册格式: {name} ({plugin.__class__.__name__})")

    @classmethod
    def unregister(cls, format_name: str) -> bool:
        """取消注册格式插件

        Args:
            format_name: 格式名称

        Returns:
            bool: 是否成功取消注册
        """
        name = format_name.upper()
        if name in cls._formats:
            del cls._formats[name]
            logger.debug(f"取消注册格式: {name}")
            return True
        return False

    @classmethod
    def get(cls, format_name: str) -> Optional[FormatPlugin]:
        """获取格式插件

        Args:
            format_name: 格式名称

        Returns:
            FormatPlugin | None: 格式插件实例或 None
        """
        cls._ensure_initialized()
        return cls._formats.get(format_name.upper())

    @classmethod
    def detect_format(cls, path: Union[str, Path]) -> Optional[str]:
        """自动检测文件格式

        按优先级顺序尝试各格式插件的 can_read 方法。

        Args:
            path: 文件路径

        Returns:
            str | None: 检测到的格式名称, 未知格式返回 None
        """
        cls._ensure_initialized()
        path = Path(path)

        if not path.exists():
            logger.warning(f"文件不存在: {path}")
            return None

        # 按优先级排序
        sorted_plugins = sorted(
            cls._formats.values(),
            key=lambda p: p.priority,
        )

        for plugin in sorted_plugins:
            try:
                if plugin.can_read(path):
                    logger.debug(f"检测到格式: {plugin.format_name}")
                    return plugin.format_name
            except Exception as e:
                logger.debug(f"{plugin.format_name} 检测失败: {e}")
                continue

        logger.warning(f"无法识别文件格式: {path}")
        return None

    @classmethod
    def scan(
        cls,
        path: Union[str, Path],
        format_name: Optional[str] = None,
    ) -> FormatMetadata:
        """扫描文件元数据

        Args:
            path: 文件路径
            format_name: 强制指定格式, None 表示自动检测

        Returns:
            FormatMetadata: 文件元数据

        Raises:
            ValueError: 无法识别文件格式
        """
        cls._ensure_initialized()
        path = Path(path)

        if format_name is None:
            format_name = cls.detect_format(path)

        if format_name is None:
            raise ValueError(f"无法识别文件格式: {path}")

        plugin = cls._formats.get(format_name.upper())
        if plugin is None:
            raise ValueError(f"未注册的格式: {format_name}")

        return plugin.scan(path)

    @classmethod
    def read(
        cls,
        path: Union[str, Path],
        format_name: Optional[str] = None,
        **kwargs,
    ) -> Union[xr.DataArray, da.Array, np.ndarray]:
        """统一读取接口

        Args:
            path: 文件路径
            format_name: 强制指定格式, None 表示自动检测
            **kwargs: 传递给格式插件的参数

        Returns:
            xr.DataArray | da.Array | np.ndarray: 读取的数据

        Raises:
            ValueError: 无法识别文件格式
        """
        cls._ensure_initialized()
        path = Path(path)

        if format_name is None:
            format_name = cls.detect_format(path)

        if format_name is None:
            raise ValueError(f"无法识别文件格式: {path}")

        plugin = cls._formats.get(format_name.upper())
        if plugin is None:
            raise ValueError(f"未注册的格式: {format_name}")

        logger.debug(f"使用 {format_name} 格式读取: {path}")
        data = plugin.read(path, **kwargs)

        # 尝试附加 Inventory
        if isinstance(data, (xr.DataArray, xr.Dataset)) and "inventory" not in data.attrs:
            try:
                # 优先检查插件是否已经注入了 inventory，如果没有则通过 scan 尝试
                meta = plugin.scan(path)
                data.attrs["inventory"] = meta.to_inventory()
            except Exception as e:
                logger.warning(f"无法生成 Inventory: {e}")

        return data

    @classmethod
    def list_formats(cls) -> list[str]:
        """列出所有已注册的格式

        Returns:
            list[str]: 格式名称列表
        """
        cls._ensure_initialized()
        return list(cls._formats.keys())

    @classmethod
    def list_extensions(cls) -> dict[str, str]:
        """列出所有支持的文件扩展名及其对应格式

        Returns:
            dict[str, str]: 扩展名到格式名的映射
        """
        cls._ensure_initialized()
        ext_map = {}
        for name, plugin in cls._formats.items():
            for ext in plugin.file_extensions:
                ext_map[ext.lower()] = name
        return ext_map

    @classmethod
    def _ensure_initialized(cls) -> None:
        """确保内置格式已注册"""
        if cls._initialized:
            return

        # 延迟导入并注册内置格式 (按优先级顺序)
        # P0 优先级: 工业标准格式
        from .apsensing import APSensingFormatPlugin
        from .febus import FebusFormatPlugin
        from .prodml import PRODMLFormatPlugin
        from .silixa import SilixaFormatPlugin
        from .terra15 import Terra15FormatPlugin

        cls.register(PRODMLFormatPlugin())
        cls.register(SilixaFormatPlugin())
        cls.register(FebusFormatPlugin())
        cls.register(APSensingFormatPlugin())
        cls.register(Terra15FormatPlugin())

        # 通用格式
        from .dat import DATFormatPlugin
        from .h5 import H5FormatPlugin

        cls.register(H5FormatPlugin())
        cls.register(DATFormatPlugin())

        # 云原生和科学数据格式
        from .netcdf import NetCDFFormatPlugin
        from .zarr_format import ZarrFormatPlugin

        cls.register(ZarrFormatPlugin())
        cls.register(NetCDFFormatPlugin())

        # 地震学格式
        from .miniseed import MiniSEEDFormatPlugin
        from .segy import SEGYFormatPlugin

        cls.register(SEGYFormatPlugin())
        cls.register(MiniSEEDFormatPlugin())

        # 可选格式 (需要额外依赖)
        try:
            from .tdms import TDMSFormatPlugin

            cls.register(TDMSFormatPlugin())
        except ImportError:
            logger.debug("TDMS 格式不可用 (需要 nptdms)")

        cls._initialized = True
        logger.debug(f"初始化完成, 已注册 {len(cls._formats)} 种格式")

    @classmethod
    def reset(cls) -> None:
        """重置注册表 (主要用于测试)"""
        cls._formats.clear()
        cls._initialized = False


# 导出公共接口
__all__ = [
    "FormatPlugin",
    "FormatMetadata",
    "FormatRegistry",
]
