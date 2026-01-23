import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd

from ..acquisition.formats import FormatMetadata, FormatRegistry
from .dasframe import DASFrame


class DASSpool:
    """DAS 数据集管理器, 统一多文件访问接口

    功能:
    - 虚拟合并多个文件为单一数据集视图
    - 支持时间/空间范围查询
    - 支持分块迭代处理
    - 支持索引缓存加速二次访问
    """

    def __init__(
        self,
        path: Union[str, Path, List[Path]],
        format: Optional[str] = None,
    ):
        """初始化 Spool

        Args:
            path: 单文件路径、目录路径、通配符模式或文件列表
            format: 强制指定格式, None 表示自动检测
        """
        self._format = format
        self._files: List[Path] = []
        self._meta_cache: Dict[Path, FormatMetadata] = {}

        # 索引 DataFrame (path, start_time, end_time, n_samples, ...)
        self._index: Optional[pd.DataFrame] = None

        self._resolve_paths(path)

    def _resolve_paths(self, path: Union[str, Path, List[Path]]) -> None:
        """解析路径"""
        if isinstance(path, list):
            self._files = [Path(p) for p in path]
        elif isinstance(path, (str, Path)):
            path = Path(path)
            # Check for glob pattern in string path or name
            if "*" in str(path) or "?" in str(path):
                # Glob pattern
                parent = path.parent
                # if parent is just the pattern (e.g. "*.h5"), parent is "."
                if not parent.exists() and not str(parent) == ".":
                    # The parent part itself might be part of glob,
                    # e.g. "data_*/file.h5"
                    # For simplicity, assume simple globs or recursive search
                    # from a base dir. If users pass "data/*.h5", parent is "data".
                    pass

                # Use glob on the parent if it exists, otherwise assume current dir
                search_dir = parent if parent.name else Path(".")
                pattern = path.name
                self._files = sorted(list(search_dir.glob(pattern)))

            elif path.is_dir():
                # Directory, scan all supported extensions
                self._files = []
                ext_map = FormatRegistry.list_extensions()
                # Use set of extensions to avoid duplicates
                exts = set(ext_map.keys())
                for ext in exts:
                    # Case insensitive search is tricky with glob on linux,
                    # assume lowercase ext or provided ext
                    self._files.extend(path.glob(f"*{ext}"))
                    self._files.extend(path.glob(f"*{ext.upper()}"))
                self._files = sorted(list(set(self._files)))
            else:
                # Single file
                if path.exists():
                    self._files = [path]
                else:
                    # Might be a glob pattern that didn't match existing path check?
                    # Re-check for glob chars just in case user passed a string path
                    # that doesn't exist
                    if "*" in str(path) or "?" in str(path):
                        # Logic duplicated above but for completeness
                        parent = path.parent if path.parent.name else Path(".")
                        pattern = path.name
                        self._files = sorted(list(parent.glob(pattern)))
                    else:
                        warnings.warn(f"File not found: {path}")
                        self._files = []

        if not self._files and path:
            warnings.warn(f"No files found for path: {path}")

    def update(self) -> "DASSpool":
        """更新文件索引 (增量扫描新文件)"""
        data_list = []

        for p in self._files:
            if p not in self._meta_cache:
                try:
                    meta = FormatRegistry.scan(p, format_name=self._format)
                    self._meta_cache[p] = meta
                except Exception:
                    # Skip invalid files or errors during scan
                    continue

            meta = self._meta_cache[p]
            # Try to parse start time if available
            start_time = None
            if meta.start_time:
                try:
                    start_time = pd.to_datetime(meta.start_time)
                except Exception:
                    pass

            # Estimate end time
            end_time = None
            if start_time is not None and meta.sampling_rate and meta.n_samples:
                duration = meta.n_samples / meta.sampling_rate
                end_time = start_time + pd.Timedelta(seconds=duration)

            data_list.append(
                {
                    "path": p,
                    "start_time": start_time,
                    "end_time": end_time,
                    "n_samples": meta.n_samples,
                    "n_channels": meta.n_channels,
                    "sampling_rate": meta.sampling_rate,
                    "channel_spacing": meta.channel_spacing,
                }
            )

        if data_list:
            self._index = pd.DataFrame(data_list).sort_values("start_time")
        else:
            self._index = pd.DataFrame(columns=pd.Index(["path", "start_time", "end_time"]))

        return self

    def select(
        self,
        time: Optional[Tuple[Union[str, datetime], Union[str, datetime]]] = None,
        distance: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> "DASSpool":
        """延迟筛选子集 (返回新 Spool, 不加载数据)"""
        if self._index is None:
            self.update()

        if self._index is None or self._index.empty:
            return self

        new_spool = DASSpool([], format=self._format)
        # Copy cache
        new_spool._meta_cache = self._meta_cache.copy()

        df = self._index.copy()

        if time:
            t_start, t_end = pd.to_datetime(time[0]), pd.to_datetime(time[1])
            # Filter files that overlap with [t_start, t_end]
            # Handle cases where start_time might be NaT (no time info in file)
            # If no time info, we might keep them or drop them.
            # For now, drop rows where time is NaT if filtering by time.
            df = df.dropna(subset=["start_time", "end_time"])

            mask = (df["start_time"] < t_end) & (df["end_time"] > t_start)
            df = df[mask]

        new_spool._index = df
        new_spool._files = df["path"].tolist()

        return new_spool

    def chunk(
        self,
        time: Optional[Union[str, int]] = None,
        overlap: Optional[Union[str, int]] = 0,
    ) -> Iterator["DASFrame"]:
        """指定输出分块策略 (目前按文件迭代)"""
        # Phase 1 simple implementation: Yield DASFrame for each file in index
        if self._index is None:
            self.update()

        if self._index is None or self._index.empty:
            return

        for _, row in self._index.iterrows():
            da = FormatRegistry.read(row["path"], format_name=self._format)
            # Ensure fs and dx are passed if not in attrs
            fs = row["sampling_rate"]
            dx = row["channel_spacing"] or 1.0

            # Create DASFrame
            # Note: DASFrame expects data with (time, distance) dims or handled inside
            frame = DASFrame(da, fs=fs, dx=dx)
            yield frame

    def __iter__(self) -> Iterator["DASFrame"]:
        """迭代返回 DASFrame 对象"""
        return self.chunk()

    def to_frame(self) -> "DASFrame":
        """将 Spool 中的所有文件虚拟合并为一个连续的 DASFrame。

        基于 Dask 延迟加载，不会立即读取所有数据到内存。
        """
        if self._index is None:
            self.update()

        if self._index is None or self._index.empty:
            raise ValueError("Spool 为空，无法转换为 DASFrame")

        # 读取所有 DataArrays
        data_arrays = []
        for _, row in self._index.iterrows():
            da = FormatRegistry.read(row["path"], format_name=self._format)
            data_arrays.append(da)

        # 沿着 time 维度合并
        # 注意：这里假设所有文件的通道数和采样率一致
        # TODO: 处理采样率不一致的情况（重采样）或通道对齐
        import xarray as xr

        combined_da = xr.concat(data_arrays, dim="time")
        # Standardize dimension name to distance if it is channel
        if "channel" in combined_da.dims:
            combined_da = combined_da.rename({"channel": "distance"})
        combined_da = combined_da.chunk({"time": "auto", "distance": -1})

        # 获取基础参数
        first_row = self._index.iloc[0]
        fs = first_row["sampling_rate"]
        dx = first_row["channel_spacing"] or 1.0

        # 注入 Inventory 信息（如果有的话）
        metadata = {}
        if "inventory" in combined_da.attrs:
            metadata["inventory"] = combined_da.attrs["inventory"]

        return DASFrame(combined_da, fs=fs, dx=dx, **metadata)

    def __len__(self) -> int:
        """返回文件数量"""
        return len(self._files)

    def __getitem__(self, idx: int) -> "DASFrame":
        """索引访问"""
        if self._index is None:
            self.update()

        if self._index is None or self._index.empty:
            raise IndexError("Spool is empty")

        row = self._index.iloc[idx]
        da = FormatRegistry.read(row["path"], format_name=self._format)
        return DASFrame(da, fs=row["sampling_rate"], dx=row["channel_spacing"] or 1.0)


# 便捷函数
def spool(path: Union[str, Path, List[Path]], **kwargs) -> DASSpool:
    """创建 Spool 的便捷函数"""
    return DASSpool(path, **kwargs)
