"""DASMatrix API层，提供高级接口和DSL。"""

import warnings

from .dasframe import DASFrame
from .df import from_array, read
from .df import stream as _df_stream
from .spool import DASSpool, spool
from .stream import Stream


def stream(*args, **kwargs):
    """Return a placeholder stream frame."""
    return _df_stream(*args, **kwargs)


def stream_func(*args, **kwargs):
    """Deprecated alias for `stream()`."""
    warnings.warn("`stream_func` is deprecated; use `stream` instead.", DeprecationWarning, stacklevel=2)
    return stream(*args, **kwargs)


# 导出接口
__all__ = [
    "DASFrame",
    "read",
    "from_array",
    "stream",
    "stream_func",
    "DASSpool",
    "spool",
    "Stream",
]
