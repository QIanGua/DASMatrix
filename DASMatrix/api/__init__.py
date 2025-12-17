"""DASMatrix API层，提供高级接口和DSL。"""

# 先导入df模块，再导入DASFrame类
from .dasframe import DASFrame
from .df import from_array, read, stream

# 导出接口
__all__ = ["DASFrame", "read", "from_array", "stream"]
