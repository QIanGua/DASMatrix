from .acquisition.das_reader import DASReader
from .api import DASFrame, DASSpool, from_array, read, spool, stream
from .config.sampling_config import SamplingConfig
from .examples import get_example_frame, get_example_spool, list_example_types
from .processing.das_processor import DASProcessor
from .units import Hz, get_quantity, get_unit, kHz, m, magnitude, ms, s
from .utils.time import to_datetime64, to_float, to_timedelta64
from .visualization.das_visualizer import DASVisualizer

__version__ = "0.1.0"

__all__ = [
    # 核心类
    "DASReader",
    "DASVisualizer",
    "DASProcessor",
    "SamplingConfig",
    "DASFrame",
    "DASSpool",
    # API 函数
    "read",
    "from_array",
    "stream",
    "spool",
    # 示例数据
    "get_example_frame",
    "get_example_spool",
    "list_example_types",
    # 时间工具
    "to_datetime64",
    "to_timedelta64",
    "to_float",
    # 单位系统
    "get_quantity",
    "get_unit",
    "magnitude",
    "m",
    "s",
    "ms",
    "Hz",
    "kHz",
]
