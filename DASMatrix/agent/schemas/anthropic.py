"""Anthropic Tool Use 格式的 Tool Schema。

兼容 Anthropic Claude 3.x 系列模型的 tool use 接口。

使用示例:
    >>> from DASMatrix.agent.schemas import get_anthropic_tools
    >>> tools = get_anthropic_tools()
    >>> response = anthropic.messages.create(
    ...     model="claude-sonnet-4-20250514",
    ...     tools=tools,
    ...     messages=[...]
    ... )
"""

from typing import Any, Dict, List

ANTHROPIC_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "read_das_data",
        "description": "读取 DAS 数据文件。支持自动格式检测，可读取 H5, DAT, Zarr, PRODML, FEBUS 等多种格式。返回数据 ID 用于后续操作。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "数据文件路径或通配符模式，如 '/data/experiment.h5' 或 '/data/*.dat'",
                },
                "channels": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "可选，指定读取的通道索引列表",
                },
                "time_range": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "可选，时间范围 [开始秒, 结束秒]",
                },
                "format": {
                    "type": "string",
                    "enum": [
                        "h5",
                        "dat",
                        "zarr",
                        "prodml",
                        "febus",
                        "terra15",
                        "segy",
                        "miniseed",
                    ],
                    "description": "可选，强制指定格式，默认自动检测",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_das_files",
        "description": "列出目录中的 DAS 数据文件，显示文件格式和大小。",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "目录路径"},
                "pattern": {
                    "type": "string",
                    "description": "文件匹配模式，默认 '*'，如 '*.h5' 或 'exp_*.dat'",
                },
                "recursive": {"type": "boolean", "description": "是否递归搜索子目录"},
            },
            "required": ["directory"],
        },
    },
    {
        "name": "process_signal",
        "description": """对 DAS 数据应用信号处理流水线。

支持的操作:
- detrend: 去趋势
- bandpass: 带通滤波 (需指定 low 和 high 参数，单位 Hz)
- highpass: 高通滤波 (需指定 cutoff 参数，单位 Hz)
- lowpass: 低通滤波 (需指定 cutoff 参数，单位 Hz)
- normalize: 归一化 (可选 method: zscore 或 minmax)
- demean: 去均值
- stft: 短时傅立叶变换 (可选 nperseg 参数)
- fft: 傅立叶变换
- envelope: 包络提取

示例 operations:
[
  {"op": "detrend"},
  {"op": "bandpass", "low": 10, "high": 100},
  {"op": "normalize", "method": "zscore"}
]""",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_id": {
                    "type": "string",
                    "description": "输入数据的 ID，由 read_das_data 返回",
                },
                "operations": {
                    "type": "array",
                    "description": "处理操作列表，按顺序执行",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "description": "操作名称"},
                            "low": {
                                "type": "number",
                                "description": "带通滤波低频截止 (Hz)",
                            },
                            "high": {
                                "type": "number",
                                "description": "带通滤波高频截止 (Hz)",
                            },
                            "cutoff": {
                                "type": "number",
                                "description": "高通/低通滤波截止频率 (Hz)",
                            },
                            "order": {"type": "integer", "description": "滤波器阶数"},
                            "method": {"type": "string", "description": "归一化方法"},
                            "nperseg": {
                                "type": "integer",
                                "description": "STFT 窗口大小",
                            },
                        },
                        "required": ["op"],
                    },
                },
                "output_name": {
                    "type": "string",
                    "description": "可选，输出数据的描述性名称",
                },
            },
            "required": ["data_id", "operations"],
        },
    },
    {
        "name": "compute_spectrum",
        "description": "计算指定通道的频谱，返回主要频率成分和峰值。用于分析信号的频率特性。",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_id": {"type": "string", "description": "数据 ID"},
                "channel": {"type": "integer", "description": "通道索引，默认 0"},
                "window_size": {
                    "type": "integer",
                    "description": "FFT 窗口大小，默认 1024",
                },
                "overlap": {"type": "number", "description": "窗口重叠比例，默认 0.5"},
            },
            "required": ["data_id"],
        },
    },
    {
        "name": "detect_events",
        "description": "检测数据中的异常事件，如振动、撞击、泄漏等。基于能量阈值检测。",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_id": {"type": "string", "description": "数据 ID"},
                "threshold_db": {
                    "type": "number",
                    "description": "检测阈值 (dB)，默认 -30。值越大检测越敏感。",
                },
                "min_duration_ms": {
                    "type": "number",
                    "description": "最小事件持续时间 (毫秒)，默认 10",
                },
            },
            "required": ["data_id"],
        },
    },
    {
        "name": "get_data_stats",
        "description": "获取数据的统计信息，包括均值、标准差、最大最小值、RMS 等。",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_id": {"type": "string", "description": "数据 ID"},
                "per_channel": {
                    "type": "boolean",
                    "description": "是否按通道分别统计，默认 false",
                },
            },
            "required": ["data_id"],
        },
    },
    {
        "name": "create_visualization",
        "description": """生成 DAS 数据可视化图表。

图表类型:
- waterfall: 瀑布图/热力图，显示时间 x 通道的二维数据
- spectrum: 频谱图，显示频率成分分布
- waveform: 时域波形图，显示单通道信号随时间变化
- spectrogram: 时频谱图，显示信号的时频特性""",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_id": {"type": "string", "description": "数据 ID"},
                "plot_type": {
                    "type": "string",
                    "enum": ["waterfall", "spectrum", "waveform", "spectrogram"],
                    "description": "图表类型",
                },
                "output_path": {"type": "string", "description": "可选，图片保存路径"},
                "channels": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "可选，绘制的通道列表",
                },
                "time_range": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "可选，时间范围 [开始秒, 结束秒]",
                },
            },
            "required": ["data_id", "plot_type"],
        },
    },
    {
        "name": "list_session_objects",
        "description": "列出当前会话中的所有数据对象及其信息（如形状、来源、处理历史等）。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "delete_object",
        "description": "删除指定的数据对象以释放内存。",
        "input_schema": {
            "type": "object",
            "properties": {"data_id": {"type": "string", "description": "要删除的对象 ID"}},
            "required": ["data_id"],
        },
    },
    {
        "name": "assess_data_quality",
        "description": "评估数据质量，识别噪声特征。返回坏道列表、工频干扰判定、趋势项、信噪比估计等指标。",
        "input_schema": {
            "type": "object",
            "properties": {"data_id": {"type": "string", "description": "数据 ID"}},
            "required": ["data_id"],
        },
    },
    {
        "name": "apply_cleaning_recipe",
        "description": """应用预定义的清洗套餐。

可选套餐:
- standard_denoise: 标准去噪 (去趋势 + 1Hz 高通)
- remove_powerline: 去除工频干扰 (50Hz/60Hz)
- seismic_enhance: 地震信号增强 (1-100Hz 带通 + Z-score 归一化)""",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_id": {"type": "string", "description": "数据 ID"},
                "recipe_name": {
                    "type": "string",
                    "enum": ["standard_denoise", "remove_powerline", "seismic_enhance"],
                    "description": "清洗套餐名称",
                },
            },
            "required": ["data_id", "recipe_name"],
        },
    },
]


def get_anthropic_tools() -> List[Dict[str, Any]]:
    """获取 Anthropic 格式的工具定义列表。

    Returns:
        符合 Anthropic tool use 格式的工具列表
    """
    return ANTHROPIC_TOOL_SCHEMAS
