"""OpenAI Function Calling 格式的 Tool Schema。

兼容 OpenAI GPT-4, GPT-4o 等模型的 function calling 接口。

使用示例:
    >>> from DASMatrix.agent.schemas import get_openai_tools
    >>> tools = get_openai_tools()
    >>> response = openai.chat.completions.create(
    ...     model="gpt-4o",
    ...     tools=tools,
    ...     messages=[...]
    ... )
"""

from typing import Any, Dict, List

OPENAI_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_das_data",
            "description": "读取 DAS 数据文件。支持自动格式检测，可读取 H5, DAT, Zarr, PRODML, FEBUS 等多种格式。返回数据 ID 用于后续操作。",
            "parameters": {
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
                        "minItems": 2,
                        "maxItems": 2,
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
    },
    {
        "type": "function",
        "function": {
            "name": "list_das_files",
            "description": "列出目录中的 DAS 数据文件，显示文件格式和大小。",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "目录路径"},
                    "pattern": {
                        "type": "string",
                        "description": "文件匹配模式，默认 '*'，如 '*.h5' 或 'exp_*.dat'",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "是否递归搜索子目录",
                    },
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_signal",
            "description": "对 DAS 数据应用信号处理流水线。支持的操作包括: detrend(去趋势), bandpass(带通滤波), highpass(高通滤波), lowpass(低通滤波), normalize(归一化), stft(短时傅立叶变换), fft(傅立叶变换), envelope(包络提取)。",
            "parameters": {
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
                                "op": {
                                    "type": "string",
                                    "enum": [
                                        "detrend",
                                        "bandpass",
                                        "highpass",
                                        "lowpass",
                                        "normalize",
                                        "stft",
                                        "fft",
                                        "envelope",
                                        "demean",
                                    ],
                                    "description": "操作名称",
                                },
                                "low": {
                                    "type": "number",
                                    "description": "带通/高通滤波的低频截止 (Hz)",
                                },
                                "high": {
                                    "type": "number",
                                    "description": "带通/低通滤波的高频截止 (Hz)",
                                },
                                "cutoff": {
                                    "type": "number",
                                    "description": "高通/低通滤波的截止频率 (Hz)",
                                },
                                "order": {
                                    "type": "integer",
                                    "description": "滤波器阶数，默认 4",
                                },
                                "method": {
                                    "type": "string",
                                    "enum": ["zscore", "minmax"],
                                    "description": "归一化方法",
                                },
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
    },
    {
        "type": "function",
        "function": {
            "name": "compute_spectrum",
            "description": "计算指定通道的频谱，返回主要频率成分和峰值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {"type": "string", "description": "数据 ID"},
                    "channel": {"type": "integer", "description": "通道索引，默认 0"},
                    "window_size": {
                        "type": "integer",
                        "description": "FFT 窗口大小，默认 1024",
                    },
                    "overlap": {
                        "type": "number",
                        "description": "窗口重叠比例，默认 0.5",
                    },
                },
                "required": ["data_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_events",
            "description": "检测数据中的异常事件，如振动、撞击等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {"type": "string", "description": "数据 ID"},
                    "threshold_db": {
                        "type": "number",
                        "description": "检测阈值 (dB)，默认 -30",
                    },
                    "min_duration_ms": {
                        "type": "number",
                        "description": "最小事件持续时间 (毫秒)，默认 10",
                    },
                },
                "required": ["data_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_data_stats",
            "description": "获取数据的统计信息，包括均值、标准差、最大最小值等。",
            "parameters": {
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
    },
    {
        "type": "function",
        "function": {
            "name": "create_visualization",
            "description": "生成 DAS 数据可视化图表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {"type": "string", "description": "数据 ID"},
                    "plot_type": {
                        "type": "string",
                        "enum": ["waterfall", "spectrum", "waveform", "spectrogram"],
                        "description": "图表类型: waterfall(瀑布图), spectrum(频谱图), waveform(波形图), spectrogram(时频图)",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "可选，图片保存路径",
                    },
                    "channels": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "可选，绘制的通道",
                    },
                    "time_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "可选，时间范围 [开始秒, 结束秒]",
                    },
                },
                "required": ["data_id", "plot_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_session_objects",
            "description": "列出当前会话中的所有数据对象及其信息。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_object",
            "description": "删除指定的数据对象以释放内存。",
            "parameters": {
                "type": "object",
                "properties": {"data_id": {"type": "string", "description": "要删除的对象 ID"}},
                "required": ["data_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_data_quality",
            "description": "评估数据质量，识别噪声特征，如工频干扰、坏道、趋势项等。",
            "parameters": {
                "type": "object",
                "properties": {"data_id": {"type": "string", "description": "数据 ID"}},
                "required": ["data_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_cleaning_recipe",
            "description": "应用预定义的清洗套餐。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_id": {"type": "string", "description": "数据 ID"},
                    "recipe_name": {
                        "type": "string",
                        "enum": [
                            "standard_denoise",
                            "remove_powerline",
                            "seismic_enhance",
                        ],
                        "description": "套餐名称: standard_denoise(去趋势+高通), remove_powerline(去除50/60Hz), seismic_enhance(地震增强)",
                    },
                },
                "required": ["data_id", "recipe_name"],
            },
        },
    },
]


def get_openai_tools() -> List[Dict[str, Any]]:
    """获取 OpenAI 格式的工具定义列表。

    Returns:
        符合 OpenAI function calling 格式的工具列表
    """
    return OPENAI_TOOL_SCHEMAS
