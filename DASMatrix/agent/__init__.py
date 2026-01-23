"""DASMatrix AI Agent 集成模块。

提供 AI Agent (如 Claude, GPT) 可调用的工具函数，
支持自然语言驱动的 DAS 数据分析流程。

使用示例:
    >>> from DASMatrix.agent import DASAgentTools
    >>> tools = DASAgentTools()
    >>> data = tools.read_das_data("data.h5")
    >>> result = tools.process_signal(data.id, [{"op": "bandpass", "low": 10, "high": 100}])
"""

from .schemas import get_anthropic_tools, get_openai_tools
from .session import AgentSession
from .tools import DASAgentTools

__all__ = [
    "DASAgentTools",
    "AgentSession",
    "get_openai_tools",
    "get_anthropic_tools",
]
