"""Tool Schema 定义模块。

提供 OpenAI 和 Anthropic 格式的 Tool Schema，供 AI Agent 调用。
"""

from .anthropic import ANTHROPIC_TOOL_SCHEMAS, get_anthropic_tools
from .openai import OPENAI_TOOL_SCHEMAS, get_openai_tools

__all__ = [
    "get_openai_tools",
    "get_anthropic_tools",
    "OPENAI_TOOL_SCHEMAS",
    "ANTHROPIC_TOOL_SCHEMAS",
]
