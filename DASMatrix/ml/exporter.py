"""DASMatrix 模型转换工具。

支持将训练好的 PyTorch 模型导出为 ONNX 格式。
"""

import logging
from typing import Any, List, Optional

try:
    import torch
except ImportError:
    torch: Any = None

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: Any,
    dummy_input: Any,
    export_path: str,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    opset_version: int = 14,
):
    """将 PyTorch 模型导出为 ONNX 格式。"""
    if torch is None:
        raise ImportError("PyTorch not installed.")

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}},
    )

    logger.info(f"Model exported to {export_path}")
