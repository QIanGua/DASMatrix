"""DASMatrix ML 推理流水线。

提供从 DASFrame 到 AI 预测结果的完整流水线。
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

import numpy as np

if TYPE_CHECKING:
    from ..api.dasframe import DASFrame
from .model import DASModel

logger = logging.getLogger(__name__)


class InferencePipeline:
    """DAS 推理流水线容器。"""

    def __init__(
        self, model: DASModel, preprocess_fn: Optional[Callable] = None, postprocess_fn: Optional[Callable] = None
    ):
        """初始化推理流水线。

        Args:
            model: 已初始化的 DASModel 实例。
            preprocess_fn: 可选的预处理函数，输入为 numpy 数组，输出为模型输入格式。
            postprocess_fn: 可选的后处理函数，输入为模型输出，输出为业务所需格式。
        """
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

    def run(self, frame: Union["DASFrame", np.ndarray]) -> Any:
        """执行推理。"""
        # 1. 获取原始数组
        data: np.ndarray
        if type(frame).__name__ == "DASFrame":
            # 延迟调用 collect 以避免运行时 isinstance 依赖
            data = getattr(frame, "collect")()
        else:
            data = cast(np.ndarray, frame)

        # 2. 预处理
        x = data
        if self.preprocess_fn:
            x = self.preprocess_fn(data)

        # 确保输入维度正确（例如：Batch, Channel, Time 或 Batch, Time, Channel）
        if hasattr(x, "ndim") and cast(Any, x).ndim == 2:
            x = cast(Any, x)[np.newaxis, ...]

        # 3. 推理
        logits = self.model.predict(cast(np.ndarray, x))

        # 4. 后处理
        result = logits
        if self.postprocess_fn:
            result = self.postprocess_fn(logits)

        return result
