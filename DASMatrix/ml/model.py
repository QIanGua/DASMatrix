"""DASMatrix ML 推理层核心模块。

提供统一的模型加载与推理接口，支持 PyTorch 和 ONNX 后端。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch: Any = None
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort

    ORT_AVAILABLE = True
except ImportError:
    ort: Any = None
    ORT_AVAILABLE = False

logger = logging.getLogger(__name__)


class DASModel(ABC):
    """DAS 模型抽象基类。"""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model: Any = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """加载模型逻辑。"""
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """模型推理逻辑。"""
        pass


class TorchModel(DASModel):
    """PyTorch 模型封装。"""

    def _load_model(self):
        if torch is None:
            raise ImportError("PyTorch not installed. Please install 'torch'.")
        self.model = torch.load(self.model_path, map_location=self.device)
        self.model.eval()
        logger.info(f"Loaded PyTorch model from {self.model_path}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        with cast(Any, torch).no_grad():
            tensor_x = cast(Any, torch).from_numpy(x).to(self.device)
            output = self.model(tensor_x)
            return output.cpu().numpy()


class ONNXModel(DASModel):
    """ONNX 模型封装。"""

    def _load_model(self):
        if ort is None:
            raise ImportError("ONNXRuntime not installed. Please install 'onnxruntime'.")
        self.session = ort.InferenceSession(self.model_path)
        logger.info(f"Loaded ONNX model from {self.model_path}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: x})
        return np.asarray(output[0])
