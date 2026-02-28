"""验证 DASMatrix AI 推理功能的简单示例脚本。"""

import numpy as np

import DASMatrix as dm
from DASMatrix.ml.pipeline import InferencePipeline


def mock_preprocess(x):
    """模拟预处理：调整维度。"""
    return x.astype(np.float32)


def main():
    # 1. 创建示例数据
    frame = dm.get_example_frame("event")
    print(f"Original frame shape: {frame.shape}")

    from typing import Any

    class SimpleModel(dm.ml.model.DASModel):
        def __init__(self):
            # 仅供测试使用，不加载真实模型
            self.model: Any = None
            self.device = "cpu"
            self.model_path = ""

        def _load_model(self):
            pass

        def predict(self, x):
            # 模拟输出：[Batch, 2] 的分类概率
            import torch

            return torch.tensor([[0.1, 0.9]])

        def eval(self):
            pass

    # 3. 初始化推理流水线
    # 注意：在真实环境下，你会使用 TorchModel("path.pth")
    # 这里我们绕过加载逻辑直接测试逻辑链路
    model = SimpleModel()

    # 封装到 InferencePipeline
    pipeline = InferencePipeline(
        model=model,  # 这里应为适配后的 DASModel 子类，此处仅为链路演示
        preprocess_fn=mock_preprocess,
    )

    # 4. 在 DASFrame 上直接调用预测
    # 正常流程会通过 predict(pipeline) 内部调用 pipeline.run(self)
    try:
        # 由于 SimpleModel 不是真正的 DASModel，我们直接调用 pipeline.run
        result = pipeline.run(frame)
        print(f"Inference result: {result}")
        print("AI Pipeline path verified successfully!")
    except Exception as e:
        print(f"Pipeline test failed: {e}")


if __name__ == "__main__":
    main()
