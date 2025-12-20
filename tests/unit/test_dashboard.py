"""DASDashboard 单元测试"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # 使用非交互式后端进行测试

from DASMatrix.visualization import DASDashboard
from DASMatrix.config import VisualizationConfig


class TestDASDashboard:
    """测试 DASDashboard 类"""

    def test_init(self):
        """测试初始化"""
        dashboard = DASDashboard(n_channels=64, fs=1000)
        assert dashboard.n_channels == 64
        assert dashboard.fs == 1000
        assert dashboard.fig is not None
        plt.close(dashboard.fig)

    def test_update(self):
        """测试数据更新"""
        dashboard = DASDashboard(n_channels=64, fs=1000)
        chunk = np.random.randn(100, 64)
        events = np.zeros(100)
        events[50] = 1  # 模拟一个事件
        
        # 应该能正常更新而不报错
        dashboard.update(chunk, events=events)
        
        # 检查内部状态更新
        assert len(dashboard.metrics_history["times"]) == 1
        assert len(dashboard.event_log) == 1
        plt.close(dashboard.fig)

    def test_custom_config(self):
        """测试自定义配置"""
        config = VisualizationConfig.for_screen()
        dashboard = DASDashboard(n_channels=32, fs=500, config=config)
        assert dashboard.config.dpi == config.dpi
        plt.close(dashboard.fig)
