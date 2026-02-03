"""混合执行引擎模块。

本模块提供 DASFrame 延迟计算的核心执行引擎，负责：
- 计算图优化（算子融合等）
- 调度不同后端（NumPy/SciPy、Numba JIT）
- 递归执行计算图节点
"""

from typing import Any, List

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import median_filter as scipy_median_filter

from ..core.computation_graph import (
    ComputationGraph,
    FusionNode,
    Node,
    OperationNode,
    SourceNode,
)
from .backends.numba_backend import NumbaBackend
from .planner.optimizer import ExecutionPlanner


class HybridEngine:
    """混合执行引擎，协调优化器和多后端执行。

    HybridEngine 是 DASFrame 延迟计算的核心组件，负责：
    1. 调用 ExecutionPlanner 优化计算图
    2. 根据节点类型调度合适的后端
    3. 递归执行计算图并缓存中间结果

    Attributes:
        planner: 执行计划优化器
        numba_backend: Numba JIT 后端
    """

    def __init__(self) -> None:
        """初始化混合执行引擎。"""
        self.planner = ExecutionPlanner()
        self.numba_backend = NumbaBackend()

    def compute(self, graph: ComputationGraph) -> Any:
        """执行计算图并返回结果。

        这是引擎的主入口点，负责优化和执行整个计算图。

        Args:
            graph: 要执行的计算图

        Returns:
            Any: 计算图根节点的计算结果（通常为 NumPy 数组）

        Raises:
            ValueError: 当计算图为空时
        """
        opt_graph = self.planner.optimize(graph)

        if not opt_graph.root:
            raise ValueError("Empty computation graph")

        return self._execute_node(opt_graph.root)

    def _execute_node(self, node: Node) -> Any:
        """递归执行单个节点。

        根据节点类型分发到合适的后端执行。支持结果缓存以避免重复计算。

        Args:
            node: 要执行的节点

        Returns:
            Any: 节点的计算结果
        """
        if node.computed:
            return node.result

        input_data: List[Any] = []
        for inp in node.inputs:
            input_data.append(self._execute_node(inp))

        result: Any = None

        if isinstance(node, SourceNode):
            result = node.data

        elif isinstance(node, FusionNode):
            data = input_data[0]
            result = self.numba_backend.execute(node, data)

        elif isinstance(node, OperationNode):
            data = input_data[0] if input_data else None
            if data is None:
                raise ValueError(f"Input data for node {node.name} is None")
            result = self._execute_single_op(node, data)

        node.result = result
        node.computed = True
        return result

    def _execute_single_op(self, node: OperationNode, data: np.ndarray) -> Any:
        """执行单个操作节点。"""
        op = node.operation
        kwargs = node.kwargs or {}

        if op == "slice":
            t_slice = kwargs.get("t", slice(None))
            x_slice = kwargs.get("x", slice(None))
            return data[t_slice, x_slice]

        elif op == "detrend":
            return scipy_signal.detrend(data, axis=0)

        elif op == "demean":
            return data - np.mean(data, axis=0, keepdims=True)

        elif op == "abs":
            return np.abs(data)

        elif op == "scale":
            factor = kwargs.get("factor", 1.0)
            return data * factor

        elif op == "normalize":
            method = kwargs.get("method", "minmax")
            if method == "zscore":
                mean = np.mean(data, axis=0, keepdims=True)
                std = np.std(data, axis=0, keepdims=True)
                std = np.where(std == 0, 1, std)  # 避免除以零
                return (data - mean) / std
            else:  # minmax
                min_val = np.min(data, axis=0, keepdims=True)
                max_val = np.max(data, axis=0, keepdims=True)
                range_val = max_val - min_val
                range_val = np.where(range_val == 0, 1, range_val)
                return 2 * (data - min_val) / range_val - 1

        elif op == "bandpass":
            low = kwargs.get("low")
            high = kwargs.get("high")
            order = kwargs.get("order", 4)
            fs = kwargs.get("fs", 1000)
            nyq = fs / 2
            sos = scipy_signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
            return scipy_signal.sosfiltfilt(sos, data, axis=0)

        elif op == "lowpass":
            cutoff = kwargs.get("cutoff")
            order = kwargs.get("order", 4)
            fs = kwargs.get("fs", 1000)
            nyq = fs / 2
            sos = scipy_signal.butter(order, cutoff / nyq, btype="low", output="sos")
            return scipy_signal.sosfiltfilt(sos, data, axis=0)

        elif op == "highpass":
            cutoff = kwargs.get("cutoff")
            order = kwargs.get("order", 4)
            fs = kwargs.get("fs", 1000)
            nyq = fs / 2
            sos = scipy_signal.butter(order, cutoff / nyq, btype="high", output="sos")
            return scipy_signal.sosfiltfilt(sos, data, axis=0)

        elif op == "notch":
            freq = kwargs.get("freq")
            Q = kwargs.get("Q", 30)
            fs = kwargs.get("fs", 1000)
            w0 = freq / (fs / 2)
            b, a = scipy_signal.iirnotch(w0, Q)
            return scipy_signal.filtfilt(b, a, data, axis=0)

        elif op == "median_filter":
            k = kwargs.get("k", 5)
            axis = kwargs.get("axis", "time")
            if axis == "time":
                size = (k, 1)
            else:
                size = (1, k)
            return scipy_median_filter(data, size=size)

        elif op == "fft":
            # Match DASFrame.fft(): real FFT magnitude along time axis
            return np.abs(np.fft.rfft(data, axis=0))

        elif op == "stft":
            nperseg = kwargs.get("nperseg", 256)
            noverlap = kwargs.get("noverlap", nperseg // 2)
            fs = kwargs.get("fs", 1000)
            window = kwargs.get("window", "hann")
            # 对每个通道计算 STFT
            n_channels = data.shape[1] if data.ndim > 1 else 1
            results = []
            for ch in range(n_channels):
                ch_data = data[:, ch] if data.ndim > 1 else data
                f, t, Zxx = scipy_signal.stft(
                    ch_data,
                    fs=fs,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    window=window,
                )
                results.append(np.abs(Zxx))
            # 返回 (freq, time, channel)
            return np.stack(results, axis=-1)

        elif op == "hilbert":
            return scipy_signal.hilbert(data, axis=0)

        elif op == "envelope":
            analytic = scipy_signal.hilbert(data, axis=0)
            return np.abs(analytic)

        elif op == "fk_filter":
            v_min = kwargs.get("v_min")
            v_max = kwargs.get("v_max")
            dx = kwargs.get("dx", 1.0)
            fs = kwargs.get("fs", 1000)  # Should verify if fs is passed or available

            # FK Transform
            nt, nx = data.shape
            fk = np.fft.fftshift(np.fft.fft2(data))

            # Axes
            freqs = np.fft.fftshift(np.fft.fftfreq(nt, 1.0 / fs))
            k = np.fft.fftshift(np.fft.fftfreq(nx)) / dx

            # Meshgrid
            K, F = np.meshgrid(k, freqs)

            # Masking
            with np.errstate(divide="ignore", invalid="ignore"):
                V = F / K

            mask = np.ones_like(fk, dtype=bool)
            if v_min is not None:
                mask &= (np.abs(V) >= abs(v_min)) | np.isnan(V)
            if v_max is not None:
                mask &= (np.abs(V) <= abs(v_max)) | np.isnan(V)

            fk_filtered = fk * mask

            # Inverse Transform
            return np.fft.ifft2(np.fft.ifftshift(fk_filtered)).real

        else:
            raise NotImplementedError(f"Unsupported operation: {op}")
