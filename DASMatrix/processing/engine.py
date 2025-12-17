"""混合执行引擎，负责协调 Planner 和 Backends。"""

import numpy as np
from typing import Any, Union
from scipy import signal as scipy_signal
from scipy.ndimage import median_filter as scipy_median_filter

from ..core.computation_graph import ComputationGraph, FusionNode, Node, SourceNode, NodeDomain, OperationNode
from .planner.optimizer import ExecutionPlanner
from .backends.numba_backend import NumbaBackend
from .backends.polars_backend import PolarsBackend


class HybridEngine:
    """混合张量执行引擎。"""

    def __init__(self):
        self.planner = ExecutionPlanner()
        self.numba_backend = NumbaBackend()
        # PolarsBackend 通常需要绑定特定的 DataFrame，这里可能动态创建或管理
        
    def compute(self, graph: ComputationGraph) -> Any:
        """执行计算图。"""
        
        # 1. 优化图 (算子融合)
        opt_graph = self.planner.optimize(graph)
        
        if not opt_graph.root:
            raise ValueError("Empty computation graph")

        # 2. 执行 (简单递归或线性执行)
        # 注意: 这里简化实现，假设已经线性化且单一输出
        return self._execute_node(opt_graph.root)

    def _execute_node(self, node: Node) -> Any:
        """递归执行节点。"""
        if node.computed:
            return node.result

        # 先计算输入 data
        input_data = []
        for inp in node.inputs:
            input_data.append(self._execute_node(inp))
            
        result = None
        
        # 根据节点类型分发
        if isinstance(node, SourceNode):
            result = node.data
            
        elif isinstance(node, FusionNode):
            # 信号域融合节点 -> Numba Backend
            # 假设单输入 (data from prev node)
            data = input_data[0]
            result = self.numba_backend.execute(node, data)
            
        elif isinstance(node, OperationNode):
            # 未被融合的独立节点 - 使用 NumPy/SciPy 回退
            data = input_data[0] if input_data else None
            result = self._execute_single_op(node, data)
                
        node.result = result
        node.computed = True
        return result

    def _execute_single_op(self, node: OperationNode, data: np.ndarray) -> Any:
        """执行单个操作节点。"""
        op = node.operation
        kwargs = node.kwargs
        
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
            sos = scipy_signal.butter(order, [low/nyq, high/nyq], btype='band', output='sos')
            return scipy_signal.sosfiltfilt(sos, data, axis=0)
        
        elif op == "lowpass":
            cutoff = kwargs.get("cutoff")
            order = kwargs.get("order", 4)
            fs = kwargs.get("fs", 1000)
            nyq = fs / 2
            sos = scipy_signal.butter(order, cutoff/nyq, btype='low', output='sos')
            return scipy_signal.sosfiltfilt(sos, data, axis=0)
        
        elif op == "highpass":
            cutoff = kwargs.get("cutoff")
            order = kwargs.get("order", 4)
            fs = kwargs.get("fs", 1000)
            nyq = fs / 2
            sos = scipy_signal.butter(order, cutoff/nyq, btype='high', output='sos')
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
            return np.abs(np.fft.fft(data, axis=0))
        
        elif op == "stft":
            nperseg = kwargs.get("nperseg", 256)
            noverlap = kwargs.get("noverlap", nperseg // 2)
            fs = kwargs.get("fs", 1000)
            # 对每个通道计算 STFT
            n_channels = data.shape[1] if data.ndim > 1 else 1
            results = []
            for ch in range(n_channels):
                ch_data = data[:, ch] if data.ndim > 1 else data
                f, t, Zxx = scipy_signal.stft(ch_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
                results.append(np.abs(Zxx))
            # 返回 (freq, time, channel)
            return np.stack(results, axis=-1)
        
        elif op == "hilbert":
            return scipy_signal.hilbert(data, axis=0)
        
        elif op == "envelope":
            analytic = scipy_signal.hilbert(data, axis=0)
            return np.abs(analytic)
        
        else:
            raise NotImplementedError(f"Unsupported operation: {op}")

