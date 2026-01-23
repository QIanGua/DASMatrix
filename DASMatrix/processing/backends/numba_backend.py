"""Numba 后端实现，负责 JIT 编译和执行融合算子。"""

from typing import List

import numba
import numpy as np

from ...core.computation_graph import FusionNode


class NumbaBackend:
    """Numba 高性能计算后端。"""

    def __init__(self):
        self._cache = {}  # 缓存编译后的内核

    def execute(self, node: FusionNode, data: np.ndarray) -> np.ndarray:
        """执行融合节点。

        Args:
            node: FusionNode
            data: 输入数据 (numpy array)

        Returns:
            np.ndarray: 计算结果
        """
        # 1. 预计算阶段 (Pre-computation for Reduction Ops)
        # 某些算子如 detrend, demean 需要预先扫描数据获取全局参数
        aux_params = self._prepare_aux_params(node, data)

        # 2. 获取/编译内核
        kernel_key = self._get_kernel_signature(node)
        if kernel_key not in self._cache:
            self._cache[kernel_key] = self._compile_kernel(node, has_aux=bool(aux_params))

        kernel_func = self._cache[kernel_key]

        # 3. 准备输出数组
        out = np.empty_like(data)

        # 4. 执行内核
        # 参数: (inp, out, *aux_values)
        args = [data, out]
        if aux_params:
            args.extend(aux_params)

        kernel_func(*args)

        return out

    def _prepare_aux_params(self, node: FusionNode, data: np.ndarray) -> List[np.ndarray]:
        """使用 Numba 加速的统计预计算。"""
        n_samples, n_channels = data.shape
        params = []

        # 检查是否需要统计信息
        needs_detrend = any(op.operation == "detrend" for op in node.fused_nodes)
        needs_demean = any(op.operation == "demean" for op in node.fused_nodes)
        needs_normalize = any(op.operation == "normalize" for op in node.fused_nodes)

        if needs_detrend or needs_demean or needs_normalize:
            # 调用高性能 JIT 辅助函数
            return self._compute_stats_numba(data, needs_detrend, needs_demean, needs_normalize)

        return params

    @staticmethod
    @numba.njit(parallel=True, fastmath=True)
    def _compute_stats_numba(data, compute_detrend: bool, compute_demean: bool, compute_normalize: bool):
        n_samples, n_channels = data.shape
        dtype = data.dtype

        results = []

        if compute_detrend:
            # Linear regression stats
            n = n_samples
            sum_x = n * (n - 1) / 2
            sum_x2 = n * (n - 1) * (2 * n - 1) / 6
            denom = n * sum_x2 - sum_x**2

            ks = np.zeros(n_channels, dtype=dtype)
            bs = np.zeros(n_channels, dtype=dtype)

            for j in numba.prange(n_channels):  # type: ignore
                s_y = 0.0
                s_xy = 0.0
                for i in range(n_samples):
                    val = data[i, j]
                    s_y += val
                    s_xy += i * val

                k = (n * s_xy - sum_x * s_y) / denom
                b = (s_y - k * sum_x) / n
                ks[j] = k
                bs[j] = b

            results.append(ks)
            results.append(bs)

        if compute_normalize:
            # Z-score needs mean and std
            means = np.zeros(n_channels, dtype=dtype)
            stds = np.zeros(n_channels, dtype=dtype)
            for j in numba.prange(n_channels):  # type: ignore
                # Welford algorithm for single-pass mean/std
                m = 0.0
                m2 = 0.0
                for i in range(n_samples):
                    val = data[i, j]
                    delta = val - m
                    m += delta / (i + 1)
                    delta2 = val - m
                    m2 += delta * delta2

                means[j] = m
                variance = m2 / n_samples
                stds[j] = np.sqrt(variance) if variance > 0 else 1.0

            results.append(means)
            results.append(stds)

        elif compute_demean:
            means = np.zeros(n_channels, dtype=dtype)
            for j in numba.prange(n_channels):  # type: ignore
                s_y = 0.0
                for i in range(n_samples):
                    s_y += data[i, j]
                means[j] = s_y / n_samples
            results.append(means)

        return results

    def _get_kernel_signature(self, node: FusionNode) -> str:
        sig = "fuse"
        for op in node.fused_nodes:
            sig += f"_{op.operation}"
        return sig

    def _compile_kernel(self, node: FusionNode, has_aux: bool):
        ops_code = []
        aux_idx = 0  # 追踪辅助参数索引

        # 辅助参数名列表 (在 kernel 签名中使用)
        aux_arg_names = []

        for op in node.fused_nodes:
            if op.operation == "detrend":
                # 需要 slope 和 intercept
                k_name = f"aux_{aux_idx}"
                b_name = f"aux_{aux_idx + 1}"
                aux_arg_names.extend([k_name, b_name])
                aux_idx += 2

                # val = val - (k[j] * i + b[j])
                ops_code.append(f"val = val - ({k_name}[j] * i + {b_name}[j])")

            elif op.operation == "demean":
                m_name = f"aux_{aux_idx}"
                aux_arg_names.extend([m_name])
                aux_idx += 1
                ops_code.append(f"val = val - {m_name}[j]")

            elif op.operation == "abs":
                ops_code.append("val = abs(val)")

            elif op.operation == "scale":
                factor = op.kwargs.get("factor", 1.0)
                ops_code.append(f"val = val * {factor}")

            elif op.operation == "bandpass":
                # Placeholder: Pass-through
                # TODO: Implement IIR/FIR filter state or sosfilt
                pass

            elif op.operation == "normalize":
                # Assume z-score normalization (uses mean and std from aux)
                m_name = f"aux_{aux_idx}"
                s_name = f"aux_{aux_idx + 1}"
                aux_arg_names.extend([m_name, s_name])
                aux_idx += 2
                ops_code.append(f"val = (val - {m_name}[j]) / {s_name}[j]")

            # TODO: Add filter support (requires stateful loop or simple FIR/IIR)

        kernel_body = "\n            ".join(ops_code)

        # 构建函数签名
        base_args = ["inp", "out"]
        all_args = base_args + aux_arg_names
        args_str = ", ".join(all_args)

        code = f"""
def fused_kernel({args_str}):
    rows, cols = inp.shape
    for i in prange(rows):
        for j in range(cols):
            val = inp[i, j]
            {kernel_body}
            out[i, j] = val
"""

        global_scope = {
            "numba": numba,
            "prange": numba.prange,
            "abs": abs,
        }

        exec(code, global_scope)
        func = global_scope["fused_kernel"]

        return numba.njit(parallel=True, fastmath=True)(func)
