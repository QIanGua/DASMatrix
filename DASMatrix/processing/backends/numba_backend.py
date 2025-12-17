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
            self._cache[kernel_key] = self._compile_kernel(
                node, has_aux=bool(aux_params)
            )

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

    def _prepare_aux_params(
        self, node: FusionNode, data: np.ndarray
    ) -> List[np.ndarray]:
        """为需要归约的算子准备参数 (如 mean, trend)。"""
        params = []
        n_samples, n_channels = data.shape

        for op in node.fused_nodes:
            if op.operation == "detrend":
                # Linear Detrend: y = kx + b
                # axis=time (0)
                # 预计算每个通道的 slope (k) 和 intercept (b)
                # 使用 numpy.polyfit 或简单的线性回归公式
                # x = np.arange(n_samples)
                # 简单实现：
                x = np.arange(n_samples, dtype=data.dtype)
                # 批量计算 (n_samples, n_channels)
                # k = cov(x, y) / var(x)
                # b = mean(y) - k * mean(x)

                # 优化: x 的 mean 和 var 是常数
                x_mean = np.mean(x)
                x_var_sum = np.sum((x - x_mean) ** 2)

                y_mean = np.mean(data, axis=0)
                # xy_cov_sum = np.sum((x - x_mean)[:, None] * (data - y_mean), axis=0) # Slow?
                # Faster: dot product

                # X_centered shape: (N,)
                x_centered = x - x_mean
                # Data shape: (N, C)
                # k = (x_c . y) / sum_sq_x

                k = np.dot(x_centered, data) / x_var_sum  # (C,)
                b = y_mean - k * x_mean

                params.append(k.astype(data.dtype))
                params.append(b.astype(data.dtype))

            elif op.operation == "demean":
                means = np.mean(data, axis=0)
                params.append(means.astype(data.dtype))

        return params

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
