"""Numba backend fusion unit tests."""

import numpy as np
from scipy import signal

from DASMatrix.core.computation_graph import FusionNode, NodeDomain, OperationNode, SourceNode
from DASMatrix.processing.backends.numba_backend import NumbaBackend


def _run_fused_ops(data: np.ndarray, ops: list[tuple[str, dict]]) -> np.ndarray:
    source = SourceNode(data, domain=NodeDomain.SIGNAL)
    nodes = [OperationNode(name, [source], kwargs=kwargs, domain=NodeDomain.SIGNAL) for name, kwargs in ops]
    fusion = FusionNode(nodes)
    backend = NumbaBackend()
    return backend.execute(fusion, data)


def test_fusion_aux_order_demean_then_detrend() -> None:
    np.random.seed(0)
    data = np.random.randn(256, 8).astype(np.float32)

    expected = signal.detrend(data - data.mean(axis=0, keepdims=True), axis=0)

    result = _run_fused_ops(
        data,
        [
            ("demean", {}),
            ("detrend", {}),
        ],
    )

    np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)


def test_fusion_aux_order_normalize_then_demean() -> None:
    np.random.seed(1)
    data = np.random.randn(256, 6).astype(np.float32)

    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    normalized = (data - mean) / std
    expected = normalized - normalized.mean(axis=0, keepdims=True)

    result = _run_fused_ops(
        data,
        [
            ("normalize", {"method": "zscore"}),
            ("demean", {}),
        ],
    )

    np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)
