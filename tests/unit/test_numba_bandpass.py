"""Test Numba implementation of bandpass filter."""

import numpy as np
import pytest
from scipy import signal

from DASMatrix.core.computation_graph import (
    ComputationGraph,
    FusionNode,
    OperationNode,
    SourceNode,
)
from DASMatrix.processing.backends.numba_backend import NumbaBackend


class TestNumbaBandpass:
    """Test Bandpass implementation in Numba Backend."""

    def test_bandpass_correctness(self):
        """Verify that Numba bandpass matches Scipy implementation (causal sosfilt)."""
        # Setup data
        rows = 2000  # Time
        cols = 5     # Channels
        fs = 1000.0
        # Create random signal
        np.random.seed(42)
        data = np.random.randn(rows, cols).astype(np.float64)

        # Create Graph Node
        source = SourceNode(data, name="source")
        # bandpass: low=10, high=100
        low = 10.0
        high = 100.0
        order = 4

        op_bandpass = OperationNode(
            operation="bandpass",
            inputs=[source],
            name="bandpass_op",
            kwargs={"low": low, "high": high, "order": order, "fs": fs}
        )

        # Fusion Node
        fusion_node = FusionNode([op_bandpass], name="fused_bandpass")

        # Execute with NumbaBackend
        backend = NumbaBackend()
        result_numba = backend.execute(fusion_node, data)

        # Execute with Scipy (Reference)
        nyq = 0.5 * fs
        sos = signal.butter(
            order, [low / nyq, high / nyq], btype="band", output="sos"
        )
        # We implemented sosfilt (Direct Form II Transposed), causal.
        # Scipy default sosfilt is axis=-1, but here data is (Time, Channel).
        # We need axis=0.
        result_scipy = signal.sosfilt(sos, data, axis=0)

        # Comparison
        # Since we use float64, precision should be high.
        # sosfilt might have slight numerical differences due to implementation details or order,
        # but Numba implementation follows standard DF-II Transposed logic.

        np.testing.assert_allclose(
            result_numba,
            result_scipy,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Numba bandpass result does not match Scipy sosfilt"
        )

    def test_bandpass_chain(self):
        """Verify bandpass works in a chain (e.g. detrend -> bandpass -> abs)."""
        rows = 1000
        cols = 3
        fs = 500.0
        data = np.random.randn(rows, cols) * 10.0 + 100.0 # Offset

        # Add trend
        t = np.arange(rows)
        trend = t.reshape(-1, 1) * 0.1
        data += trend

        # 1. Detrend
        # 2. Bandpass
        # 3. Abs

        # Reference Scipy Calculation
        # Detrend
        step1 = signal.detrend(data, axis=0)

        # Bandpass
        sos = signal.butter(4, [5.0/250.0, 50.0/250.0], btype="band", output="sos")
        step2 = signal.sosfilt(sos, step1, axis=0)

        # Abs
        expected = np.abs(step2)

        # Numba Calculation
        source = SourceNode(data)
        op1 = OperationNode("detrend", [source])
        op2 = OperationNode("bandpass", [op1], kwargs={"low": 5.0, "high": 50.0, "order": 4, "fs": fs})
        op3 = OperationNode("abs", [op2])

        fusion_node = FusionNode([op1, op2, op3])

        backend = NumbaBackend()
        result_numba = backend.execute(fusion_node, data)

        np.testing.assert_allclose(result_numba, expected, rtol=1e-5, atol=1e-5)
