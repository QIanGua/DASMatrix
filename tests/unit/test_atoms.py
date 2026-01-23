import numpy as np
from scipy import signal

from DASMatrix.api.dasframe import DASFrame
from DASMatrix.processing.atoms import Partial, Sequential, SosFilt


class TestAtoms:
    """Test stateful Atoms pipeline."""

    def test_sos_filt_continuity(self):
        """Verify that SosFilt maintains continuity across chunks."""
        fs = 1000
        t = np.arange(2000) / fs
        data = np.sin(2 * np.pi * 10 * t).reshape(-1, 1)

        sos = signal.butter(4, 20, btype="high", fs=fs, output="sos")

        zi = signal.sosfilt_zi(sos)
        zi = np.tile(zi[:, :, np.newaxis], (1, 1, 1))
        expected_causal, _ = signal.sosfilt(sos, data, axis=0, zi=zi)

        atom = SosFilt(sos)
        chunk1 = DASFrame(data[:1000], fs=fs)
        chunk2 = DASFrame(data[1000:], fs=fs)

        filtered1 = atom(chunk1).collect()
        filtered2 = atom(chunk2).collect()

        chunked_result = np.concatenate([filtered1, filtered2], axis=0)

        np.testing.assert_allclose(chunked_result, expected_causal, atol=1e-6)

    def test_sequential_pipeline(self):
        """Test Sequential pipeline with Partial and SosFilt."""
        fs = 1000
        data = np.random.randn(1000, 2)
        sos = signal.butter(2, 50, btype="low", fs=fs, output="sos")

        pipeline = Sequential([Partial("demean"), SosFilt(sos), Partial("abs")])

        frame = DASFrame(data, fs=fs)
        result = pipeline(frame).collect()

        assert result.shape == (1000, 2)
        assert np.all(result >= 0)

    def test_pipeline_reset(self):
        """Test that reset clearing the filter state."""
        fs = 1000
        data = np.ones((100, 1))
        sos = signal.butter(2, 10, btype="low", fs=fs, output="sos")

        atom = SosFilt(sos)
        _ = atom(DASFrame(data, fs=fs))
        assert atom._zi is not None

        atom.reset()
        assert atom._zi is None
