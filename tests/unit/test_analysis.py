import numpy as np

from DASMatrix.api.dasframe import DASFrame


class TestAnalysis:
    """Test analysis algorithms."""

    def test_sta_lta_basic(self):
        """Verify basic STA/LTA behavior."""
        fs = 1000
        n_samples = 2000
        data = np.random.randn(n_samples, 1) * 0.1
        data[1000:1100, 0] += 2.0

        frame = DASFrame(data, fs=fs)
        ratio_frame = frame.sta_lta(n_sta=10, n_lta=100)
        ratio = ratio_frame.collect()

        assert ratio.shape == (n_samples, 1)
        peak_idx = np.argmax(ratio)
        assert 1000 <= peak_idx <= 1150
        assert ratio[peak_idx] > 5.0
