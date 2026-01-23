import numpy as np

from DASMatrix.api.dasframe import DASFrame


def test_template_match_1d():
    fs = 1000.0
    nt = 1000
    nx = 5
    data = np.random.randn(nt, nx).astype(np.float32)

    t_signal = np.linspace(0, 0.1, 100)
    signal = np.sin(2 * np.pi * 50 * t_signal).astype(np.float32)
    data[500:600, 2] += 2.0 * signal

    frame = DASFrame(data, fs=fs)

    match_frame = frame.template_match(signal)
    results = match_frame.collect()

    assert results.shape == (nt - len(signal) + 1, nx)

    peak_idx = np.argmax(results[:, 2])
    assert peak_idx == 500
    assert results[peak_idx, 2] > 0.7


def test_template_match_2d():
    fs = 100.0
    nt, nx = 200, 20
    data = np.random.randn(nt, nx).astype(np.float32) * 0.1

    mt, mx = 20, 5
    template = np.zeros((mt, mx), dtype=np.float32)
    for j in range(mx):
        template[j : j + 10, j] = 1.0

    data[100:120, 10:15] += template

    frame = DASFrame(data, fs=fs)
    match_frame = frame.template_match(template)
    results = match_frame.collect()

    assert results.shape == (nt - mt + 1, nx - mx + 1)

    max_idx = np.unravel_index(np.argmax(results), results.shape)
    assert max_idx == (100, 10)
    assert results[max_idx] > 0.9
