"""Hybrid Engine Unit Tests."""

import numpy as np
import pytest
from scipy import signal

from DASMatrix.api.dasframe import DASFrame


@pytest.fixture
def sample_data():
    """Generate deterministic random sample data."""
    np.random.seed(42)
    # Shape: (Time=1000, Channels=10)
    return np.random.randn(1000, 10).astype(np.float32)


@pytest.fixture
def df(sample_data):
    return DASFrame(sample_data, fs=100.0)


def test_demean_correctness(sample_data, df):
    """Test if demean matches numpy."""
    expected = sample_data - np.mean(sample_data, axis=0)

    # Run Hybrid Engine
    result = df.demean(axis="time").collect()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_detrend_correctness(sample_data, df):
    """Test if detrend matches scipy.signal.detrend (linear)."""
    # SciPy linear detrend
    expected = signal.detrend(sample_data, axis=0)

    # Hybrid Engine
    result = df.detrend(axis="time").collect()

    # Check
    np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3, err_msg="Detrend result mismatch")


def test_abs_scale_correctness(sample_data, df):
    """Test abs and scale."""
    expected = np.abs(sample_data) * 2.5

    result = df.abs().scale(2.5).collect()

    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_chain_fusion_correctness(sample_data, df):
    """Test chaining multiple ops: detrend -> abs -> scale."""
    # Numpy/SciPy Equivalent
    step1 = signal.detrend(sample_data, axis=0)
    step2 = np.abs(step1)
    expected = step2 * 0.5

    # Hybrid Engine
    result = df.detrend(axis="time").abs().scale(0.5).collect()

    np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)


def test_collect_hybrid_engine_matches_xarray(sample_data, df):
    """Ensure hybrid collect matches xarray collect for supported ops."""
    pipeline = df.demean(axis="time").abs().scale(0.5)
    expected = pipeline.collect(engine="xarray")
    result = pipeline.collect(engine="hybrid")

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_collect_hybrid_fft_matches_xarray(sample_data, df):
    """Ensure hybrid FFT matches xarray FFT for supported ops."""
    pipeline = df.fft()
    expected = pipeline.collect(engine="xarray")
    result = pipeline.collect(engine="hybrid")

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_collect_hybrid_fk_filter_matches_xarray(sample_data, df):
    """Ensure hybrid FK filter matches xarray FK filter for supported ops."""
    pipeline = df.fk_filter(v_min=10.0, v_max=100.0, dx=1.0)
    expected = pipeline.collect(engine="xarray")
    result = pipeline.collect(engine="hybrid")

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_collect_hybrid_median_filter_matches_xarray(sample_data, df):
    """Ensure hybrid median filter matches xarray median filter for supported ops."""
    pipeline = df.median_filter(k=3, axis="time")
    expected = pipeline.collect(engine="xarray")
    result = pipeline.collect(engine="hybrid")

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_collect_hybrid_stft_matches_xarray(sample_data, df):
    """Ensure hybrid STFT matches xarray STFT for supported ops."""
    pipeline = df.stft(nperseg=64, noverlap=32, window="hann")
    expected = pipeline.collect(engine="xarray")
    result = pipeline.collect(engine="hybrid")

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_bandpass_placeholder(sample_data, df):
    """Test bandpass (placeholder test)."""
    # Just ensure it runs without error and returns correct shape
    result = df.bandpass(1, 10).collect()
    assert result.shape == sample_data.shape
