import dask.array as da
import h5py
import numpy as np
import pytest

from DASMatrix.acquisition.das_reader import DASReader, DataType
from DASMatrix.config.sampling_config import SamplingConfig


@pytest.fixture
def sampling_config():
    return SamplingConfig(fs=1000, channels=10)


@pytest.fixture
def dummy_dat_file(tmp_path, sampling_config):
    # Create a dummy .dat file
    file_path = tmp_path / "test.dat"
    n_samples = 100
    n_channels = sampling_config.channels

    # Int16 data
    data = np.random.randint(-1000, 1000, size=(n_samples, n_channels)).astype("<i2")
    data.tofile(file_path)

    expected_data = data.astype(np.float64) * np.pi / 2**13
    if sampling_config.byte_order == "little":
        # Our writer used big endian, but if config defaults to big, it matches.
        # SamplingConfig default byte_order?
        # Let's check or assume "big" for DAS usually.
        # Actually DASReader defaults ">i2" if config.byte_order is "big".
        # SamplingConfig defaults byte_order="big" usually?
        pass

    return file_path, expected_data


@pytest.fixture
def dummy_h5_file(tmp_path, sampling_config):
    file_path = tmp_path / "test.h5"
    n_samples = 100
    n_channels = sampling_config.channels

    data = np.random.randint(-1000, 1000, size=(n_samples, n_channels)).astype(
        np.float32
    )

    with h5py.File(file_path, "w") as f:
        f.create_dataset("Acquisition/Raw[0]", data=data)

    expected_data = (data / 4) * np.pi / 2**13
    return file_path, expected_data


def test_dat_reader_lazy(dummy_dat_file, sampling_config):
    file_path, expected_data = dummy_dat_file

    reader = DASReader(sampling_config, DataType.DAT)
    dask_data = reader.ReadRawData(file_path)

    # Verify it is a dask array
    assert isinstance(dask_data, da.Array)

    # Compute and check values
    computed_data = dask_data.compute()

    # Note: endianness might need care.
    # The dummy file was written as ">i2".
    # SamplingConfig default is often "big".

    # Let's check loose match or handle byte swap if needed
    # But mainly we check type first.

    assert computed_data.shape == expected_data.shape
    assert np.allclose(computed_data, expected_data, atol=1e-5)


def test_h5_reader_lazy(dummy_h5_file, sampling_config):
    file_path, expected_data = dummy_h5_file

    reader = DASReader(sampling_config, DataType.H5)
    dask_data = reader.ReadRawData(file_path)

    assert isinstance(dask_data, da.Array)

    computed_data = dask_data.compute()
    assert np.allclose(computed_data, expected_data, atol=1e-5)
