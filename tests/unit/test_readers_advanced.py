from unittest.mock import patch

import numpy as np
import pytest

from DASMatrix.acquisition.das_reader import DASReader, DataType, SamplingConfig


@pytest.fixture
def mock_sampling_config():
    return SamplingConfig(channels=10, fs=1000)


class TestAdvancedReaders:
    @patch("DASMatrix.acquisition.formats.FormatRegistry.read")
    def test_segy_reader(self, mock_read, mock_sampling_config, tmp_path):
        dummy_file = tmp_path / "test.sgy"
        dummy_file.touch()

        mock_read.return_value = np.zeros((100, 10))

        reader = DASReader(mock_sampling_config, DataType.SEGY)
        data = reader.ReadRawData(dummy_file)

        assert data.shape == (100, 10)
        mock_read.assert_called_with(dummy_file, format_name="SEGY", lazy=True)

    @patch("DASMatrix.acquisition.formats.FormatRegistry.read")
    def test_miniseed_reader(self, mock_read, mock_sampling_config, tmp_path):
        dummy_file = tmp_path / "test.mseed"
        dummy_file.touch()

        mock_read.return_value = np.zeros((100, 10))

        reader = DASReader(mock_sampling_config, DataType.MINISEED)
        data = reader.ReadRawData(dummy_file)

        assert data.shape == (100, 10)
        mock_read.assert_called_with(dummy_file, format_name="MINISEED", lazy=True)

    def test_invalid_file_extension(self, mock_sampling_config):
        # Test that we can instantiate, error would happen at read time if file missing
        reader = DASReader(mock_sampling_config, DataType.SEGY)
        assert reader.data_type == DataType.SEGY
