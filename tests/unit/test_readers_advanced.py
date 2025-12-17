
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path

from DASMatrix.acquisition.das_reader import DASReader, DataType, SamplingConfig

@pytest.fixture
def mock_sampling_config():
    return SamplingConfig(channels=10, fs=1000)

class TestAdvancedReaders:
    
    @patch('DASMatrix.acquisition.das_reader.obspy')
    def test_segy_reader(self, mock_obspy, mock_sampling_config, tmp_path):
        # Create a dummy file to pass existence check
        dummy_file = tmp_path / "test.sgy"
        dummy_file.touch()

        # Mock ObsPy stream and traces
        mock_stream = MagicMock()
        mock_trace = MagicMock()
        # Create dummy data: 100 samples
        mock_trace.data = np.random.rand(100).astype(np.float32)
        
        # Reader expects stack of traces. 
        # If we have 10 channels, stream should have 10 traces
        mock_stream = [mock_trace for _ in range(10)]
        
        mock_obspy.read.return_value = mock_stream

        reader = DASReader(mock_sampling_config, DataType.SEGY)
        data = reader.ReadRawData(dummy_file)

        # Expected shape: (100, 10) because logic is stack([tr.data], axis=1)
        assert data.shape == (100, 10)
        mock_obspy.read.assert_called_with(str(dummy_file), format="SEGY")

    @patch('DASMatrix.acquisition.das_reader.obspy')
    def test_miniseed_reader(self, mock_obspy, mock_sampling_config, tmp_path):
        # Create a dummy file
        dummy_file = tmp_path / "test.mseed"
        dummy_file.touch()

        # Mock ObsPy stream
        mock_stream = MagicMock()
        mock_trace = MagicMock()
        mock_trace.data = np.random.rand(100).astype(np.float32)
        mock_stream.__iter__.return_value = [mock_trace for _ in range(10)]
        mock_stream.__len__.return_value = 10 
        
        # Logic iterates over stream
        mock_obspy.read.return_value = mock_stream

        reader = DASReader(mock_sampling_config, DataType.MINISEED)
        data = reader.ReadRawData(dummy_file)

        assert data.shape == (100, 10)
        mock_obspy.read.assert_called_with(str(dummy_file), format="MSEED")
        mock_stream.merge.assert_called_once()

    def test_invalid_file_extension(self, mock_sampling_config):
        # Test that we can instantiate, error would happen at read time if file missing
        reader = DASReader(mock_sampling_config, DataType.SEGY)
        assert reader.data_type == DataType.SEGY

