from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from DASMatrix.acquisition.das_reader import DASReader, DataType, SamplingConfig
from DASMatrix.api import stream_func
from DASMatrix.processing.das_processor import DASProcessor


def test_stream_func_warns_and_returns_frame():
    with pytest.warns(DeprecationWarning):
        frame = stream_func("tcp://127.0.0.1:9000", fs=1000.0)
    assert frame.shape == (0, 1)
    assert frame.fs == 1000.0


def test_das_reader_legacy_method_warns():
    config = SamplingConfig(fs=1000.0, channels=4)
    reader = DASReader(config, DataType.SEGY)
    with patch("DASMatrix.acquisition.formats.FormatRegistry.read", return_value=np.zeros((8, 4))):
        with pytest.warns(DeprecationWarning):
            arr = reader.ReadRawData(Path("dummy.sgy"))
    assert arr.shape == (8, 4)


def test_das_processor_legacy_method_warns():
    config = SamplingConfig(fs=1000.0, channels=4)
    processor = DASProcessor(config)
    data = np.random.randn(64, 4)

    with pytest.warns(DeprecationWarning):
        out = processor.FKFilter(data, v_min=100.0, dx=1.0)
    assert out.shape == data.shape
