import h5py
import numpy as np
import pytest

from DASMatrix.api.dasframe import DASFrame
from DASMatrix.api.spool import DASSpool


@pytest.fixture
def multiple_h5_files(tmp_path):
    files = []
    for i in range(3):
        p = tmp_path / f"data_{i}.h5"
        with h5py.File(p, "w") as f:
            f.attrs["SamplingFrequency"] = 100.0
            f.attrs["SpatialSamplingInterval"] = 1.0
            f.attrs["StartTime"] = f"2024-01-01T00:00:0{i}Z"
            data = np.full((100, 10), i, dtype=np.float32)
            f.create_dataset("Data", data=data)
        files.append(p)
    return files


def test_spool_to_frame_concatenation(multiple_h5_files):
    spool = DASSpool(multiple_h5_files).update()
    assert len(spool) == 3

    frame = spool.to_frame()
    assert isinstance(frame, DASFrame)

    assert frame.shape == (300, 10)

    data = frame.collect()

    factor = (1 / 4) * (np.pi / (2**13))

    assert np.allclose(data[:100], 0)
    assert np.allclose(data[100:200], 1 * factor)
    assert np.allclose(data[200:300], 2 * factor)

    processed = frame.bandpass(1, 40).collect()
    assert processed.shape == (300, 10)
