import h5py
import numpy as np
import pytest

from DASMatrix.api.dasframe import DASFrame
from DASMatrix.api.spool import DASSpool
from DASMatrix.core.inventory import DASInventory


@pytest.fixture
def persisted_spool_dir(tmp_path):
    spool_dir = tmp_path / "spool_files"
    spool_dir.mkdir()
    cache_dir = tmp_path / "cache"

    files = []
    for i in range(5):
        p = spool_dir / f"data_{i}.hdf5"
        with h5py.File(p, "w") as f:
            f.attrs["SamplingFrequency"] = 1000.0
            f.attrs["SpatialSamplingInterval"] = 1.0
            f.attrs["ProjectName"] = f"Project_{i % 2}"
            f.attrs["IUModel"] = "QuantX" if i < 3 else "iDAS"
            f.attrs["StartTime"] = f"2026-01-01T00:00:{i:02d}Z"
            f.create_dataset("Data", data=np.random.randn(100, 10).astype(np.float32))
        files.append(p)

    return spool_dir, cache_dir


def test_spool_persistence_and_filtering(persisted_spool_dir):
    spool_dir, cache_dir = persisted_spool_dir

    spool = DASSpool(spool_dir, cache_path=cache_dir).update()
    assert len(spool) == 5
    assert (cache_dir / "index.parquet").exists()
    assert (cache_dir / "meta_cache.pkl").exists()

    spool2 = DASSpool(spool_dir, cache_path=cache_dir).update()
    assert len(spool2) == 5

    quantx_spool = spool2.select(model="QuantX")
    assert len(quantx_spool) == 3

    idas_spool = spool2.select(model="iDAS")
    assert len(idas_spool) == 2

    proj0_spool = spool2.select(ProjectName="Project_0")
    assert len(proj0_spool) == 3


def test_unit_normalization():
    data = np.ones((100, 10))
    fs = 1000.0

    from datetime import datetime

    from DASMatrix.core.inventory import Acquisition, FiberGeometry, Interrogator

    inv = DASInventory(
        project_name="Test",
        acquisition=Acquisition(start_time=datetime.now(), n_channels=10, data_unit="rad/s"),
        fiber=FiberGeometry(gauge_length=10.0, channel_spacing=1.0),
        interrogator=Interrogator(model="iDAS", sampling_rate=fs, wavelength=1550.0),
    )

    frame = DASFrame(data, fs=fs, inventory=inv)
    assert frame.get_unit() == "rad/s"

    norm_frame = frame.to_standard_units()
    assert norm_frame.get_unit() == "strain_rate"

    val = norm_frame.collect()[0, 0]
    assert 1e-9 < val < 1e-7

    m_frame = norm_frame.convert_units("strain_rate")
    assert m_frame.get_unit() == "strain_rate"
