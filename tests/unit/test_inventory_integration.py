from typing import Any, cast

import h5py
import numpy as np
import pytest

from DASMatrix.acquisition.formats.prodml import PRODMLFormatPlugin
from DASMatrix.core.inventory import DASInventory


@pytest.fixture
def dummy_prodml_file(tmp_path):
    path = tmp_path / "test_data.h5"
    with h5py.File(path, "w") as f:
        # Create PRODML structure
        f.attrs["SamplingFrequency"] = 500.0
        f.attrs["SpatialSamplingInterval"] = 2.0
        f.attrs["GaugeLength"] = 10.0
        f.attrs["StartTime"] = "2024-01-01T12:00:00Z"
        f.attrs["ProjectName"] = "Integration Test"
        f.attrs["SchemaVersion"] = "2.1"

        acq = f.create_group("Acquisition")
        raw = acq.create_group("Raw[0]")
        raw.create_dataset("RawData", data=np.random.randn(1000, 10).astype(np.float32))

    return path


def test_prodml_inventory_scan(dummy_prodml_file):
    plugin = PRODMLFormatPlugin()
    assert plugin.can_read(dummy_prodml_file)

    meta = plugin.scan(dummy_prodml_file)
    assert meta.sampling_rate == 500.0
    assert meta.channel_spacing == 2.0
    assert meta.inventory is not None

    inv = meta.inventory
    assert isinstance(inv, DASInventory)
    assert inv.project_name == "Integration Test"
    assert inv.acquisition.n_channels == 10
    assert cast(Any, inv.interrogator).sampling_rate == 500.0
    assert cast(Any, inv.fiber).channel_spacing == 2.0
    assert inv.acquisition.start_time.year == 2024


def test_registry_read_with_inventory(dummy_prodml_file):
    from DASMatrix.acquisition.formats import FormatRegistry

    data = FormatRegistry.read(dummy_prodml_file)
    assert "inventory" in cast(Any, data).attrs
    inv = cast(Any, data).attrs["inventory"]
    assert inv.project_name == "Integration Test"
    assert cast(Any, inv.interrogator).sampling_rate == 500.0
