import json
from datetime import datetime, timezone

import numpy as np

from DASMatrix.api.dasframe import DASFrame
from DASMatrix.core.inventory import (
    Acquisition,
    DASInventory,
    FiberGeometry,
    Interrogator,
)


def test_inventory_creation():
    inv = DASInventory(
        project_name="Test Project",
        acquisition=Acquisition(
            start_time=datetime.now(timezone.utc),
            n_channels=100,
            data_unit="strain_rate",
        ),
        fiber=FiberGeometry(gauge_length=10.0, channel_spacing=1.0),
        interrogator=Interrogator(model="Test Unit", sampling_rate=1000.0),
    )
    assert inv.project_name == "Test Project"
    assert inv.acquisition.n_channels == 100
    assert inv.fiber.gauge_length == 10.0


def test_inventory_serialization(tmp_path):
    inv = DASInventory(
        project_name="Serialization Test",
        acquisition=Acquisition(start_time=datetime(2023, 1, 1), n_channels=50),
    )

    # To JSON string
    json_str = inv.to_json()
    data = json.loads(json_str)
    assert data["project_name"] == "Serialization Test"

    # To JSON file
    file_path = tmp_path / "inv.json"
    inv.to_json(file_path)

    # From JSON file
    loaded = DASInventory.from_json(file_path)
    assert loaded.project_name == inv.project_name
    assert loaded.acquisition.n_channels == 50


def test_dasframe_inventory_integration():
    data = np.zeros((100, 10))
    inv = DASInventory(
        project_name="Frame Test",
        acquisition=Acquisition(start_time=datetime.now(timezone.utc), n_channels=10),
    )

    # Create frame with inventory
    frame = DASFrame(data, fs=1000.0, inventory=inv)

    assert frame.inventory is not None
    assert frame.inventory.project_name == "Frame Test"

    # Check property access
    assert frame.inventory.acquisition.n_channels == 10

    # Check serialization persistence (in memory) via operations
    frame2 = frame.detrend()
    assert frame2.inventory is not None
    assert frame2.inventory.project_name == "Frame Test"


def test_dasframe_inventory_dict_conversion():
    data = np.zeros((100, 10))
    inv_dict = {
        "project_name": "Dict Test",
        "acquisition": {"start_time": "2023-01-01T00:00:00", "n_channels": 10},
    }

    frame = DASFrame(data, fs=1000.0, inventory=inv_dict)
    assert isinstance(frame.inventory, DASInventory)
    assert frame.inventory.project_name == "Dict Test"
