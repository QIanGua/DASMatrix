from datetime import datetime

import numpy as np

from DASMatrix.api.dasframe import DASFrame
from DASMatrix.core.event import DASEvent, EventCatalog
from DASMatrix.core.inventory import Acquisition, DASInventory


class TestEventSystem:
    """Test EventCatalog and trigger detection."""

    def test_trigger_detection(self):
        """Verify trigger_detection logic."""
        fs = 100.0
        nt, nx = 1000, 10
        data = np.zeros((nt, nx))

        # Create an event at t=2s to 3s (sample 200-300)
        # Channels 2-5
        data[200:300, 2:6] = 5.0

        inv = DASInventory(
            project_name="Test",
            acquisition=Acquisition(
                start_time=datetime(2024, 1, 1, 12, 0, 0), n_channels=nx, n_samples=nt, data_unit="unknown"
            ),
        )

        frame = DASFrame(data, fs=fs, inventory=inv)

        # Trigger detection
        catalog = frame.trigger_detection(threshold=2.0)

        assert isinstance(catalog, EventCatalog)
        # Should detect 1 event
        assert len(catalog) == 1

        # Check event details
        # We need to access the dataframe
        event = catalog.df.row(0, named=True)
        assert event["min_channel"] == 2
        assert event["max_channel"] == 5

        # Check time
        expected_start = datetime(2024, 1, 1, 12, 0, 2)
        diff = abs((event["start_time"] - expected_start).total_seconds())
        assert diff < 0.1

    def test_catalog_io(self, tmp_path):
        """Test EventCatalog CSV I/O."""
        evt = DASEvent(id="test_1", start_time=datetime(2024, 1, 1), min_channel=0, max_channel=10, confidence=0.9)
        catalog = EventCatalog([evt])

        csv_path = tmp_path / "events.csv"
        catalog.to_csv(csv_path)

        loaded = EventCatalog.from_csv(csv_path)
        assert len(loaded) == 1
        assert loaded.df["id"][0] == "test_1"
        assert loaded.df["confidence"][0] == 0.9
