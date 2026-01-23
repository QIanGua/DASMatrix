"""
DASMatrix Metadata Inventory Demo
=================================

This script demonstrates how to use the Metadata Inventory System to manage
structured metadata for DAS datasets.
"""

from datetime import datetime, timezone

import numpy as np

from DASMatrix import from_array
from DASMatrix.core.inventory import (
    Acquisition,
    DASInventory,
    FiberGeometry,
    Interrogator,
)


def main():
    print("=== DASMatrix Inventory Demo ===\n")

    # 1. Create a DASInventory object manually
    print("1. Creating Inventory...")
    inv = DASInventory(
        project_name="Demo Project",
        experiment_name="Field Test 2024",
        acquisition=Acquisition(
            start_time=datetime.now(timezone.utc),
            n_channels=100,
            n_samples=5000,
            data_unit="strain_rate",
        ),
        fiber=FiberGeometry(gauge_length=10.0, channel_spacing=1.0, total_length=100.0),
        interrogator=Interrogator(
            manufacturer="OptaSense",
            model="O4",
            sampling_rate=1000.0,
            wavelength=1550.0,
        ),
    )
    print(inv)
    print("\nInventory JSON preview:")
    print(inv.to_json()[:200] + "...\n")

    # 2. Attach Inventory to a DASFrame
    print("2. Attaching to DASFrame...")
    data = np.random.randn(5000, 100)
    frame = from_array(data, fs=1000.0, inventory=inv)

    print(f"Frame fs: {frame.fs} Hz")
    if frame.inventory:
        print(f"Frame Project: {frame.inventory.project_name}")
        print(f"Frame Channels: {frame.inventory.acquisition.n_channels}")
    else:
        print("ERROR: Inventory not attached!")

    # 3. Check persistence through operations
    print("\n3. Testing persistence through detrend()...")
    frame_detrend = frame.detrend()

    if frame_detrend.inventory:
        print("PASS: Inventory preserved in new frame.")
        print(f"New Frame Project: {frame_detrend.inventory.project_name}")
    else:
        print("FAIL: Inventory lost in operation.")


if __name__ == "__main__":
    main()
