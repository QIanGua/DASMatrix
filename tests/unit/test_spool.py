import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from DASMatrix import spool


class TestDASSpool(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.create_dummy_files()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_dummy_files(self):
        # Create 3 dummy H5 files with time sequence
        fs = 1000.0
        n_samples = 1000  # 1 second
        dx = 1.0
        n_channels = 10

        start_time = datetime(2024, 1, 1, 0, 0, 0)

        for i in range(3):
            file_start = start_time + timedelta(seconds=i)
            # Create H5 file
            fname = self.test_dir / f"test_{i}.h5"

            with h5py.File(fname, "w") as f:
                # Use standard H5 structure compatible with H5FormatPlugin (Generic H5)
                # H5FormatPlugin expects Acquisition/Raw[0] usually or similar
                # Let's check H5FormatPlugin...
                # Actually, H5Reader (legacy) expects Acquisition/Raw[0].
                # But FormatRegistry uses H5FormatPlugin.
                # Let's use a structure that H5FormatPlugin likely supports
                # or make it generic.
                # If we use PRODML structure, it will be detected as PRODML.

                # Let's try to mimic PRODML for better detection or just generic H5
                # H5FormatPlugin (generic) is not shown in read_file calls earlier.
                # Let's assume standard layout or use "PRODML" structure as I saw
                # PRODML plugin implemented.

                # Using PRODML structure:
                grp = f.create_group("Acquisition/Raw[0]")
                data = np.random.randn(n_samples, n_channels).astype(np.float32)
                grp.create_dataset("RawData", data=data)

                f.attrs["SamplingFrequency"] = fs
                f.attrs["SpatialSamplingInterval"] = dx
                f.attrs["StartTime"] = file_start.isoformat()
                f.attrs["SchemaVersion"] = "2.0"  # Trigger PRODML detection

    def test_spool_find_files(self):
        s = spool(self.test_dir / "*.h5")
        self.assertEqual(len(s._files), 3)

    def test_spool_update(self):
        s = spool(self.test_dir / "*.h5").update()
        self.assertIsNotNone(s._index)
        assert s._index is not None  # type narrowing for type checker
        self.assertEqual(len(s._index), 3)

        # Check metadata
        first_row = s._index.iloc[0]
        self.assertEqual(first_row["sampling_rate"], 1000.0)
        self.assertEqual(first_row["n_channels"], 10)

        # Check time sorting
        self.assertTrue(pd.to_datetime(s._index.iloc[0]["start_time"]) < pd.to_datetime(s._index.iloc[1]["start_time"]))

    def test_spool_select(self):
        s = spool(self.test_dir / "*.h5").update()

        # Select middle second
        t_start = datetime(2024, 1, 1, 0, 0, 0) + timedelta(seconds=0.5)
        t_end = datetime(2024, 1, 1, 0, 0, 0) + timedelta(seconds=1.5)

        # Should select first and second file because:
        # File 0: 00:00:00 - 00:00:01 (overlaps 00:00:00.5 - 00:00:01)
        # File 1: 00:00:01 - 00:00:02 (overlaps 00:00:01 - 00:00:01.5)
        # File 2: 00:00:02 - 00:00:03 (no overlap with end 00:00:01.5 ?)
        # Actually t_end is 00:00:01.5. File 2 starts at 00:00:02. So File 2 excluded.

        subset = s.select(time=(t_start, t_end))
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset._files[0].name, "test_0.h5")
        self.assertEqual(subset._files[1].name, "test_1.h5")

    def test_spool_chunk(self):
        s = spool(self.test_dir / "*.h5").update()

        count = 0
        for frame in s:
            self.assertEqual(frame.shape, (1000, 10))
            self.assertEqual(frame.fs, 1000.0)
            count += 1

        self.assertEqual(count, 3)

    def test_spool_getitem(self):
        s = spool(self.test_dir / "*.h5").update()
        frame = s[1]
        self.assertEqual(frame.shape, (1000, 10))
