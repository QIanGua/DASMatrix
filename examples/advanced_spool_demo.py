import os
import shutil
from pathlib import Path

import h5py
import numpy as np

from DASMatrix import spool


def create_sample_files(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for i in range(3):
        path = Path(target_dir) / f"shot_{i:03d}.h5"
        with h5py.File(path, "w") as f:
            f.attrs["SamplingFrequency"] = 1000.0
            f.attrs["SpatialSamplingInterval"] = 1.0
            f.attrs["IUModel"] = "QuantX" if i < 2 else "iDAS"
            f.attrs["ProjectName"] = "HPC-Test"
            f.create_dataset("Data", data=np.random.randn(100, 10).astype(np.float32))
    return target_dir


def main():
    data_path = create_sample_files("demo_data")
    cache_path = Path("demo_cache")

    print("1. Initializing Spool with Persistence...")
    s = spool(data_path, cache_path=cache_path).update()
    print(f"Files found: {len(s)}")

    print("\n2. Metadata Filtering...")
    quantx_files = s.select(iu_model="QuantX")
    print(f"QuantX files: {len(quantx_files)}")

    print("\n3. Virtual Merging...")
    long_frame = s.to_frame()
    print(f"Combined Frame Shape: {long_frame.shape}")
    print(f"Sample data rate: {long_frame.fs} Hz")

    print("\n4. Units Management...")
    print(f"Current units: {long_frame.get_unit()}")

    shutil.rmtree(data_path)
    if cache_path.exists():
        shutil.rmtree(cache_path)


if __name__ == "__main__":
    main()
