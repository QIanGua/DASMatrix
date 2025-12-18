import numpy as np
import xarray as xr
from scipy import signal


def debug_kernel(data):
    print(f"Kernel Input Shape: {data.shape}")
    return signal.detrend(data, axis=-1)


def main():
    nt, nx = 100, 10
    data_np = np.random.randn(nt, nx)

    # Coords
    coords = {
        "time": np.arange(nt),
        "distance": np.arange(nx),
    }

    da_xr = xr.DataArray(data_np, dims=("time", "distance"), coords=coords).chunk(
        {"time": -1}
    )  # Contiguous time

    print(f"Original Shape: {da_xr.shape}")

    # Apply ufunc
    res = xr.apply_ufunc(
        debug_kernel,
        da_xr,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        dask="parallelized",
        output_dtypes=[float],
    )

    print("Computing...")
    res.compute()
    print(f"Result Shape: {res.shape}")


if __name__ == "__main__":
    main()
