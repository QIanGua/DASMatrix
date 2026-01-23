import numba
import numpy as np


@numba.njit(parallel=True, fastmath=True)
def sliding_ncc_1d(data: np.ndarray, template: np.ndarray) -> np.ndarray:
    n_temp = template.shape[0]
    shape = data.shape
    n_samples = shape[-1]
    n_out = n_samples - n_temp + 1

    if n_out <= 0:
        return np.zeros(shape[:-1] + (0,), dtype=data.dtype)

    temp_energy = np.sum(template**2)

    n_traces = 1
    for s in shape[:-1]:
        n_traces *= s

    data_reshaped = data.reshape(n_traces, n_samples)
    results_reshaped = np.zeros((n_traces, n_out), dtype=data.dtype)

    for j in numba.prange(n_traces):  # type: ignore
        trace = data_reshaped[j]
        for i in range(n_out):
            window = trace[i : i + n_temp]
            window_energy = np.sum(window**2)

            if window_energy == 0 or temp_energy == 0:
                results_reshaped[j, i] = 0.0
                continue

            cross_prod = np.sum(window * template)
            results_reshaped[j, i] = cross_prod / np.sqrt(window_energy * temp_energy)

    return results_reshaped.reshape(shape[:-1] + (n_out,))


@numba.njit(parallel=True, fastmath=True)
def sliding_ncc_2d(data: np.ndarray, template: np.ndarray) -> np.ndarray:
    nt, nx = data.shape[-2], data.shape[-1]
    mt, mx = template.shape

    out_t = nt - mt + 1
    out_x = nx - mx + 1

    if out_t <= 0 or out_x <= 0:
        # Construct empty result with correct shape
        return np.zeros(data.shape[:-2] + (0, 0), dtype=data.dtype)

    temp_energy = np.sum(template**2)
    if temp_energy == 0:
        return np.zeros(data.shape[:-2] + (out_t, out_x), dtype=data.dtype)

    # Flatten batches
    flat_data = data.reshape(-1, nt, nx)
    n_batches = flat_data.shape[0]
    flat_results = np.zeros((n_batches, out_t, out_x), dtype=data.dtype)

    for b in numba.prange(n_batches):  # type: ignore
        for j in range(out_x):
            for i in range(out_t):
                window = flat_data[b, i : i + mt, j : j + mx]
                window_energy = np.sum(window**2)

                if window_energy == 0:
                    flat_results[b, i, j] = 0.0
                    continue

                cross_prod = np.sum(window * template)
                flat_results[b, i, j] = cross_prod / np.sqrt(window_energy * temp_energy)

    return flat_results.reshape(data.shape[:-2] + (out_t, out_x))


def match_template_1d(data: np.ndarray, template: np.ndarray, axis: int = 0) -> np.ndarray:
    if axis != -1 and axis != data.ndim - 1:
        data_swapped = np.swapaxes(data, axis, -1)
        res = sliding_ncc_1d(data_swapped, template)
        return np.swapaxes(res, axis, -1)
    else:
        return sliding_ncc_1d(data, template)
