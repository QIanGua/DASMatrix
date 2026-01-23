"""
Event Detection Algorithms
==========================

Algorithms for detecting events in DAS data, such as STA/LTA.
"""

import numpy as np


def sta_lta(data: np.ndarray, n_sta: int, n_lta: int, axis: int = 0) -> np.ndarray:
    """Recursive STA/LTA algorithm for event detection."""
    if n_sta >= n_lta:
        raise ValueError("n_sta must be smaller than n_lta")

    sq_data = np.square(data)

    def moving_average(a, n):
        a_swapped = np.swapaxes(a, axis, 0)
        ret = np.cumsum(a_swapped, axis=0)
        ret[n:] = ret[n:] - ret[:-n]
        res = ret[n - 1 :] / n
        return np.swapaxes(res, axis, 0)

    sta = moving_average(sq_data, n_sta)
    lta = moving_average(sq_data, n_lta)

    diff = n_lta - n_sta

    sl = [slice(None)] * data.ndim
    sl[axis] = slice(diff, None)
    sta_aligned = sta[tuple(sl)]

    lta = np.where(lta == 0, 1e-10, lta)
    ratio = sta_aligned / lta

    padding_shape = list(data.shape)
    padding_shape[axis] = n_lta - 1
    padding = np.zeros(padding_shape, dtype=ratio.dtype)

    return np.concatenate([padding, ratio], axis=axis)
