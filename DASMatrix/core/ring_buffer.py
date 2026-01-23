"""
Ring Buffer Implementation
==========================

A high-performance circular buffer based on NumPy for real-time data streaming.
Prioritizes zero-allocation operations where possible and minimizes memory churn.
"""

from threading import Lock
from typing import Optional, Union

import numpy as np


class RingBuffer:
    """
    A circular buffer for storing real-time data streams.

    Features:
    - Fixed memory footprint (pre-allocated).
    - Thread-safe pushing and viewing.
    - Efficient circular wrapping.
    """

    def __init__(self, capacity: int, n_channels: int, dtype: Union[type, np.dtype] = np.float32):
        """
        Initialize the RingBuffer.

        Args:
            capacity: Number of time samples to store.
            n_channels: Number of channels.
            dtype: Data type of the buffer.
        """
        self.capacity = capacity
        self.n_channels = n_channels
        self.dtype = dtype

        # Pre-allocate buffer
        self._buffer = np.zeros((capacity, n_channels), dtype=dtype)

        # Pointers and counters
        self._cursor = 0  # Points to the next write position
        self._size = 0  # Current number of valid samples stored
        self._lock = Lock()

    @property
    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return self._size >= self.capacity

    def push(self, data: np.ndarray) -> None:
        """
        Push new data into the buffer.

        If data length exceeds capacity, only the most recent 'capacity' samples are kept.

        Args:
            data: Input array of shape (time, channels) or (time,).
        """
        # Ensure 2D shape
        if data.ndim == 1:
            data = data[:, np.newaxis]

        n_samples = data.shape[0]

        if n_samples == 0:
            return

        with self._lock:
            # Case 1: Data is larger than capacity, keep only latest
            if n_samples >= self.capacity:
                self._buffer[:] = data[-self.capacity :]
                self._cursor = 0
                self._size = self.capacity
                return

            # Case 2: Data fits, potentially wrapping around
            # Calculate space at the end
            space_end = self.capacity - self._cursor

            if n_samples <= space_end:
                # No wrapping needed
                self._buffer[self._cursor : self._cursor + n_samples] = data
                self._cursor += n_samples
            else:
                # Wrapping needed
                # Part 1: Fill to end
                self._buffer[self._cursor :] = data[:space_end]
                # Part 2: Wrap to beginning
                remaining = n_samples - space_end
                self._buffer[:remaining] = data[space_end:]
                self._cursor = remaining

            # Update cursor and size (handle wrapping for cursor if it hit exactly capacity)
            if self._cursor == self.capacity:
                self._cursor = 0

            self._size = min(self._size + n_samples, self.capacity)

    def view(self, samples: Optional[int] = None) -> np.ndarray:
        """
        Get a view of the most recent data.

        Args:
            samples: Number of recent samples to retrieve. If None, returns all valid data.

        Returns:
            A numpy array containing the requested data.
            Note: This may return a copy if the data wraps around the buffer end,
            or a view if it is contiguous.
        """
        with self._lock:
            if self._size == 0:
                return np.empty((0, self.n_channels), dtype=self.dtype)

            count = self._size if samples is None else min(samples, self._size)

            # Determine start index (handling circularity back from cursor)
            # The newest data ends at self._cursor - 1
            # So start index is (self._cursor - count) % self.capacity
            start_idx = (self._cursor - count) % self.capacity

            # Check if contiguous
            if start_idx + count <= self.capacity:
                # Contiguous chunk: return a view
                return self._buffer[start_idx : start_idx + count]
            else:
                # Wraps around: must return a copy (concatenation)
                # Part 1: From start to end
                part1 = self._buffer[start_idx:]
                # Part 2: From beginning to remaining
                remaining = count - (self.capacity - start_idx)
                part2 = self._buffer[:remaining]
                return np.concatenate((part1, part2), axis=0)

    def clear(self) -> None:
        """Reset the buffer state."""
        with self._lock:
            self._cursor = 0
            self._size = 0
            # Optional: zero out memory (usually not strictly necessary for ring buffer logic)
            # self._buffer.fill(0)

    def __repr__(self) -> str:
        return f"RingBuffer(capacity={self.capacity}, channels={self.n_channels}, size={self._size})"
