"""
Stream API
==========

Provides a high-level interface for handling real-time DAS data streams.
Integrates with RingBuffer for efficient data buffering.
"""

import logging
import threading
import time
from typing import Callable, Optional, Union

import numpy as np

from ..core.ring_buffer import RingBuffer
from .dasframe import DASFrame
from .df import _create_dasframe  # Helper to create frames efficiently

logger = logging.getLogger(__name__)


class Stream:
    """
    A real-time data stream handler.

    Manages a circular buffer of incoming data and provides methods to
    start/stop data ingestion and process chunks.
    """

    def __init__(
        self,
        source: Union[str, Callable],
        buffer_duration: float = 60.0,
        fs: float = 1000.0,
        n_channels: int = 1,
        chunk_size: int = 1000,
    ):
        """
        Initialize the Stream.

        Args:
            source: Data source. Can be a URL string (e.g., 'tcp://...') or a generator/callable.
                    If 'simulation', generates random noise.
            buffer_duration: Length of the internal ring buffer in seconds.
            fs: Sampling rate in Hz.
            n_channels: Number of channels.
            chunk_size: Number of samples per processing chunk.
        """
        self.source = source
        self.fs = fs
        self.n_channels = n_channels
        self.chunk_size = chunk_size

        # Initialize RingBuffer
        capacity = int(buffer_duration * fs)
        self.buffer = RingBuffer(capacity, n_channels)

        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[DASFrame], None]] = None

    def start(self, callback: Optional[Callable[[DASFrame], None]] = None) -> None:
        """
        Start the stream ingestion and processing.

        Args:
            callback: Function to call with each new chunk (as DASFrame).
        """
        if self.running:
            return

        self.running = True
        self._callback = callback

        # Start ingestion thread
        self._thread = threading.Thread(target=self._ingest_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the stream."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _ingest_loop(self) -> None:
        """Internal loop to fetch data from source and push to buffer."""

        # Handle simulation source
        if self.source == "simulation":
            self._run_simulation()
        else:
            # Placeholder for other sources (TCP, ZMQ, etc.)
            raise NotImplementedError(f"Source type '{self.source}' not yet implemented.")

    def _run_simulation(self) -> None:
        """Generate simulated data."""
        dt = self.chunk_size / self.fs

        while self.running:
            loop_start = time.time()

            # Generate random noise chunk
            chunk_data = np.random.randn(self.chunk_size, self.n_channels).astype(np.float32)

            # Add some synthetic events
            if np.random.random() < 0.1:
                event_start = np.random.randint(0, self.chunk_size - 50)
                event_ch = np.random.randint(0, self.n_channels)
                chunk_data[event_start : event_start + 50, event_ch] += 5.0

            # Push to buffer
            self.buffer.push(chunk_data)

            # Process callback
            if self._callback:
                # Create a lightweight DASFrame for the chunk
                # Note: This creates a frame for JUST the new chunk.
                # Use stream.view() to look back at historical data.
                frame = _create_dasframe(
                    chunk_data,
                    fs=self.fs,
                    # Add minimal metadata
                    source="stream_simulation",
                    timestamp=time.time(),
                )
                try:
                    self._callback(frame)
                except Exception as e:
                    logger.warning("Error in stream callback: %s", e, exc_info=True)

            # Sleep to simulate real-time rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)

    def view(self, duration: Optional[float] = None) -> DASFrame:
        """
        Get a view of the recent history from the buffer as a DASFrame.

        Args:
            duration: Duration in seconds to retrieve. If None, retrieves full buffer.
        """
        if duration is None:
            samples = None
        else:
            samples = int(duration * self.fs)

        data = self.buffer.view(samples)

        # Return as DASFrame
        return _create_dasframe(data, fs=self.fs, source="stream_buffer")
