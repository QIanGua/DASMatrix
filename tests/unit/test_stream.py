import time

import numpy as np

from DASMatrix.api.dasframe import DASFrame
from DASMatrix.api.stream import Stream


def test_stream_initialization():
    s = Stream(source="simulation", buffer_duration=10.0, fs=100.0, n_channels=5)
    assert s.buffer.capacity == 1000
    assert s.n_channels == 5
    assert not s.running


def test_stream_simulation_flow():
    # Use a small chunk size and fast rate for testing
    s = Stream(
        source="simulation",
        buffer_duration=5.0,
        fs=1000.0,
        chunk_size=100,
        n_channels=1,
    )

    received_frames = []

    def callback(frame):
        received_frames.append(frame)

    s.start(callback=callback)

    # Let it run for a bit (needs enough time for at least one chunk)
    time.sleep(0.3)

    s.stop()

    assert len(received_frames) > 0
    assert isinstance(received_frames[0], DASFrame)
    assert received_frames[0].data.shape == (100, 1)

    # Check buffer content
    view = s.view()
    assert len(view.data) > 0
    # Should correspond roughly to the number of chunks processed
    min_expected = len(received_frames) * 100
    assert len(view.data) >= min_expected


def test_stream_view():
    s = Stream(source="simulation", buffer_duration=2.0, fs=100.0)
    # Push some manual data to buffer to test view wrapper
    data = np.ones((50, 1))
    s.buffer.push(data)

    frame = s.view(duration=0.5)
    assert isinstance(frame, DASFrame)
    assert frame.data.shape == (50, 1)
    assert frame.fs == 100.0
