import numpy as np

from DASMatrix.core.ring_buffer import RingBuffer


def test_initialization():
    rb = RingBuffer(capacity=100, n_channels=5)
    assert rb.capacity == 100
    assert rb.n_channels == 5
    assert rb.is_full is False
    assert rb._size == 0
    # Check underlying buffer shape
    assert rb._buffer.shape == (100, 5)


def test_push_contiguous():
    rb = RingBuffer(capacity=10, n_channels=1)
    data = np.arange(5).reshape(-1, 1).astype(np.float32)

    rb.push(data)
    assert rb._size == 5
    assert not rb.is_full

    # Check content
    view = rb.view()
    np.testing.assert_array_equal(view, data)


def test_push_wrapping():
    # Capacity 10
    rb = RingBuffer(capacity=10, n_channels=1)

    # 1. Fill 8 samples
    data1 = np.arange(8).reshape(-1, 1).astype(np.float32)
    rb.push(data1)
    assert rb._size == 8

    # 2. Push 4 more samples (should wrap around)
    # Total pushed 12, capacity 10. Buffer should hold last 10: [2..11]
    data2 = np.arange(8, 12).reshape(-1, 1).astype(np.float32)
    rb.push(data2)

    assert rb._size == 10
    assert rb.is_full

    view = rb.view()
    expected = np.arange(2, 12).reshape(-1, 1).astype(np.float32)
    np.testing.assert_array_equal(view, expected)


def test_push_overflow():
    # Push data larger than capacity
    rb = RingBuffer(capacity=5, n_channels=2)
    data = np.zeros((10, 2))
    # Mark the last 5 rows to identify them
    data[5:, :] = 1

    rb.push(data)

    assert rb._size == 5
    assert rb.is_full

    view = rb.view()
    # Should be the last 5 rows (all ones)
    expected = np.ones((5, 2))
    np.testing.assert_array_equal(view, expected)


def test_view_subset():
    rb = RingBuffer(capacity=10, n_channels=1)
    data = np.arange(10).reshape(-1, 1).astype(np.float32)
    rb.push(data)

    # Get last 3 samples
    view_recent = rb.view(samples=3)
    expected = np.arange(7, 10).reshape(-1, 1).astype(np.float32)
    np.testing.assert_array_equal(view_recent, expected)


def test_clear():
    rb = RingBuffer(capacity=10, n_channels=1)
    rb.push(np.ones((5, 1)))
    assert rb._size == 5

    rb.clear()
    assert rb._size == 0
    assert rb.view().shape == (0, 1)


def test_initialization_failure():
    # Optional: check if invalid arguments raise error (if implementation adds checks)
    pass
