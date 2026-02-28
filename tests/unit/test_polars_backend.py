import polars as pl
import pytest

from DASMatrix.processing.backends.polars_backend import PolarsBackend


def test_select_channels():
    meta = pl.DataFrame(
        {
            "channel_index": [0, 1, 2, 3],
            "quality": [0.1, 0.6, 0.4, 0.9],
        }
    )
    backend = PolarsBackend(meta)
    selected = backend.select_channels(pl.col("quality") > 0.5)
    assert selected == [1, 3]


def test_filter_not_implemented():
    meta = pl.DataFrame({"channel_index": [0]})
    backend = PolarsBackend(meta)
    with pytest.raises(NotImplementedError):
        backend.filter("channel_index > 0")
