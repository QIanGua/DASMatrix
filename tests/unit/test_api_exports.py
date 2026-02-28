import DASMatrix


def test_stream_export_is_callable():
    assert callable(DASMatrix.stream)


def test_stream_class_exported():
    assert DASMatrix.Stream.__name__ == "Stream"
