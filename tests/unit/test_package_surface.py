import importlib


def test_core_public_packages_importable():
    for name in [
        "DASMatrix.api",
        "DASMatrix.core",
        "DASMatrix.analysis",
        "DASMatrix.agent",
        "DASMatrix.utils",
    ]:
        assert importlib.import_module(name) is not None
