from pathlib import Path

import DASMatrix.visualization.web.server as web_server


def test_web_static_index_exists():
    static_index = Path(web_server.__file__).with_name("static") / "index.html"
    assert static_index.is_file()
