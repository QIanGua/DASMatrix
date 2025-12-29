"""格式注册表单元测试"""

import tempfile
from pathlib import Path

import numpy as np

from DASMatrix.acquisition.formats import FormatMetadata, FormatPlugin, FormatRegistry


class TestFormatRegistry:
    """FormatRegistry 测试类"""

    def test_list_formats(self):
        """测试列出所有格式"""
        formats = FormatRegistry.list_formats()
        assert isinstance(formats, list)
        assert "DAT" in formats
        assert "H5" in formats
        assert "SEGY" in formats
        assert "MINISEED" in formats

    def test_get_plugin(self):
        """测试获取插件"""
        plugin = FormatRegistry.get("H5")
        assert plugin is not None
        assert plugin.format_name == "H5"

        plugin = FormatRegistry.get("h5")  # 小写也应该工作
        assert plugin is not None

        plugin = FormatRegistry.get("UNKNOWN")
        assert plugin is None

    def test_list_extensions(self):
        """测试列出扩展名"""
        ext_map = FormatRegistry.list_extensions()
        assert isinstance(ext_map, dict)
        assert ".h5" in ext_map
        assert ".dat" in ext_map
        assert ext_map[".h5"] == "H5"
        assert ext_map[".dat"] == "DAT"

    def test_detect_format_by_extension(self):
        """测试通过扩展名检测格式"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = Path(f.name)
            f.write(b"\x89HDF\r\n\x1a\n" + b"\x00" * 100)  # HDF5 魔数

        try:
            fmt = FormatRegistry.detect_format(temp_path)
            assert fmt == "H5"
        finally:
            temp_path.unlink()

    def test_detect_format_dat(self):
        """测试检测 DAT 格式"""
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            temp_path = Path(f.name)
            f.write(b"\x00" * 1600)  # 模拟 DAT 数据

        try:
            fmt = FormatRegistry.detect_format(temp_path)
            assert fmt == "DAT"
        finally:
            temp_path.unlink()

    def test_detect_format_nonexistent(self):
        """测试检测不存在的文件"""
        fmt = FormatRegistry.detect_format("/nonexistent/file.h5")
        assert fmt is None


class TestFormatPlugin:
    """FormatPlugin 测试类"""

    def test_plugin_attributes(self):
        """测试插件属性"""
        plugin = FormatRegistry.get("DAT")
        assert plugin is not None
        assert plugin.format_name == "DAT"
        assert plugin.version == "1.0.0"
        assert ".dat" in plugin.file_extensions

    def test_plugin_can_read(self):
        """测试插件 can_read 方法"""
        plugin = FormatRegistry.get("DAT")
        assert plugin is not None

        # 创建临时 DAT 文件
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            temp_path = Path(f.name)
            f.write(b"\x00" * 1600)

        try:
            assert plugin.can_read(temp_path) is True
        finally:
            temp_path.unlink()

    def test_plugin_scan(self):
        """测试插件 scan 方法"""
        plugin = FormatRegistry.get("DAT")
        assert plugin is not None

        # 创建临时 DAT 文件 (800 通道 × 100 采样 × 2 字节)
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            temp_path = Path(f.name)
            data = np.zeros((100, 800), dtype=np.int16)
            data.tofile(f)

        try:
            meta = plugin.scan(temp_path)
            assert isinstance(meta, FormatMetadata)
            assert meta.n_samples == 100
            assert meta.n_channels == 800
            assert meta.format_name == "DAT"
        finally:
            temp_path.unlink()

    def test_plugin_read(self):
        """测试插件 read 方法"""
        plugin = FormatRegistry.get("DAT")
        assert plugin is not None

        # 创建临时 DAT 文件
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            temp_path = Path(f.name)
            data = np.arange(800 * 100, dtype=np.int16).reshape((100, 800))
            data.tofile(f)

        try:
            result = plugin.read(temp_path, lazy=False)
            assert result is not None
            assert result.shape == (100, 800)
            assert hasattr(result, "dims")
            assert "time" in result.dims  # type: ignore[operator]
            assert "channel" in result.dims  # type: ignore[operator]
        finally:
            temp_path.unlink()


class TestFormatMetadata:
    """FormatMetadata 测试类"""

    def test_metadata_creation(self):
        """测试元数据创建"""
        meta = FormatMetadata(
            n_samples=10000,
            n_channels=800,
            sampling_rate=5000.0,
            format_name="TEST",
        )
        assert meta.n_samples == 10000
        assert meta.n_channels == 800
        assert meta.sampling_rate == 5000.0
        assert meta.format_name == "TEST"

    def test_metadata_optional_fields(self):
        """测试元数据可选字段"""
        meta = FormatMetadata(
            n_samples=10000,
            n_channels=800,
            sampling_rate=5000.0,
            channel_spacing=1.0,
            gauge_length=10.0,
            start_time="2024-01-01T00:00:00",
        )
        assert meta.channel_spacing == 1.0
        assert meta.gauge_length == 10.0
        assert meta.start_time == "2024-01-01T00:00:00"


class TestCustomPlugin:
    """自定义插件测试类"""

    def test_register_custom_plugin(self):
        """测试注册自定义插件"""

        class TestFormatPlugin(FormatPlugin):
            format_name = "TESTFORMAT"
            version = "1.0.0"
            file_extensions = (".test",)
            priority = 100

            def can_read(self, path: Path) -> bool:
                return path.suffix.lower() == ".test"

            def scan(self, path: Path) -> FormatMetadata:
                return FormatMetadata(
                    n_samples=0,
                    n_channels=0,
                    sampling_rate=1000.0,
                    format_name=self.format_name,
                )

            def read(  # type: ignore[override]
                self,
                path: Path,
                channels=None,
                time_slice=None,
                lazy: bool = True,
                **kwargs,
            ):
                return None

        # 注册插件
        FormatRegistry.register(TestFormatPlugin())

        # 验证注册成功
        assert "TESTFORMAT" in FormatRegistry.list_formats()
        plugin = FormatRegistry.get("TESTFORMAT")
        assert plugin is not None
        assert plugin.format_name == "TESTFORMAT"

        # 清理
        FormatRegistry.unregister("TESTFORMAT")
        assert "TESTFORMAT" not in FormatRegistry.list_formats()
