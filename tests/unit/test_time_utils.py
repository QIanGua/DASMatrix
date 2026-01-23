"""时间工具模块测试。"""

from datetime import datetime, timedelta

import numpy as np
import pytest


class TestToDatetime64:
    """测试 to_datetime64 函数。"""

    def test_from_string(self):
        """测试从字符串转换。"""
        from DASMatrix.utils.time import to_datetime64

        result = to_datetime64("2024-01-01T12:00:00")
        assert isinstance(result, np.datetime64)

    def test_from_datetime(self):
        """测试从 datetime 对象转换。"""
        from DASMatrix.utils.time import to_datetime64

        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = to_datetime64(dt)
        assert isinstance(result, np.datetime64)

    def test_from_timestamp(self):
        """测试从时间戳转换。"""
        from DASMatrix.utils.time import to_datetime64

        # Unix epoch
        result = to_datetime64(0)
        assert isinstance(result, np.datetime64)

    def test_from_datetime64(self):
        """测试从 datetime64 直接返回。"""
        from DASMatrix.utils.time import to_datetime64

        original = np.datetime64("2024-01-01")
        result = to_datetime64(original)
        assert result == original

    def test_from_none(self):
        """测试 None 返回 NaT。"""
        from DASMatrix.utils.time import to_datetime64

        result = to_datetime64(None)
        assert np.isnat(result)

    def test_invalid_type_raises(self):
        """测试无效类型抛出错误。"""
        from DASMatrix.utils.time import to_datetime64

        with pytest.raises(TypeError):
            to_datetime64([1, 2, 3])  # type: ignore


class TestToTimedelta64:
    """测试 to_timedelta64 函数。"""

    def test_from_int_seconds(self):
        """测试从整数秒转换。"""
        from DASMatrix.utils.time import to_timedelta64

        result = to_timedelta64(60)
        assert isinstance(result, np.timedelta64)
        # 60秒
        assert result == np.timedelta64(60, "s")

    def test_from_float_seconds(self):
        """测试从浮点数秒转换。"""
        from DASMatrix.utils.time import to_timedelta64

        result = to_timedelta64(1.5)
        assert isinstance(result, np.timedelta64)

    def test_from_string(self):
        """测试从字符串解析。"""
        from DASMatrix.utils.time import to_timedelta64

        result = to_timedelta64("10s")
        assert isinstance(result, np.timedelta64)

    def test_from_string_ms(self):
        """测试解析毫秒字符串。"""
        from DASMatrix.utils.time import to_timedelta64

        result = to_timedelta64("500ms")
        assert isinstance(result, np.timedelta64)

    def test_from_string_hours(self):
        """测试解析小时字符串。"""
        from DASMatrix.utils.time import to_timedelta64

        result = to_timedelta64("2h")
        assert isinstance(result, np.timedelta64)

    def test_from_timedelta(self):
        """测试从 timedelta 对象转换。"""
        from DASMatrix.utils.time import to_timedelta64

        td = timedelta(hours=1)
        result = to_timedelta64(td)
        assert isinstance(result, np.timedelta64)

    def test_from_timedelta64(self):
        """测试从 timedelta64 直接返回。"""
        from DASMatrix.utils.time import to_timedelta64

        original = np.timedelta64(100, "s")
        result = to_timedelta64(original)
        assert result == original

    def test_from_none(self):
        """测试 None 返回 NaT。"""
        from DASMatrix.utils.time import to_timedelta64

        result = to_timedelta64(None)
        assert np.isnat(result)

    def test_with_unit(self):
        """测试指定单位。"""
        from DASMatrix.utils.time import to_timedelta64

        result = to_timedelta64(1, "h")
        # 1 小时 = 3600 秒
        assert result / np.timedelta64(1, "s") == 3600


class TestToFloat:
    """测试 to_float 函数。"""

    def test_timedelta_to_seconds(self):
        """测试 timedelta 转秒。"""
        from DASMatrix.utils.time import to_float

        delta = np.timedelta64(1500, "ms")
        result = to_float(delta)
        assert abs(result - 1.5) < 0.01

    def test_timedelta_to_ms(self):
        """测试 timedelta 转毫秒。"""
        from DASMatrix.utils.time import to_float

        delta = np.timedelta64(1500, "ms")
        result = to_float(delta, "ms")
        assert abs(result - 1500.0) < 1.0

    def test_datetime_to_float(self):
        """测试 datetime 转浮点数。"""
        from DASMatrix.utils.time import to_float

        dt = np.datetime64("1970-01-01T00:00:01", "s")
        result = to_float(dt)
        # 1 秒从 epoch
        assert abs(result - 1.0) < 0.01

    def test_invalid_unit_raises(self):
        """测试无效单位抛出错误。"""
        from DASMatrix.utils.time import to_float

        delta = np.timedelta64(1, "s")
        with pytest.raises(ValueError):
            to_float(delta, "invalid")  # type: ignore


class TestImportFromModule:
    """测试从主模块导入。"""

    def test_import_from_dasmatrix(self):
        """测试从 DASMatrix 导入。"""
        from DASMatrix import to_datetime64, to_float, to_timedelta64

        assert to_datetime64 is not None
        assert to_timedelta64 is not None
        assert to_float is not None
