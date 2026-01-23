"""单位系统模块测试。"""


class TestUnitsImport:
    """测试单位导入。"""

    def test_basic_units_import(self):
        """测试基本单位可以导入。"""
        from DASMatrix.units import Hz, kHz, m, mm, ms, s

        assert m is not None
        assert s is not None
        assert ms is not None
        assert Hz is not None
        assert kHz is not None
        assert mm is not None

    def test_unit_registry(self):
        """测试单位注册表。"""
        from DASMatrix.units import ureg

        assert ureg is not None

    def test_quantity_class(self):
        """测试 Quantity 类。"""
        from DASMatrix.units import Quantity

        assert Quantity is not None


class TestGetQuantity:
    """测试 get_quantity 函数。"""

    def test_parse_simple_quantity(self):
        """测试解析简单数值+单位。"""
        from DASMatrix.units import get_quantity

        q = get_quantity("10 Hz")
        assert q.magnitude == 10
        assert str(q.units) == "hertz"

    def test_parse_with_prefix(self):
        """测试解析带前缀的单位。"""
        from DASMatrix.units import get_quantity

        q = get_quantity("5 kHz")
        assert q.magnitude == 5
        assert "kilohertz" in str(q.units)

    def test_parse_compound_unit(self):
        """测试解析复合单位。"""
        from DASMatrix.units import get_quantity

        q = get_quantity("100 m/s")
        assert q.magnitude == 100


class TestGetUnit:
    """测试 get_unit 函数。"""

    def test_get_meter(self):
        """测试获取米单位。"""
        from DASMatrix.units import get_unit

        unit = get_unit("meter")
        assert unit is not None

    def test_get_hertz(self):
        """测试获取赫兹单位。"""
        from DASMatrix.units import get_unit

        unit = get_unit("hertz")
        assert unit is not None


class TestUnitOperations:
    """测试单位运算。"""

    def test_quantity_multiplication(self):
        """测试数值与单位相乘。"""
        from DASMatrix.units import m

        distance = 100 * m
        assert distance.magnitude == 100

    def test_unit_conversion(self):
        """测试单位转换。"""
        from DASMatrix.units import km, m

        distance = 1 * km
        in_meters = distance.to(m)
        assert in_meters.magnitude == 1000

    def test_frequency_conversion(self):
        """测试频率转换。"""
        from DASMatrix.units import Hz, kHz

        freq = 1 * kHz
        in_hz = freq.to(Hz)
        assert in_hz.magnitude == 1000


class TestMagnitude:
    """测试 magnitude 函数。"""

    def test_magnitude_from_quantity(self):
        """测试从 Quantity 获取量值。"""
        from DASMatrix.units import Hz, magnitude

        freq = 10 * Hz
        assert magnitude(freq) == 10.0

    def test_magnitude_from_float(self):
        """测试从浮点数获取量值。"""
        from DASMatrix.units import magnitude

        assert magnitude(3.14) == 3.14


class TestToBaseUnits:
    """测试 to_base_units 函数。"""

    def test_km_to_base(self):
        """测试千米转换为基本单位。"""
        from DASMatrix.units import km, to_base_units

        distance = 1 * km
        base = to_base_units(distance)
        assert base.magnitude == 1000

    def test_kHz_to_base(self):
        """测试 kHz 转换为基本单位。"""
        from DASMatrix.units import kHz, to_base_units

        freq = 1 * kHz
        base = to_base_units(freq)
        assert base.magnitude == 1000
