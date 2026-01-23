"""示例数据模块测试。"""

import numpy as np
import pytest


class TestGetExampleFrame:
    """测试 get_example_frame 函数。"""

    def test_random_das(self):
        """测试随机 DAS 数据生成。"""
        from DASMatrix.examples import get_example_frame

        frame = get_example_frame("random_das")
        assert frame is not None
        assert frame.shape == (10000, 100)

    def test_sine_wave(self):
        """测试正弦波数据生成。"""
        from DASMatrix.examples import get_example_frame

        frame = get_example_frame("sine_wave")
        assert frame is not None
        assert frame.shape == (10000, 100)

    def test_chirp(self):
        """测试 chirp 数据生成。"""
        from DASMatrix.examples import get_example_frame

        frame = get_example_frame("chirp")
        assert frame is not None
        assert frame.shape == (10000, 100)

    def test_impulse(self):
        """测试脉冲数据生成。"""
        from DASMatrix.examples import get_example_frame

        frame = get_example_frame("impulse")
        assert frame is not None
        assert frame.shape == (10000, 100)

    def test_event(self):
        """测试事件数据生成。"""
        from DASMatrix.examples import get_example_frame

        frame = get_example_frame("event")
        assert frame is not None
        assert frame.shape == (10000, 100)

    def test_custom_shape(self):
        """测试自定义形状。"""
        from DASMatrix.examples import get_example_frame

        frame = get_example_frame("random_das", n_samples=5000, n_channels=50)
        assert frame.shape == (5000, 50)

    def test_custom_fs(self):
        """测试自定义采样率。"""
        from DASMatrix.examples import get_example_frame

        frame = get_example_frame("random_das", fs=2000.0)
        assert frame.fs == 2000.0

    def test_reproducibility(self):
        """测试可重复性。"""
        from DASMatrix.examples import get_example_frame

        frame1 = get_example_frame("random_das", seed=123)
        frame2 = get_example_frame("random_das", seed=123)
        np.testing.assert_array_equal(frame1.collect(), frame2.collect())

    def test_unknown_type_raises_error(self):
        """测试未知类型抛出错误。"""
        from DASMatrix.examples import get_example_frame

        with pytest.raises(ValueError, match="Unknown example type"):
            get_example_frame("unknown_type")  # type: ignore


class TestGetExampleSpool:
    """测试 get_example_spool 函数。"""

    def test_diverse_das_spool(self):
        """测试生成多样化 Spool。"""
        from DASMatrix.examples import get_example_spool

        spool = get_example_spool("diverse_das")
        assert len(spool) == 3

    def test_continuous_spool(self):
        """测试生成连续 Spool。"""
        from DASMatrix.examples import get_example_spool

        spool = get_example_spool("continuous")
        assert len(spool) == 3

    def test_spool_iteration(self):
        """测试 Spool 迭代。"""
        from DASMatrix.examples import get_example_spool

        spool = get_example_spool("diverse_das", n_frames=2)
        frames = list(spool)
        assert len(frames) == 2

    def test_spool_indexing(self):
        """测试 Spool 索引。"""
        from DASMatrix.examples import get_example_spool

        spool = get_example_spool("diverse_das")
        frame = spool[0]
        assert frame is not None


class TestListExampleTypes:
    """测试 list_example_types 函数。"""

    def test_returns_dict(self):
        """测试返回字典。"""
        from DASMatrix.examples import list_example_types

        types = list_example_types()
        assert isinstance(types, dict)
        assert "frame" in types
        assert "spool" in types

    def test_frame_types(self):
        """测试帧类型列表。"""
        from DASMatrix.examples import list_example_types

        types = list_example_types()
        assert "random_das" in types["frame"]
        assert "sine_wave" in types["frame"]
        assert "chirp" in types["frame"]

    def test_spool_types(self):
        """测试 Spool 类型列表。"""
        from DASMatrix.examples import list_example_types

        types = list_example_types()
        assert "diverse_das" in types["spool"]
        assert "continuous" in types["spool"]


class TestImportFromModule:
    """测试从主模块导入。"""

    def test_import_from_dasmatrix(self):
        """测试从 DASMatrix 导入。"""
        from DASMatrix import get_example_frame, get_example_spool

        assert get_example_frame is not None
        assert get_example_spool is not None
