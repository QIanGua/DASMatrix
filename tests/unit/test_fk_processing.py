import numpy as np
import pytest

from DASMatrix.api.dasframe import DASFrame
from DASMatrix.config.sampling_config import SamplingConfig
from DASMatrix.processing.das_processor import DASProcessor


@pytest.fixture
def sample_data():
    """生成带有线性同相轴（特定速度）的合成数据"""
    nt = 512
    nx = 50
    dt = 0.001
    dx = 1.0

    t = np.arange(nt) * dt
    x = np.arange(nx) * dx

    # 生成一个视速度为 1000 m/s 的信号
    # f(t - x/v)
    v = 1000.0
    data = np.zeros((nt, nx))

    # Ricker wavelet (Mexican hat) to ensure zero mean and reduce DC effects
    for i in range(nx):
        delay = x[i] / v
        t_shifted = t - 0.2 - delay
        # Ricker
        f0 = 30.0  # Hz
        a = (np.pi * f0 * t_shifted) ** 2
        data[:, i] = (1 - 2 * a) * np.exp(-a)

    return data, nt, nx, dt, dx


@pytest.fixture
def das_processor():
    config = SamplingConfig(fs=1000.0, channels=50)
    return DASProcessor(config)


class TestFKProcessing:
    def test_fk_transform_shape(self, das_processor, sample_data):
        data, nt, nx, dt, dx = sample_data
        fk, freqs, k = das_processor.f_k_transform(data)

        assert fk.shape == (nt, nx)
        assert freqs.shape == (nt,)
        assert k.shape == (nx,)

        # 0频应该在中心
        assert np.isclose(freqs[nt // 2], 0)
        assert np.isclose(k[nx // 2], 0)

    def test_inverse_fk_transform(self, das_processor, sample_data):
        data, _, _, _, _ = sample_data
        fk, _, _ = das_processor.f_k_transform(data)
        data_recovered = das_processor.iwf_k_transform(fk)

        # 验证重建误差很小
        assert np.allclose(data, data_recovered, atol=1e-10)

    def test_fk_filter_pass(self, das_processor, sample_data):
        """测试 FK 滤波保留目标信号"""
        data, _, _, _, dx = sample_data
        # 信号速度 1000 m/s
        # 滤波保留 > 500 m/s
        filtered_data = das_processor.fk_filter(data, v_min=500, v_max=None, dx=dx)

        # 应该保留大部分能量
        energy_in = np.sum(data**2)
        energy_out = np.sum(filtered_data**2)

        # 由于边缘效应和窗口，可能会有些损失，但应保留绝大部分
        assert energy_out / energy_in > 0.9

    def test_fk_filter_reject(self, das_processor, sample_data):
        """测试 FK 滤波去除目标信号"""
        data, _, _, _, dx = sample_data
        # 信号速度 1000 m/s
        # 滤波保留 < 500 m/s (应该去除 1000 m/s 的信号)
        # 注意：FKFilter 现在逻辑是 v_min 是最小保留速度。
        # 如果 v_min=500, v_max=None，则保留 |v|>500。信号 1000 在保留区。
        # 如果要去除 1000，我们需要保留 |v| < 500 => v_max=500

        filtered_data = das_processor.fk_filter(data, v_min=None, v_max=500, dx=dx)

        energy_in = np.sum(data**2)
        energy_out = np.sum(filtered_data**2)

        # 应该去除大部分能量
        assert energy_out / energy_in < 0.1


class TestDASFrameFK:
    def test_dasframe_fk_filter(self, sample_data):
        data, nt, nx, dt, dx = sample_data
        df = DASFrame(data, fs=1.0 / dt)

        # 链式调用
        # 1. 速度滤波 v > 500 (应该保留)
        result_pass = df.fk_filter(v_min=500, dx=dx).collect()

        # 2. 速度滤波 v < 500 (应该去除)
        result_reject = df.fk_filter(v_max=500, dx=dx).collect()

        energy_in = np.sum(data**2)
        energy_pass = np.sum(result_pass**2)
        energy_reject = np.sum(result_reject**2)

        assert energy_pass / energy_in > 0.9
        assert energy_reject / energy_in < 0.1

    def test_dasframe_plot_fk(self, sample_data):
        data, nt, nx, dt, dx = sample_data
        df = DASFrame(data, fs=1.0 / dt)

        # Test that plot_fk runs and returns a figure
        try:
            import matplotlib.pyplot as plt

            fig = df.plot_fk(dx=dx, title="Test FK")
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plot_fk failed with error: {e}")
