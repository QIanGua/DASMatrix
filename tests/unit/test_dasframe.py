"""DASFrame 单元测试"""

import numpy as np
import pytest
import xarray as xr

from DASMatrix.api import df
from DASMatrix.api.dasframe import DASFrame


class TestDASFrameCreation:
    """测试 DASFrame 创建"""

    def test_create_from_array(self):
        """测试从数组创建 DASFrame"""
        data = np.random.randn(1000, 64)
        frame = DASFrame(data, fs=10000)

        assert frame._data.shape == (1000, 64)
        assert frame._fs == 10000

    def test_create_from_df_api(self):
        """测试通过 df.from_array 创建"""
        data = np.random.randn(1000, 64)
        frame = df.from_array(data, fs=10000)

        assert frame._data.shape == (1000, 64)
        assert frame._fs == 10000

    def test_create_1d_array(self):
        """测试从一维数组创建（自动扩展为二维）"""
        data = np.random.randn(1000)
        frame = df.from_array(data, fs=10000)

        assert frame._data.shape == (1000, 1)

    def test_xarray_initialization(self):
        """测试 xarray DataArray 初始化和分块"""
        data = np.random.randn(1000, 10)
        frame = DASFrame(data, fs=1000, dx=1.0)

        # 验证底层数据是 xarray DataArray
        assert isinstance(frame.data, xr.DataArray)
        assert frame.data.shape == data.shape
        assert frame.data.dims == ("time", "distance")
        # 验证数据已分块
        assert frame.data.chunks is not None


class TestDASFrameBasicOperations:
    """测试 DASFrame 基本操作"""

    @pytest.fixture
    def sample_frame(self):
        """创建测试用的 DASFrame"""
        # 创建包含正弦波的测试数据
        t = np.linspace(0, 1, 10000)
        data = np.sin(2 * np.pi * 50 * t)  # 50Hz 正弦波
        data = np.tile(data.reshape(-1, 1), (1, 8))  # 8 通道
        return DASFrame(data, fs=10000)

    def test_slice(self, sample_frame):
        """测试切片操作"""
        sliced = sample_frame.slice(t=slice(0, 1000), x=slice(0, 4))
        result = sliced.collect()

        assert result.shape == (1000, 4)

    def test_detrend(self, sample_frame):
        """测试去趋势"""
        result = sample_frame.detrend().collect()

        assert result.shape == sample_frame._data.shape

    def test_normalize(self, sample_frame):
        """测试归一化"""
        result = sample_frame.normalize(method="zscore").collect()

        # Z-score 归一化后均值应接近 0，标准差应接近 1
        assert np.abs(result.mean()) < 1e-10
        assert np.allclose(result.std(axis=0), 1, atol=1e-5)

    def test_normalize_minmax(self, sample_frame):
        """测试 MinMax 归一化"""
        result = sample_frame.normalize(method="minmax").collect()

        # MinMax 归一化后范围应为 [-1, 1]
        assert np.allclose(result.min(axis=0), -1, atol=1e-5)
        assert np.allclose(result.max(axis=0), 1, atol=1e-5)

    def test_demean(self, sample_frame):
        """测试去均值"""
        # 添加一个偏移量
        frame = DASFrame(sample_frame._data + 10.0, fs=sample_frame._fs)
        result = frame.demean().collect()

        # 去均值后均值应接近 0
        assert np.abs(result.mean(axis=0)).max() < 1e-10


class TestDASFrameFiltering:
    """测试 DASFrame 滤波功能"""

    @pytest.fixture
    def sample_frame(self):
        """创建包含多频率成分的测试数据"""
        t = np.linspace(0, 1, 10000)
        # 50Hz + 150Hz + 300Hz
        data = (
            np.sin(2 * np.pi * 50 * t)
            + 0.5 * np.sin(2 * np.pi * 150 * t)
            + 0.3 * np.sin(2 * np.pi * 300 * t)
        )
        data = data.reshape(-1, 1)
        return DASFrame(data, fs=10000)

    def test_lowpass(self, sample_frame):
        """测试低通滤波"""
        result = sample_frame.lowpass(cutoff=100).collect()

        assert result.shape == sample_frame._data.shape

    def test_highpass(self, sample_frame):
        """测试高通滤波"""
        result = sample_frame.highpass(cutoff=100).collect()

        assert result.shape == sample_frame._data.shape

    def test_bandpass(self, sample_frame):
        """测试带通滤波"""
        result = sample_frame.bandpass(low=40, high=60).collect()

        assert result.shape == sample_frame._data.shape

    def test_notch(self, sample_frame):
        """测试陷波滤波"""
        result = sample_frame.notch(freq=50).collect()

        assert result.shape == sample_frame._data.shape


class TestDASFrameTransforms:
    """测试 DASFrame 变换功能"""

    @pytest.fixture
    def sample_frame(self):
        """创建测试用的 DASFrame"""
        t = np.linspace(0, 1, 1024)
        data = np.sin(2 * np.pi * 50 * t).reshape(-1, 1)
        return DASFrame(data, fs=1024)

    def test_fft(self, sample_frame):
        """测试 FFT"""
        result = sample_frame.fft().collect()

        # FFT 结果应该是复数或能量谱（取决于实现）
        assert result is not None

    def test_fft_frequency_peak(self):
        """测试 FFT 频率峰值位置"""
        # 创建纯 10Hz 信号
        nt = 1000
        t = np.linspace(0, 1, nt)
        data = np.sin(2 * np.pi * 10 * t)[:, None] * np.ones((1, 5))
        fs = 1000.0

        frame = DASFrame(data, fs=fs)
        spectrum = frame.fft()

        # 验证返回类型
        assert isinstance(spectrum, DASFrame)
        # 验证维度变化：time -> frequency
        assert "frequency" in spectrum.data.dims
        assert "time" not in spectrum.data.dims

        # 计算并验证峰值在 10Hz
        spec_val = spectrum.data.compute()
        freqs = spec_val.frequency.values
        peak_idx = spec_val.argmax(dim="frequency")
        peak_freqs = freqs[peak_idx]

        # 验证峰值在 10Hz 附近
        assert np.allclose(np.abs(peak_freqs), 10.0, atol=1.0)

    def test_stft(self, sample_frame):
        """测试 STFT"""
        result = sample_frame.stft(nperseg=256).collect()

        # STFT 结果应该是三维数组
        assert result.ndim == 3

    def test_hilbert(self, sample_frame):
        """测试 Hilbert 变换"""
        result = sample_frame.hilbert().collect()

        assert np.iscomplexobj(result)

    def test_envelope(self, sample_frame):
        """测试包络提取"""
        result = sample_frame.envelope().collect()

        # 包络应该是非负的
        assert (result >= 0).all()


class TestDASFrameStatistics:
    """测试 DASFrame 统计功能"""

    @pytest.fixture
    def sample_frame(self):
        """创建测试用的 DASFrame"""
        np.random.seed(42)
        data = np.random.randn(1000, 8)
        return DASFrame(data, fs=1000)

    def test_mean(self, sample_frame):
        """测试均值计算"""
        result = sample_frame.mean()

        assert isinstance(result, (np.ndarray, float))

    def test_std(self, sample_frame):
        """测试标准差计算"""
        result = sample_frame.std()

        assert isinstance(result, (np.ndarray, float))

    def test_max(self, sample_frame):
        """测试最大值"""
        result = sample_frame.max()

        assert isinstance(result, (np.ndarray, float))

    def test_min(self, sample_frame):
        """测试最小值"""
        result = sample_frame.min()

        assert isinstance(result, (np.ndarray, float))

    def test_rms(self, sample_frame):
        """测试 RMS 计算"""
        result = sample_frame.rms(window=100)

        # RMS 计算结果应该有有效值
        assert result is not None


class TestDASFrameChaining:
    """测试 DASFrame 链式操作"""

    def test_chain_operations(self):
        """测试链式操作"""
        t = np.linspace(0, 1, 10000)
        data = np.sin(2 * np.pi * 50 * t).reshape(-1, 1)
        data = np.tile(data, (1, 4))

        frame = DASFrame(data, fs=10000)

        # 链式操作
        result = frame.detrend().bandpass(low=10, high=100).normalize().collect()

        assert result.shape == data.shape

    def test_chain_with_stft(self):
        """测试带 STFT 的链式操作"""
        t = np.linspace(0, 1, 4096)
        data = np.sin(2 * np.pi * 50 * t).reshape(-1, 1)

        frame = DASFrame(data, fs=4096)

        result = frame.detrend().bandpass(low=10, high=100).stft(nperseg=256).collect()

        assert result.ndim == 3


class TestDASFrameDetection:
    """测试 DASFrame 检测功能"""

    def test_threshold_detect(self):
        """测试阈值检测"""
        data = np.zeros((1000, 4))
        data[500:510, :] = 10.0  # 添加突发事件

        frame = DASFrame(data, fs=1000)
        result = frame.threshold_detect()

        # 应该检测到事件
        assert result.any()
