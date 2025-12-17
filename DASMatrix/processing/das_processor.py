import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy import signal

from ..config.sampling_config import SamplingConfig


class DASProcessor:
    """DAS数据处理器，用于处理和分析DAS数据"""

    def __init__(self, sampling_config: SamplingConfig):
        """初始化DAS数据处理器

        Args:
            sampling_config: 采样配置对象，包含采样频率等信息
        """
        self.sampling_config = sampling_config
        self.fs = sampling_config.fs
        self.logger = logging.getLogger(__name__)

        # 预计算高通滤波器系数
        self.sos_highpass = self._create_filter(
            cutoff=self.sampling_config.wn, btype="high"
        )

    def _create_filter(
        self, cutoff: Union[float, List[float]], btype: str, order: int = 4
    ) -> np.ndarray:
        """创建Butterworth滤波器

        Args:
            cutoff: 截止频率 (Hz)。对于带通/带阻是 [low, high]。
            btype: 滤波器类型 ('high', 'low', 'bandpass', 'bandstop')
            order: 滤波器阶数

        Returns:
            np.ndarray: 滤波器系数 (SOS format)
        """
        return signal.butter(N=order, Wn=cutoff, btype=btype, fs=self.fs, output="sos")

    def ApplyFilter(self, data: np.ndarray, sos: np.ndarray) -> np.ndarray:
        """应用滤波器 (零相位)

        Args:
            data: 输入数据 (时间轴在 axis=0)
            sos: 滤波器系数 (SOS format)

        Returns:
            np.ndarray: 滤波后的数据
        """
        # 确保数据是浮点数类型以获得最佳精度
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float64)
        filtered_data = signal.sosfiltfilt(sos, data, axis=0)
        return filtered_data

    def NormalizeDC(self, data: np.ndarray) -> np.ndarray:
        """移除直流偏置 (按列)

        Args:
            data: 输入数据 (时间轴在 axis=0)

        Returns:
            np.ndarray: 去除直流偏置后的数据
        """
        return data - np.mean(data, axis=0, keepdims=True)

    def ProcessDifferential(self, raw_data: np.ndarray) -> np.ndarray:
        """处理原始数据，得到差分数据 (高通滤波和去直流)

        Args:
            raw_data: 原始数据 (时间轴在 axis=0)

        Returns:
            np.ndarray: 处理后的差分数据
        """
        # 高通滤波
        filtered_data = self.ApplyFilter(raw_data, self.sos_highpass)
        # DC偏置校正
        processed_data = self.NormalizeDC(filtered_data)
        return processed_data

    def IntegrateData(self, raw_data: np.ndarray) -> np.ndarray:
        """计算积分数据 (累加、高通滤波、去直流)

        Args:
            raw_data: 原始数据 (时间轴在 axis=0)

        Returns:
            np.ndarray: 计算得到的积分数据
        """
        # 计算累积和 (积分)
        int_data = np.cumsum(raw_data, axis=0)
        # 高通滤波
        filtered_int_data = self.ApplyFilter(int_data, self.sos_highpass)
        # DC偏置校正
        processed_int_data = self.NormalizeDC(filtered_int_data)
        return processed_int_data

    def ComputeSpectrum(
        self,
        data: np.ndarray,
        channel_index: int,
        window_size: int = 1024,
        overlap: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """计算指定通道信号的频谱

        Args:
            data: 输入数据 (时间轴在 axis=0)
            channel_index: 要计算频谱的通道索引
            window_size: STFT窗口大小
            overlap: STFT窗口重叠比例

        Returns:
            Dict[str, np.ndarray]: 包含频率('frequencies')和平均幅值('magnitudes')的字典
        """
        if channel_index < 0 or channel_index >= data.shape[1]:
            raise IndexError(
                f"通道索引 {channel_index} 超出范围 [0, {data.shape[1] - 1}]"
            )

        # 提取指定通道的数据
        channel_data = data[:, channel_index]

        # 计算短时傅里叶变换 (STFT)
        frequencies, times, spectrogram = signal.spectrogram(
            channel_data,
            fs=self.fs,
            window="hann",
            nperseg=window_size,
            noverlap=int(window_size * overlap),
            detrend="constant",
            scaling="spectrum",  # 返回功率谱密度
        )

        # 计算每个频率分量的平均幅值 (跨时间)
        # 注意：spectrogram 返回的是功率谱密度，需要取平方根得到幅值谱
        # 或者直接使用幅值谱 (scaling='spectrum') 返回的结果
        magnitudes = np.mean(np.abs(spectrogram), axis=1)

        return {"frequencies": frequencies, "magnitudes": magnitudes}

    def FindPeakFrequencies(
        self,
        spectrum: Dict[str, np.ndarray],
        min_freq: float = 0,
        max_freq: Optional[float] = None,
        n_peaks: int = 3,
        min_prominence: Optional[float] = None,  # 新增：峰值最小凸起度
    ) -> List[Dict[str, float]]:
        """在频谱中查找峰值频率

        Args:
            spectrum: 频谱数据字典，包含 'frequencies' 和 'magnitudes'
            min_freq: 最小频率限制 (Hz)
            max_freq: 最大频率限制 (Hz)，默认为None (使用最大频率)
            n_peaks: 返回的最大峰值数量
            min_prominence: 峰值的最小凸起度，用于过滤不显著的峰值

        Returns:
            List[Dict[str, float]]: 峰值列表，每个峰值包含 'frequency' 和 'magnitude'
        """
        frequencies = spectrum["frequencies"]
        magnitudes = spectrum["magnitudes"]

        if frequencies.size == 0 or magnitudes.size == 0:
            self.logger.warning("频谱数据为空，无法查找峰值")
            return []

        # 限制频率范围
        if max_freq is None:
            max_freq = frequencies[-1]

        mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        valid_freqs = frequencies[mask]
        valid_mags = magnitudes[mask]

        if valid_freqs.size == 0:
            self.logger.warning("在指定频率范围内没有数据点，无法查找峰值")
            return []

        # 查找峰值
        peaks, properties = signal.find_peaks(valid_mags, prominence=min_prominence)

        if len(peaks) == 0:
            self.logger.info("在指定范围内未找到显著峰值")
            return []

        # 按幅值排序峰值 (使用峰值处的幅值，而不是凸起度)
        peak_magnitudes = valid_mags[peaks]
        sorted_indices = np.argsort(peak_magnitudes)[::-1]  # 降序排序

        # 取前 n_peaks 个峰值
        top_peak_indices = peaks[sorted_indices[:n_peaks]]

        # 构建结果
        result = []
        for idx in top_peak_indices:
            result.append(
                {
                    "frequency": valid_freqs[idx],
                    "magnitude": valid_mags[idx],
                }
            )

        self.logger.debug(f"找到 {len(result)} 个峰值: {result}")
        return result

    def ApplyBandpassFilter(
        self, data: np.ndarray, low_freq: float, high_freq: float
    ) -> np.ndarray:
        """应用带通滤波器

        Args:
            data: 输入数据 (时间轴在 axis=0)
            low_freq: 低截止频率 (Hz)
            high_freq: 高截止频率 (Hz)

        Returns:
            np.ndarray: 滤波后的数据
        """
        # 设计带通滤波器
        sos_bandpass = self._create_filter(
            cutoff=[low_freq, high_freq], btype="bandpass"
        )

        # 应用滤波器
        filtered_data = self.ApplyFilter(data, sos_bandpass)
        return filtered_data

    def EvaluateFrequencyResponse(
        self, data: np.ndarray, target_frequencies: List[float], window_size: int = 1024
    ) -> List[Dict[str, Any]]:
        """分析特定频率下的响应

        Args:
            data: 输入数据 (时间轴在 axis=0)
            target_frequencies: 目标频率列表 (Hz)
            window_size: 用于频谱计算的窗口大小

        Returns:
            List[Dict[str, Any]]: 包含每个目标频率响应信息的列表
                                   每个字典包含: 'freq', 'max_response_channel',
                                   'peak_response_value', 'all_channel_responses' (RMS)
        """
        analysis_results = []

        for freq in target_frequencies:
            self.logger.debug(f"分析频率 {freq}Hz")
            # 对每个频率应用窄带带通滤波
            # 可以根据需要调整带宽，例如 +/- 5% 或固定带宽
            bandwidth_ratio = 0.10  # 例如 +/- 10%
            low_freq = freq * (1 - bandwidth_ratio / 2)
            high_freq = freq * (1 + bandwidth_ratio / 2)

            # 确保频率范围有效
            if low_freq < 0:
                low_freq = 1e-3  # 避免0Hz
            if high_freq > self.fs / 2:
                high_freq = self.fs / 2 - 1e-3  # 避免奈奎斯特频率

            if low_freq >= high_freq:
                self.logger.warning(
                    f"频率 {freq}Hz 计算出的带宽无效 [{low_freq:.2f}, {high_freq:.2f}]"
                    f"跳过此频率"
                )
                continue

            filtered_data = self.ApplyBandpassFilter(data, low_freq, high_freq)

            # 计算每个通道滤波后的均方根 (RMS) 值作为响应强度
            rms_values = np.sqrt(np.mean(filtered_data**2, axis=0))

            if rms_values.size == 0:
                self.logger.warning(f"频率 {freq}Hz 滤波后无有效响应数据")
                continue

            # 找出响应最强的通道及其RMS值
            max_response_channel = np.argmax(rms_values)
            peak_response_value = rms_values[max_response_channel]

            self.logger.debug(
                f"频率 {freq}Hz: 最大响应通道 {max_response_channel}, "
                f"峰值RMS {peak_response_value:.4f}"
            )

            analysis_results.append(
                {
                    "freq": freq,
                    "max_response_channel": int(max_response_channel),  # 确保是整数
                    "peak_response_value": peak_response_value,
                    "all_channel_responses": rms_values,  # 返回所有通道的RMS值
                }
            )

        return analysis_results
