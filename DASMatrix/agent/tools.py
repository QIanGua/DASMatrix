"""DAS 数据分析工具集。

为 AI Agent 提供的工具函数，支持数据读取、信号处理、可视化等操作。
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

import numpy as np

from ..ml.model import ONNXModel, TorchModel
from ..ml.pipeline import InferencePipeline
from .session import AgentSession


class DASAgentTools:
    """DAS 数据分析工具集。

    提供 AI Agent 可调用的工具函数，实现自然语言驱动的数据分析。

    Example:
        >>> tools = DASAgentTools()
        >>> result = tools.read_das_data("/path/to/data.h5")
        >>> print(result)
        {'id': 'data_a1b2c3d4', 'shape': [10000, 800], 'fs': 5000.0}
    """

    def __init__(self, session: Optional[AgentSession] = None) -> None:
        """初始化工具集。

        Args:
            session: 可选的会话管理器，用于多轮对话
        """
        self.session = session or AgentSession()

    # ========== 数据读取工具 ==========

    def read_das_data(
        self,
        path: str,
        channels: Optional[List[int]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """读取 DAS 数据文件。

        支持自动格式检测，可读取 H5, DAT, Zarr, PRODML 等多种格式。

        Args:
            path: 数据文件路径或通配符模式
            channels: 可选，指定读取的通道索引列表
            time_range: 可选，时间范围 (start_seconds, end_seconds)
            format: 可选，强制指定格式

        Returns:
            包含数据 ID 和元信息的字典:
            - id: 数据对象的唯一标识符，用于后续操作
            - shape: 数据形状 [n_samples, n_channels]
            - fs: 采样频率 (Hz)
            - duration: 数据时长 (秒)
        """
        from DASMatrix import read

        # 读取数据
        data = read(path, format=format)

        # 应用通道切片
        if channels:
            data = data.slice(x=slice(min(channels), max(channels) + 1))

        # 应用时间切片 (如果指定)
        if time_range:
            start_sample = int(time_range[0] * data.fs)
            end_sample = int(time_range[1] * data.fs)
            data = data.slice(t=slice(start_sample, end_sample))

        # 存储到会话
        metadata = {
            "source": path,
            "original_channels": channels,
            "time_range": time_range,
        }
        data_id = self.session.store(data, metadata)

        return {
            "id": data_id,
            "shape": list(data.shape),
            "fs": float(data.fs),
            "duration": data.shape[0] / data.fs,
            "n_channels": data.shape[1],
        }

    def list_das_files(
        self,
        directory: str,
        pattern: str = "*",
        recursive: bool = False,
    ) -> Dict[str, Any]:
        """列出目录中的 DAS 数据文件。

        Args:
            directory: 目录路径
            pattern: 文件匹配模式，默认 "*"
            recursive: 是否递归搜索子目录

        Returns:
            文件列表及统计信息
        """
        from DASMatrix.acquisition.formats import FormatRegistry

        dir_path = Path(directory)
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))

        # 检测格式
        file_infos = []
        for f in files[:50]:  # 限制返回数量
            if f.is_file():
                detected = FormatRegistry.detect_format(f)
                if detected:
                    file_infos.append(
                        {
                            "path": str(f),
                            "name": f.name,
                            "format": detected,
                            "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                        }
                    )

        return {
            "directory": directory,
            "pattern": pattern,
            "total_files": len(file_infos),
            "files": file_infos,
        }

    # ========== 信号处理工具 ==========

    def process_signal(
        self,
        data_id: str,
        operations: List[Dict[str, Any]],
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """应用信号处理流水线。

        Args:
            data_id: 输入数据的 ID
            operations: 处理操作列表，每个操作是一个字典:
                - op: 操作名称 (detrend, bandpass, highpass, lowpass,
                      normalize, stft, fft, envelope)
                - 其他参数取决于具体操作

                示例:
                [
                    {"op": "detrend"},
                    {"op": "bandpass", "low": 10, "high": 100},
                    {"op": "normalize", "method": "zscore"}
                ]
            output_name: 可选，输出数据的描述性名称

        Returns:
            处理结果信息，包含新数据的 ID
        """
        data = self.session.get(data_id)
        result = data

        applied_ops = []
        for op_config in operations:
            op_name = op_config.pop("op")
            if not hasattr(result, op_name):
                raise ValueError(f"Unknown operation: {op_name}")

            method = getattr(result, op_name)
            result = method(**op_config)
            applied_ops.append({"op": op_name, **op_config})
            op_config["op"] = op_name  # 恢复，避免修改原始参数

        # 存储结果
        metadata = {
            "source_id": data_id,
            "operations": applied_ops,
            "name": output_name,
        }
        result_id = self.session.store(result, metadata)

        return {
            "id": result_id,
            "shape": list(result.shape),
            "operations_applied": len(applied_ops),
            "pipeline": applied_ops,
        }

    def compute_spectrum(
        self,
        data_id: str,
        channel: int = 0,
        window_size: int = 1024,
        overlap: float = 0.5,
    ) -> Dict[str, Any]:
        """计算指定通道的频谱。

        Args:
            data_id: 数据 ID
            channel: 通道索引
            window_size: FFT 窗口大小
            overlap: 窗口重叠比例

        Returns:
            频谱信息，包含主要频率成分
        """
        from DASMatrix import DASProcessor, SamplingConfig

        data = self.session.get(data_id)
        arr = data.collect()

        config = SamplingConfig(fs=int(data.fs))
        processor = DASProcessor(config)

        spectrum = processor.compute_spectrum(arr, channel, window_size=window_size, overlap=overlap)

        # 找峰值
        peaks = processor.find_peak_frequencies(spectrum, n_peaks=5)

        return {
            "channel": channel,
            "window_size": window_size,
            "frequency_range": [0, data.fs / 2],
            "peak_frequencies": [{"frequency_hz": p["frequency"], "magnitude_db": p["magnitude"]} for p in peaks],
            "dominant_frequency_hz": peaks[0]["frequency"] if peaks else None,
        }

    def detect_events(
        self,
        data_id: str,
        threshold_db: float = -30,
        min_duration_ms: float = 10,
    ) -> Dict[str, Any]:
        """检测数据中的异常事件。

        Args:
            data_id: 数据 ID
            threshold_db: 检测阈值 (dB)
            min_duration_ms: 最小事件持续时间 (毫秒)

        Returns:
            检测到的事件列表
        """
        data = self.session.get(data_id)

        # 计算 RMS 能量
        arr = data.collect()
        rms = np.sqrt(np.mean(arr**2, axis=1))
        rms_db = 20 * np.log10(rms / np.max(rms) + 1e-10)

        # 简单阈值检测
        above_threshold = rms_db > threshold_db

        # 找事件区间
        events = []
        in_event = False
        event_start = 0

        for i, is_above in enumerate(above_threshold):
            if is_above and not in_event:
                event_start = i
                in_event = True
            elif not is_above and in_event:
                duration_ms = (i - event_start) / data.fs * 1000
                if duration_ms >= min_duration_ms:
                    events.append(
                        {
                            "start_time_s": event_start / data.fs,
                            "end_time_s": i / data.fs,
                            "duration_ms": duration_ms,
                            "peak_amplitude_db": float(np.max(rms_db[event_start:i])),
                        }
                    )
                in_event = False

        return {
            "threshold_db": threshold_db,
            "min_duration_ms": min_duration_ms,
            "events_detected": len(events),
            "events": events[:20],  # 限制返回数量
        }

    # ========== AI 推理工具 ==========

    def run_inference(
        self,
        data_id: str,
        model_path: str,
        backend: Literal["torch", "onnx"] = "torch",
        device: str = "cpu",
        preprocess_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """应用 AI 模型对 DAS 数据进行推理。

        Args:
            data_id: 数据 ID
            model_path: 模型文件路径 (.pth 或 .onnx)
            backend: 推理后端 ('torch' 或 'onnx')
            device: 运行设备 ('cpu' 或 'cuda')
            preprocess_config: 可选的预处理配置 (如标准化方法)

        Returns:
            推理结果摘要
        """
        data = self.session.get(data_id)

        # 初始化模型
        if backend == "torch":
            model = TorchModel(model_path, device=device)
        else:
            model = ONNXModel(model_path, device=device)

        # 构建流水线 (简单包装)
        pipeline = InferencePipeline(model)

        # 执行推理
        result = data.predict(pipeline)

        # 结果可能很大，Agent 通常只需要摘要或概率
        summary = {
            "model_path": model_path,
            "backend": backend,
            "output_shape": list(result.shape) if hasattr(result, "shape") else str(type(result)),
        }

        # 如果是分类模型，返回最大概率类别 (模拟逻辑)
        if result.ndim >= 1:
            summary["max_val"] = float(np.max(result))
            summary["min_val"] = float(np.min(result))
            if result.ndim == 2:  # [Batch, Classes]
                summary["predicted_class"] = int(np.argmax(result, axis=1)[0])

        return summary

    # ========== 统计分析工具 ==========

    def get_data_stats(
        self,
        data_id: str,
        per_channel: bool = False,
    ) -> Dict[str, Any]:
        """获取数据统计信息。

        Args:
            data_id: 数据 ID
            per_channel: 是否按通道统计

        Returns:
            统计信息字典
        """
        data = self.session.get(data_id)
        arr = data.collect()

        if per_channel:
            stats = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
            }
        else:
            stats = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "rms": float(np.sqrt(np.mean(arr**2))),
            }

        return {
            "id": data_id,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "statistics": stats,
            "per_channel": per_channel,
        }

    # ========== 可视化工具 ==========

    def create_visualization(
        self,
        data_id: str,
        plot_type: Literal["waterfall", "spectrum", "waveform", "spectrogram"],
        output_path: Optional[str] = None,
        channels: Optional[List[int]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """生成数据可视化。

        Args:
            data_id: 数据 ID
            plot_type: 图表类型
                - waterfall: 瀑布图 (时间 x 通道 热力图)
                - spectrum: 频谱图
                - waveform: 时域波形图
                - spectrogram: 时频谱图
            output_path: 可选，图片保存路径
            channels: 可选，绘制的通道
            time_range: 可选，时间范围
            **kwargs: 传递给绑图函数的其他参数

        Returns:
            包含图片路径的结果
        """
        from DASMatrix.visualization import (
            SpectrogramPlot,
            SpectrumPlot,
            WaterfallPlot,
            WaveformPlot,
        )

        data = self.session.get(data_id)
        arr = data.collect()

        # 应用切片
        if channels:
            arr = arr[:, channels]
        if time_range:
            start = int(time_range[0] * data.fs)
            end = int(time_range[1] * data.fs)
            arr = arr[start:end, :]

        # 选择绑图器
        plotters = {
            "waterfall": WaterfallPlot,
            "spectrum": SpectrumPlot,
            "waveform": WaveformPlot,
            "spectrogram": SpectrogramPlot,
        }

        plotter = plotters[plot_type]()

        # 生成图表 - 每个 plotter 返回 Figure 对象
        if plot_type == "waterfall":
            fig = plotter.plot(arr, fs=data.fs, **kwargs)  # type: ignore
        elif plot_type == "spectrum":
            # 使用第一个通道
            channel_data = arr[:, 0] if arr.ndim > 1 else arr
            from scipy.fft import rfft, rfftfreq

            spectrum = np.abs(rfft(channel_data))
            freqs = rfftfreq(len(channel_data), 1 / data.fs)
            fig = plotter.plot(freqs, 20 * np.log10(spectrum + 1e-10), **kwargs)
        elif plot_type == "waveform":
            channel_data = arr[:, 0] if arr.ndim > 1 else arr
            fig = plotter.plot(channel_data, fs=data.fs, **kwargs)  # type: ignore
        elif plot_type == "spectrogram":
            channel_data = arr[:, 0] if arr.ndim > 1 else arr
            fig = plotter.plot(channel_data, fs=data.fs, **kwargs)  # type: ignore

        # 保存图片
        if output_path is None:
            output_path = f"/tmp/das_plot_{uuid4().hex[:8]}.png"

        fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return {
            "plot_type": plot_type,
            "output_path": output_path,
            "data_id": data_id,
            "shape_plotted": list(arr.shape),
        }

    # ========== 会话管理工具 ==========

    def list_session_objects(self) -> Dict[str, Any]:
        """列出当前会话中的所有数据对象。

        Returns:
            所有对象的信息
        """
        return {
            "objects": self.session.list_objects(),
            "count": len(self.session._objects),
        }

    def delete_object(self, data_id: str) -> Dict[str, Any]:
        """删除指定的数据对象。

        Args:
            data_id: 要删除的对象 ID

        Returns:
            操作结果
        """
        success = self.session.delete(data_id)
        return {
            "deleted": success,
            "id": data_id,
        }

    # ========== 数据清洗专用工具 ==========

    def assess_data_quality(self, data_id: str) -> Dict[str, Any]:
        """评估数据质量，识别噪声特征。

        Args:
            data_id: 数据 ID

        Returns:
            包含质量指标的字典:
            - bad_channels_indices: 坏道索引列表
            - has_50hz_noise: 是否存在 50Hz 工频干扰
            - has_trend: 是否存在趋势项
            - snr_estimate_db: 估计信噪比 (dB)
        """
        from .features.cleaning import assess_data_quality

        return assess_data_quality(self.session, data_id)

    def apply_cleaning_recipe(self, data_id: str, recipe_name: str) -> Dict[str, Any]:
        """应用预定义的清洗套餐。

        Args:
            data_id: 数据 ID
            recipe_name: 套餐名称 (standard_denoise, seismic_enhance)

        Returns:
            处理结果信息
        """
        from .features.cleaning import apply_cleaning_recipe

        return apply_cleaning_recipe(self.session, data_id, recipe_name)
