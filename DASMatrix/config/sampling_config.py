from dataclasses import dataclass


@dataclass
class SamplingConfig:
    """采样配置"""

    fs: float = 30000.0  # 采样频率 (Hz)
    channels: int = 100  # 通道数
    byte_order: str = "little"  # DAT 文件字节序 ('big' 或 'little')
    wn: float = 1.0  # 高通滤波器截止频率 (Hz)
