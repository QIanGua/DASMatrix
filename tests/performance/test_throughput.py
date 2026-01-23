"""吞吐量基准测试。"""

import pytest


@pytest.mark.benchmark(group="throughput")
class TestProcessingThroughput:
    """信号处理吞吐量测试。"""

    def test_bandpass_memory_throughput(self, benchmark, large_dasframe):
        """测试内存中带通滤波吞吐量。"""

        def run_bandpass():
            # 执行计算
            return large_dasframe.bandpass(10, 100).collect()

        # 运行基准测试
        benchmark(run_bandpass)

    def test_fft_memory_throughput(self, benchmark, large_dasframe):
        """测试内存中 FFT 吞吐量。"""

        def run_fft():
            return large_dasframe.fft().collect()

        benchmark(run_fft)


@pytest.mark.benchmark(group="lazy_throughput")
class TestLazyThroughput:
    """延迟计算吞吐量测试。"""

    def test_lazy_chain_throughput(self, benchmark, lazy_dasframe):
        """测试延迟链式调用及计算吞吐量。"""

        def run_chain():
            # 构建计算图并执行
            return (
                lazy_dasframe.detrend(axis="time")
                .bandpass(10, 100)
                .normalize()
                .collect()
            )

        # 运行基准测试
        benchmark(run_chain)
