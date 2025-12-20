"""DASMatrix 实时仪表盘演示

演示如何结合 DASMatrix 处理链和高性能 DASDashboard 进行实时监测与可视化。
"""

import time
import numpy as np
from DASMatrix import from_array
from DASMatrix.visualization import DASWebDashboard as DASDashboard


def simulate_stream_chunk(fs=1000, n_channels=128, chunk_duration=0.2):
    """模拟产生一段实时 DAS 数据"""
    n_samples = int(fs * chunk_duration)
    # 背景噪声
    data = 0.5 * np.random.randn(n_samples, n_channels)
    
    # 注入突发事件 (概率 20%)
    if np.random.random() < 0.2:
        center_ch = np.random.randint(20, n_channels - 20)
        t = np.linspace(0, chunk_duration, n_samples)
        # 50Hz 信号
        signal = 5.0 * np.sin(2 * np.pi * 50 * t) * np.exp(-10 * (t - chunk_duration/2)**2)
        
        for i in range(-5, 6):
            ch = center_ch + i
            if 0 <= ch < n_channels:
                weight = 1.0 - abs(i) / 6.0
                data[:, ch] += signal * weight
                
    return data


def run_dashboard_demo(duration=30.0, lang="cn", focus_channel=64, open_browser=True):
    """运行实时仪表盘演示"""
    print(f"🚀 正在启动 DASMatrix 实时仪表盘 (语言: {lang})...")
    
    fs = 1000
    n_channels = 128
    # Data Generation Loop (10Hz for smooth UI)
    chunk_duration = 0.1  # 100ms per frame = 10 FPS
    
    # 初始化仪表盘
    dashboard = DASDashboard(
        n_channels=n_channels,
        fs=fs,
        buffer_duration=10.0,
        lang=lang,
        focus_channel=focus_channel
    )
    dashboard.show(open_browser=open_browser)
    
    # 关键修复：等待浏览器连接后再开始推流，防止冷启动时数据丢失
    if not dashboard.wait_for_client(timeout=30):
        print("⚠️ 未检测到浏览器连接，推流继续进行...")
    
    processed_duration = 0.0
    start_time = time.time()
    last_print = -1
    
    try:
        # 改为基于处理数据的累计量来驱动循环，这样 10s 演示一定能推完 10s 的数据
        while processed_duration < duration:
            loop_start = time.time()
            
            # 1. 模拟采集数据块
            raw_chunk = simulate_stream_chunk(fs, n_channels, chunk_duration)
            
            # 2. DASMatrix 处理链
            frame = from_array(raw_chunk, fs=fs)
            # 滤波 + 归一化 (模拟实际处理过程)
            processed_frame = frame.bandpass(10, 200).normalize()
            
            # 显式计算以获取处理后的数据块和检测结果
            processed_data = processed_frame.collect()
            events = processed_frame.threshold_detect(sigma=2.0) # 进一步降低阈值确保日志触发
            
            # 3. 更新仪表盘
            # 推送处理后的数据，确保 Max/RMS 数值与波形一致
            dashboard.update(
                chunk=processed_data,
                events=events
            )
            
            processed_duration += chunk_duration
            
            # 4. 打印监测状态
            if int(processed_duration) > last_print:
                print(f"📡 实时监测中... [{processed_duration:.1f}s / {duration}s]  ", end="\r", flush=True)
                last_print = int(processed_duration)
            
            # 控制频率，使其不快于真实时间 (如果处理太快则等待)
            elapsed_wall = time.time() - start_time
            if processed_duration > elapsed_wall:
                time.sleep(processed_duration - elapsed_wall)
                
    except KeyboardInterrupt:
        print("\n⏹ 用户停止监测")
    finally:
        dashboard.close()
        print("✅ 演示结束")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DASMatrix Premium Dashboard Demo")
    parser.add_argument("--duration", type=float, default=30.0, help="Demo duration in seconds")
    parser.add_argument("--lang", type=str, default="cn", choices=["cn", "en"], help="Display language")
    parser.add_argument("--ch", type=int, default=64, help="Focus channel index")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    
    args = parser.parse_args()
    run_dashboard_demo(duration=args.duration, lang=args.lang, focus_channel=args.ch, open_browser=not args.no_browser)
