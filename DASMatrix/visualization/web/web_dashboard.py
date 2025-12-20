"""DASMatrix Web Dashboard 接口

将处理后的数据推送到 Web 服务器并管理生命周期。
"""

import asyncio
import json
import logging
import time
import webbrowser
import numpy as np
from typing import Dict, List, Optional, Union

from .server import manager, run_in_background, config_state, DashboardConfig

class DASWebDashboard:
    """DAS Web 实时监控看板
    
    通过 Web 浏览器提供高性能、远程可访问的监控界面。
    """
    
    def __init__(
        self,
        n_channels: int,
        fs: float,
        buffer_duration: float = 10.0,
        lang: str = "cn",
        host: str = "127.0.0.1",
        port: int = 8050,
        focus_channel: int = 0
    ):
        global config_state
        self.n_channels = n_channels
        self.fs = fs
        self.host = host
        self.port = port
        self.lang = lang
        
        # 设置服务器端的全局状态
        from . import server
        server.config_state = DashboardConfig(
            n_channels=n_channels,
            fs=fs,
            buffer_duration=buffer_duration,
            lang=lang,
            focus_channel=focus_channel
        )
        
        # 启动后台服务器
        self._server_thread = run_in_background(host, port)
        self.url = f"http://{host}:{port}"
        print(f"🌐 实时 Web 看板已启动: {self.url}")
        
        # 等待服务器就绪 (优化启动速度)
        max_wait = 2.0
        start_wait = time.time()
        while not server.is_ready and time.time() - start_wait < max_wait:
            time.sleep(0.05)
        
        if server.is_ready:
            print("✅ Web 服务已就绪")

    def show(self, open_browser: bool = True):
        """打开看板 (默认尝试自动打开浏览器)"""
        if open_browser:
            webbrowser.open(self.url)
        
    def wait_for_client(self, timeout: float = 30.0):
        """阻塞直到有 Web 客户端连接"""
        from . import server
        if not server.is_ready or server.main_loop is None:
            return False
            
        print("⏳ 等待浏览器连接...")
        future = asyncio.run_coroutine_threadsafe(
            server.manager.wait_for_client(timeout), 
            server.main_loop
        )
        try:
            return future.result(timeout + 5)
        except Exception:
            return False

    def update(self, chunk: np.ndarray, events: Optional[np.ndarray] = None, metadata: Optional[Dict] = None):
        """将数据块推送到 Web 端"""
        from . import server
        
        max_amp = float(np.max(np.abs(chunk)))
        rms_val = float(np.sqrt(np.mean(chunk**2)))
        
        # 提取瀑布图显示的多行数据
        n_samples = chunk.shape[0]
        buffer_dur = server.config_state.buffer_duration if server.config_state else 10.0
        chunk_duration = n_samples / self.fs
        canvas_height = 800
        
        # 计算目标行数
        # target_rows = n_samples / (buffer_duration * fs / 800) 不对，应该是 time_fraction * 800
        # chunk_time_fraction = chunk_duration / buffer_dur
        # target_rows = chunk_time_fraction * canvas_height
        target_rows = max(1, round((chunk_duration / buffer_dur) * canvas_height))
        
        # 限制 step 至少为 1
        step = max(1, n_samples // target_rows)
        waterfall_rows = chunk[::step, :].tolist()

        # 焦点通道详情
        focus_ch = 0
        if server.config_state:
            focus_ch = server.config_state.focus_channel
            
        focus_data = chunk[:, focus_ch].tolist()
        
        # 修复事件计数：统计有多少个通道触发了事件 (any over time)
        # events shape: (samples, channels)
        if events is not None:
            triggered_channels = np.any(events > 0, axis=0) # [channels]
            events_count = int(np.sum(triggered_channels))
        else:
            events_count = 0
        
        # 构建消息
        message = {
            "type": "update",
            "timestamp": time.time(),
            "metrics": {
                "max": max_amp,
                "rms": rms_val
            },
            "waterfall": waterfall_rows,
            "focus_detail": focus_data,
            "events_count": events_count
        }
        
        # 异步广播消息
        if server.main_loop is not None:
            asyncio.run_coroutine_threadsafe(server.manager.broadcast(message), server.main_loop)
        else:
            if int(time.time()) % 5 == 0:
                print("⚠️ 探测到 Web 服务器尚未完全启动，正在重试...", end="\r")
        
    def close(self):
        """关闭服务 (通常后台进程会自动随主程序退出)"""
        pass
