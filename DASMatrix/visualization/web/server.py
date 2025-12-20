"""DASMatrix Web Dashboard 后端服务器

使用 FastAPI 和 WebSocket 实现数据实时透传。
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_loop, is_ready
    main_loop = asyncio.get_running_loop()
    is_ready = True
    print("🚀 Web 服务器启动成功，Event Loop 已绑定。")
    yield
    is_ready = False

app = FastAPI(title="DASMatrix Web Dashboard", lifespan=lifespan)

# 允许跨域 (防止某些浏览器安全限制)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 连接池
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._connected_event = asyncio.Event()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        self._connected_event.set()  # 标记已有连接
        print(f"📡 浏览器已连接 WebSocket. 当前活跃数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            if len(self.active_connections) == 0:
                self._connected_event.clear()
            print(
                f"🔌 浏览器已断开 WebSocket. 剩余活跃数: {len(self.active_connections)}"
            )

    async def wait_for_client(self, timeout: float = 30.0) -> bool:
        """等待至少一个客户端连接"""
        if self.active_connections:
            return True
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return

        data = json.dumps(message)
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(data)
            except Exception as e:
                print(f"⚠️ 发送失败: {e}")
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


# 系统配置
class DashboardConfig(BaseModel):
    n_channels: int
    fs: float
    buffer_duration: float
    lang: str = "cn"
    focus_channel: int = 0


config_state: Optional[DashboardConfig] = None


@app.get("/api/config")
async def get_config():
    print(f"⚙️ 浏览器请求配置: {config_state}")
    return config_state


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 监听来自浏览器的消息 (JSON)
            data = await websocket.receive_text()
            try:
                cmd = json.loads(data)
                if cmd.get("type") == "set_focus_channel":
                    new_ch = int(cmd.get("value", 0))
                    if config_state:
                        config_state.focus_channel = new_ch
                        print(f"🎯 浏览器指令: 切换焦点通道至 {new_ch}")
            except Exception as e:
                print(f"⚠️ 无法解析浏览器消息: {e}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"❌ WebSocket 异常: {e}")
        manager.disconnect(websocket)


main_loop: Optional[asyncio.AbstractEventLoop] = None
is_ready = False


# Lifespan handled above



# 静态文件挂载 (确保 /ws 不被覆盖)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# 供 Python SDK 调用接口
async def start_server(host: str, port: int):
    import uvicorn

    # 改为 info 级别以便观察连接
    config = uvicorn.Config(
        app, host=host, port=port, log_level="info", access_log=True
    )
    server = uvicorn.Server(config)
    await server.serve()


def run_in_background(host: str, port: int):
    import threading

    thread = threading.Thread(
        target=lambda: asyncio.run(start_server(host, port)), daemon=True
    )
    thread.start()
    return thread
