import asyncio
import json
from typing import cast

from fastapi import WebSocket

from DASMatrix.visualization.web.server import ConnectionManager


class _FakeWebSocket:
    def __init__(self, fail_send: bool = False):
        self.fail_send = fail_send
        self.accepted = False
        self.sent: list[str] = []

    async def accept(self):
        self.accepted = True

    async def send_text(self, text: str):
        if self.fail_send:
            raise RuntimeError("send failed")
        self.sent.append(text)


def test_connection_manager_connect_and_broadcast():
    manager = ConnectionManager()
    ws = _FakeWebSocket()

    asyncio.run(manager.connect(cast(WebSocket, ws)))
    assert ws.accepted
    assert ws in manager.active_connections

    asyncio.run(manager.broadcast({"type": "ping"}))
    assert ws.sent
    assert json.loads(ws.sent[0])["type"] == "ping"

    manager.disconnect(cast(WebSocket, ws))
    assert ws not in manager.active_connections


def test_connection_manager_drops_failed_connection():
    manager = ConnectionManager()
    ws_ok = _FakeWebSocket()
    ws_bad = _FakeWebSocket(fail_send=True)

    asyncio.run(manager.connect(cast(WebSocket, ws_ok)))
    asyncio.run(manager.connect(cast(WebSocket, ws_bad)))
    asyncio.run(manager.broadcast({"type": "update"}))

    assert ws_ok in manager.active_connections
    assert ws_bad not in manager.active_connections
