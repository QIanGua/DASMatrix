"""Agent 会话管理器。

管理数据对象的生命周期，在 Agent 多轮对话中保持状态。
"""

from typing import Any, Dict, Optional
from uuid import uuid4


class AgentSession:
    """Agent 会话管理器。

    在多轮对话中管理数据对象的生命周期。
    每个数据对象都有一个唯一的 ID，Agent 可以通过 ID 引用数据。

    Example:
        >>> session = AgentSession()
        >>> data_id = session.store(das_frame)
        >>> retrieved = session.get(data_id)
    """

    def __init__(self) -> None:
        """初始化会话。"""
        self._objects: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def store(self, obj: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """存储对象并返回唯一 ID。

        Args:
            obj: 要存储的对象 (如 DASFrame)
            metadata: 可选的元数据

        Returns:
            str: 对象的唯一标识符
        """
        obj_id = f"data_{uuid4().hex[:8]}"
        self._objects[obj_id] = obj
        self._metadata[obj_id] = metadata or {}
        return obj_id

    def get(self, obj_id: str) -> Any:
        """通过 ID 获取对象。

        Args:
            obj_id: 对象 ID

        Returns:
            存储的对象

        Raises:
            KeyError: 如果 ID 不存在
        """
        if obj_id not in self._objects:
            raise KeyError(f"Object not found: {obj_id}")
        return self._objects[obj_id]

    def get_metadata(self, obj_id: str) -> Dict[str, Any]:
        """获取对象元数据。"""
        return self._metadata.get(obj_id, {})

    def list_objects(self) -> Dict[str, Dict[str, Any]]:
        """列出所有存储的对象及其元数据。"""
        result = {}
        for obj_id in self._objects:
            obj = self._objects[obj_id]
            result[obj_id] = {
                "type": type(obj).__name__,
                "shape": getattr(obj, "shape", None),
                **self._metadata.get(obj_id, {}),
            }
        return result

    def delete(self, obj_id: str) -> bool:
        """删除对象。

        Args:
            obj_id: 对象 ID

        Returns:
            是否成功删除
        """
        if obj_id in self._objects:
            del self._objects[obj_id]
            self._metadata.pop(obj_id, None)
            return True
        return False

    def clear(self) -> int:
        """清空所有对象。

        Returns:
            删除的对象数量
        """
        count = len(self._objects)
        self._objects.clear()
        self._metadata.clear()
        return count
