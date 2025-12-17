"""计算图实现，支持延迟计算与依赖跟踪，适配混合引擎。"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class NodeDomain(Enum):
    """节点计算域定义"""

    GENERIC = "generic"
    METADATA = "metadata"  # Polars 域
    SIGNAL = "signal"  # Numba/CuPy 域


class Node(ABC):
    """计算图节点抽象基类。"""

    def __init__(self, name: str, domain: NodeDomain = NodeDomain.GENERIC):
        self.name = name
        self.domain = domain
        self.inputs: List["Node"] = []
        self.result = None
        self.computed = False

    def add_input(self, node: "Node") -> "Node":
        self.inputs.append(node)
        return self

    @abstractmethod
    def compute(self, backend: Any = None) -> Any:
        pass

    def reset(self) -> None:
        self.result = None
        self.computed = False


class SourceNode(Node):
    """数据源节点。"""

    def __init__(
        self, data: Any, name: str = "source", domain: NodeDomain = NodeDomain.GENERIC
    ):
        super().__init__(name, domain)
        self.data = data

    def compute(self, backend: Any = None) -> Any:
        # Source 节点通常直接返回数据，但在某些后端（如Polars）可能返回 LazyFrame
        return self.data


class OperationNode(Node):
    """通用操作节点。"""

    def __init__(
        self,
        operation: str,  # 操作名称，如 'detrend'
        inputs: List[Node],
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        domain: NodeDomain = NodeDomain.GENERIC,
        name: str = "operation",
    ):
        super().__init__(name, domain)
        self.operation = operation
        self.inputs = inputs
        self.args = args
        self.kwargs = kwargs or {}

    def compute(self, backend: Any = None) -> Any:
        """
        在混合引擎中，compute 不应直接被调用。
        应该由 ExecutionPlanner 生成执行计划后，由 Backend 执行。
        这里保留是为了兼容性或调试。
        """
        raise NotImplementedError(
            "OperationNode should be executed by a Backend/Planner"
        )


class FusionNode(Node):
    """融合节点，包含多个连续的 Signal 操作。"""

    def __init__(self, nodes: List[OperationNode], name: str = "fused_kernel"):
        super().__init__(name, NodeDomain.SIGNAL)
        self.fused_nodes = nodes
        # 输入是第一个节点的输入
        if nodes:
            self.inputs = nodes[0].inputs

    def compute(self, backend: Any = None) -> Any:
        raise NotImplementedError(
            "FusionNode must be compiled and executed by NumbaBackend"
        )


class ComputationGraph:
    """计算图容器。"""

    def __init__(self, root: Optional[Node] = None):
        self.root = root

    @classmethod
    def leaf(cls, data: Any) -> "ComputationGraph":
        """创建只包含一个数据源节点的计算图。"""
        source = SourceNode(data)
        return cls(source)

    def add_node(self, node: Node) -> "ComputationGraph":
        """返回一个新的图实例，指向新的根节点（不可变风格）"""
        return ComputationGraph(node)

    @property
    def nodes(self) -> List[Node]:
        """简单的 BFS 获取所有节点（用于调试或简单遍历）"""
        if not self.root:
            return []

        # 简单实现，仅用于演示。实际可能需要拓扑排序。
        visited = set()
        stack = [self.root]
        result = []
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                result.append(node)
                stack.extend(node.inputs)
        return list(reversed(result))  # 拓扑序近似
