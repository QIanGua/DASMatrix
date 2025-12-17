"""计算图实现模块。

本模块提供延迟计算图的核心数据结构，支持：
- 节点类型：SourceNode（数据源）、OperationNode（操作）、FusionNode（融合节点）
- 计算域标记：用于引擎调度决策
- 依赖跟踪：支持计算图遍历和优化

计算图是 DASFrame 延迟计算模式的基础，所有信号处理操作首先构建计算图节点，
然后由执行引擎统一优化和执行。
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class NodeDomain(Enum):
    """节点计算域枚举。

    用于标识节点应该由哪个后端引擎执行。

    Attributes:
        GENERIC: 通用域，可由任意后端执行
        METADATA: 元数据域，由 Polars 后端处理
        SIGNAL: 信号域，由 Numba/NumPy 后端处理
    """

    GENERIC = "generic"
    METADATA = "metadata"
    SIGNAL = "signal"


class Node(ABC):
    """计算图节点抽象基类。

    所有节点类型的基类，定义了节点的基本属性和接口。

    Attributes:
        name: 节点名称，用于调试和日志
        domain: 节点计算域
        inputs: 输入节点列表
        result: 缓存的计算结果
        computed: 是否已计算的标志
    """

    def __init__(self, name: str, domain: NodeDomain = NodeDomain.GENERIC) -> None:
        """初始化节点。

        Args:
            name: 节点名称
            domain: 节点计算域，默认为 GENERIC
        """
        self.name = name
        self.domain = domain
        self.inputs: List["Node"] = []
        self.result: Any = None
        self.computed = False

    def add_input(self, node: "Node") -> "Node":
        """添加输入节点。

        Args:
            node: 要添加的输入节点

        Returns:
            Node: 返回自身以支持链式调用
        """
        self.inputs.append(node)
        return self

    @abstractmethod
    def compute(self, backend: Any = None) -> Any:
        """执行节点计算。

        Args:
            backend: 后端引擎实例

        Returns:
            Any: 计算结果
        """
        pass

    def reset(self) -> None:
        """重置节点状态，清除缓存结果。"""
        self.result = None
        self.computed = False


class SourceNode(Node):
    """数据源节点，代表计算图的输入数据。

    SourceNode 是计算图的叶子节点，不依赖其他节点，
    直接持有原始数据。

    Attributes:
        data: 原始输入数据
    """

    def __init__(
        self, data: Any, name: str = "source", domain: NodeDomain = NodeDomain.GENERIC
    ) -> None:
        """初始化数据源节点。

        Args:
            data: 原始输入数据，通常为 NumPy 数组
            name: 节点名称，默认为 "source"
            domain: 节点计算域
        """
        super().__init__(name, domain)
        self.data = data

    def compute(self, backend: Any = None) -> Any:
        """返回原始数据。

        Returns:
            Any: 原始输入数据
        """
        return self.data


class OperationNode(Node):
    """操作节点，代表一个信号处理操作。

    OperationNode 封装了单个操作的全部信息，包括操作名称、参数等。

    Attributes:
        operation: 操作名称，如 'detrend', 'bandpass' 等
        args: 位置参数元组
        kwargs: 关键字参数字典
    """

    def __init__(
        self,
        operation: str,
        inputs: List[Node],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        domain: NodeDomain = NodeDomain.GENERIC,
        name: str = "operation",
    ) -> None:
        """初始化操作节点。

        Args:
            operation: 操作名称
            inputs: 输入节点列表
            args: 位置参数
            kwargs: 关键字参数
            domain: 节点计算域
            name: 节点名称
        """
        super().__init__(name, domain)
        self.operation = operation
        self.inputs = inputs
        self.args = args
        self.kwargs = kwargs or {}

    def compute(self, backend: Any = None) -> Any:
        """执行操作计算。

        注意：在混合引擎中，此方法不应直接调用，
        应由 ExecutionPlanner 生成执行计划后由 Backend 执行。

        Raises:
            NotImplementedError: 始终抛出，提示使用正确的执行方式
        """
        raise NotImplementedError(
            "OperationNode should be executed by a Backend/Planner"
        )


class FusionNode(Node):
    """融合节点，包含多个可合并的连续操作。

    FusionNode 是优化器的产物，将多个连续的 Signal 域操作融合为
    单个节点，由 Numba JIT 编译为高效内核。

    Attributes:
        fused_nodes: 被融合的操作节点列表
    """

    def __init__(self, nodes: List[OperationNode], name: str = "fused_kernel") -> None:
        """初始化融合节点。

        Args:
            nodes: 要融合的操作节点列表
            name: 节点名称，默认为 "fused_kernel"
        """
        super().__init__(name, NodeDomain.SIGNAL)
        self.fused_nodes = nodes
        if nodes:
            self.inputs = nodes[0].inputs

    def compute(self, backend: Any = None) -> Any:
        """执行融合节点计算。

        Raises:
            NotImplementedError: 融合节点必须由 NumbaBackend 编译执行
        """
        raise NotImplementedError(
            "FusionNode must be compiled and executed by NumbaBackend"
        )


class ComputationGraph:
    """计算图容器，管理节点之间的依赖关系。

    ComputationGraph 是不可变的，每次添加节点都返回新的图实例。

    Attributes:
        root: 计算图的根节点（输出节点）
    """

    def __init__(self, root: Optional[Node] = None) -> None:
        """初始化计算图。

        Args:
            root: 根节点，默认为 None 表示空图
        """
        self.root = root

    @classmethod
    def leaf(cls, data: Any) -> "ComputationGraph":
        """创建只包含一个数据源节点的计算图。

        Args:
            data: 原始输入数据

        Returns:
            ComputationGraph: 新的计算图实例
        """
        source = SourceNode(data)
        return cls(source)

    def add_node(self, node: Node) -> "ComputationGraph":
        """添加节点并返回新的图实例。

        采用不可变风格，返回一个指向新根节点的新图。

        Args:
            node: 要添加的节点

        Returns:
            ComputationGraph: 新的计算图实例
        """
        return ComputationGraph(node)

    @property
    def nodes(self) -> List[Node]:
        """获取计算图中的所有节点（拓扑序）。

        Returns:
            List[Node]: 按拓扑顺序排列的节点列表
        """
        if not self.root:
            return []

        visited: set[Node] = set()
        stack = [self.root]
        result: List[Node] = []
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                result.append(node)
                stack.extend(node.inputs)
        return list(reversed(result))
