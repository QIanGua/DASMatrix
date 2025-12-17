"""计算图实现，支持延迟计算与依赖跟踪。"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple


class Node(ABC):
    """计算图节点抽象基类。"""

    def __init__(self, name: str):
        """初始化节点。

        Args:
            name: 节点名称
        """
        self.name = name
        self.inputs = []  # 输入节点列表
        self.result = None  # 计算结果缓存
        self.computed = False  # 是否已计算

    def add_input(self, node: "Node") -> "Node":
        """添加输入节点。

        Args:
            node: 输入节点

        Returns:
            Node: 当前节点
        """
        self.inputs.append(node)
        return self

    @abstractmethod
    def compute(self, backend: Optional[str] = None) -> Any:
        """执行节点计算。

        Args:
            backend: 可选的计算后端

        Returns:
            Any: 计算结果
        """
        pass

    def reset(self) -> None:
        """重置节点计算状态。"""
        self.result = None
        self.computed = False


class SourceNode(Node):
    """数据源节点，表示计算图的起点。"""

    def __init__(self, data: Any, name: str = "source"):
        """初始化数据源节点。

        Args:
            data: 源数据
            name: 节点名称
        """
        super().__init__(name)
        self.data = data

    def compute(self, backend: Optional[str] = None) -> Any:
        """返回源数据。"""
        if not self.computed:
            self.result = self.data
            self.computed = True
        return self.result


class OperationNode(Node):
    """操作节点，表示对数据的处理操作。"""

    def __init__(
        self,
        operation: Callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        name: str = "operation",
    ):
        """初始化操作节点。

        Args:
            operation: 操作函数
            args: 操作函数的位置参数
            kwargs: 操作函数的关键字参数
            name: 节点名称
        """
        super().__init__(name)
        self.operation = operation
        self.args = args
        self.kwargs = kwargs or {}

    def compute(self, backend: Optional[str] = None) -> Any:
        """执行操作。"""
        if not self.computed:
            # 先计算所有输入节点
            input_results = [node.compute(backend) for node in self.inputs]

            # 执行操作
            self.result = self.operation(*input_results, *self.args, **self.kwargs)
            self.computed = True

        return self.result


class ComputationGraph:
    """计算图，管理节点之间的依赖关系和执行顺序。"""

    def __init__(self, root: Optional[Node] = None):
        """初始化计算图。

        Args:
            root: 根节点，通常是数据源节点
        """
        self.root = root
        self.nodes = []
        if root:
            self.nodes.append(root)

    @classmethod
    def leaf(cls, data: Any) -> "ComputationGraph":
        """创建只包含一个数据源节点的计算图。

        Args:
            data: 源数据

        Returns:
            ComputationGraph: 新的计算图
        """
        source = SourceNode(data)
        return cls(source)

    def add(self, operation: Callable, *args, **kwargs) -> "ComputationGraph":
        """向计算图添加操作节点。

        Args:
            operation: 操作函数
            *args: 操作函数的位置参数
            **kwargs: 操作函数的关键字参数

        Returns:
            ComputationGraph: 新的计算图
        """
        # 创建新的计算图
        new_graph = ComputationGraph()

        # 创建操作节点
        op_node = OperationNode(operation, args, kwargs)

        # 将当前计算图的根节点作为操作节点的输入
        op_node.add_input(self.root)

        # 设置新计算图的根节点
        new_graph.root = op_node
        new_graph.nodes = self.nodes + [op_node]

        return new_graph

    def compute(self, backend: Optional[str] = None) -> Any:
        """执行计算图。

        Args:
            backend: 可选的计算后端

        Returns:
            Any: 计算结果
        """
        if not self.root:
            raise ValueError("计算图没有根节点")

        return self.root.compute(backend)

    def reset(self) -> None:
        """重置计算图，清除所有节点的计算结果。"""
        for node in self.nodes:
            node.reset()
