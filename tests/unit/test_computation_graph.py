"""ComputationGraph 单元测试"""

import numpy as np
import pytest

from DASMatrix.core.computation_graph import (
    ComputationGraph,
    Node,
    OperationNode,
    SourceNode,
)


class TestSourceNode:
    """测试 SourceNode"""

    def test_create_source_node(self):
        """测试创建源节点"""
        data = np.array([1, 2, 3])
        node = SourceNode(data, name="test_source")

        assert node.name == "test_source"
        assert np.array_equal(node.data, data)

    def test_compute_source_node(self):
        """测试源节点计算"""
        data = np.array([1, 2, 3])
        node = SourceNode(data)
        result = node.compute()

        assert np.array_equal(result, data)
        assert node.computed is True

    def test_reset_source_node(self):
        """测试重置源节点"""
        data = np.array([1, 2, 3])
        node = SourceNode(data)
        node.compute()
        node.reset()

        assert node.computed is False
        assert node.result is None


class TestOperationNode:
    """测试 OperationNode"""

    def test_create_operation_node(self):
        """测试创建操作节点"""
        op = lambda x: x * 2
        node = OperationNode(op, name="double")

        assert node.name == "double"
        assert node.operation == op

    def test_compute_with_input(self):
        """测试带输入的操作节点计算"""
        source = SourceNode(np.array([1, 2, 3]))
        op_node = OperationNode(lambda x: x * 2)
        op_node.add_input(source)

        result = op_node.compute()

        assert np.array_equal(result, np.array([2, 4, 6]))

    def test_compute_with_args(self):
        """测试带参数的操作节点"""
        source = SourceNode(np.array([1, 2, 3]))
        op_node = OperationNode(lambda x, factor: x * factor, args=(3,))
        op_node.add_input(source)

        result = op_node.compute()

        assert np.array_equal(result, np.array([3, 6, 9]))

    def test_compute_with_kwargs(self):
        """测试带关键字参数的操作节点"""
        source = SourceNode(np.array([1, 2, 3]))
        op_node = OperationNode(lambda x, factor=1: x * factor, kwargs={"factor": 5})
        op_node.add_input(source)

        result = op_node.compute()

        assert np.array_equal(result, np.array([5, 10, 15]))


class TestComputationGraph:
    """测试 ComputationGraph"""

    def test_create_leaf(self):
        """测试创建叶子图"""
        data = np.array([1, 2, 3])
        graph = ComputationGraph.leaf(data)

        assert graph.root is not None
        assert isinstance(graph.root, SourceNode)

    def test_add_operation(self):
        """测试添加操作"""
        data = np.array([1, 2, 3])
        graph = ComputationGraph.leaf(data)
        new_graph = graph.add(lambda x: x * 2)

        assert new_graph is not graph  # 返回新图
        assert isinstance(new_graph.root, OperationNode)

    def test_compute_simple_graph(self):
        """测试简单图计算"""
        data = np.array([1, 2, 3])
        graph = ComputationGraph.leaf(data)
        new_graph = graph.add(lambda x: x * 2)

        result = new_graph.compute()

        assert np.array_equal(result, np.array([2, 4, 6]))

    def test_compute_chained_operations(self):
        """测试链式操作计算"""
        data = np.array([1, 2, 3])
        graph = (
            ComputationGraph.leaf(data)
            .add(lambda x: x + 1)
            .add(lambda x: x * 2)
            .add(lambda x: x - 1)
        )

        result = graph.compute()

        # (1+1)*2-1=3, (2+1)*2-1=5, (3+1)*2-1=7
        assert np.array_equal(result, np.array([3, 5, 7]))

    def test_reset_graph(self):
        """测试重置图"""
        data = np.array([1, 2, 3])
        graph = ComputationGraph.leaf(data).add(lambda x: x * 2)

        graph.compute()
        graph.reset()

        # 所有节点应该被重置
        for node in graph.nodes:
            assert node.computed is False
            assert node.result is None

    def test_lazy_evaluation(self):
        """测试惰性求值"""
        call_count = [0]

        def tracked_operation(x):
            call_count[0] += 1
            return x * 2

        data = np.array([1, 2, 3])
        graph = ComputationGraph.leaf(data).add(tracked_operation)

        # 创建图时不应该执行操作
        assert call_count[0] == 0

        # 调用 compute 时才执行
        graph.compute()
        assert call_count[0] == 1

        # 再次调用 compute 应该使用缓存
        graph.compute()
        assert call_count[0] == 1

    def test_empty_graph_raises(self):
        """测试空图抛出异常"""
        graph = ComputationGraph()

        with pytest.raises(ValueError, match="计算图没有根节点"):
            graph.compute()


class TestComputationGraphWithNumpy:
    """测试 ComputationGraph 与 NumPy 操作"""

    def test_numpy_operations(self):
        """测试 NumPy 操作"""
        data = np.random.randn(100, 10)
        graph = (
            ComputationGraph.leaf(data)
            .add(lambda x: np.abs(x))
            .add(lambda x: np.sqrt(x))
        )

        result = graph.compute()

        expected = np.sqrt(np.abs(data))
        assert np.allclose(result, expected)

    def test_numpy_aggregation(self):
        """测试 NumPy 聚合操作"""
        data = np.random.randn(100, 10)
        graph = ComputationGraph.leaf(data).add(lambda x: np.mean(x, axis=0))

        result = graph.compute()

        assert result.shape == (10,)
        assert np.allclose(result, np.mean(data, axis=0))
