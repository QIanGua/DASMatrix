"""ComputationGraph 单元测试"""

import numpy as np
import pytest

from DASMatrix.core.computation_graph import (
    ComputationGraph,
    NodeDomain,
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
        """测试源节点计算 (SourceNode 保留了简单 compute，但通常由 Engine 调用)"""
        data = np.array([1, 2, 3])
        node = SourceNode(data)
        result = node.compute()

        assert np.array_equal(result, data)


class TestOperationNode:
    """测试 OperationNode"""

    def test_create_operation_node(self):
        """测试创建操作节点"""
        source = SourceNode(np.array([1, 2, 3]))
        # OperationNode 现在期望 string operation，而不是 callable
        node = OperationNode(operation="detrend", inputs=[source], name="op_detrend")

        assert node.name == "op_detrend"
        assert node.operation == "detrend"
        assert node.inputs == [source]
        assert node.domain == NodeDomain.GENERIC

    def test_raises_on_direct_compute(self):
        """测试直接调用 compute 应该抛出异常 (现在由 Engine 处理)"""
        source = SourceNode(np.array([1, 2, 3]))
        op_node = OperationNode(operation="double", inputs=[source])

        with pytest.raises(NotImplementedError):
            op_node.compute()


class TestComputationGraph:
    """测试 ComputationGraph"""

    def test_create_leaf(self):
        """测试创建叶子图"""
        data = np.array([1, 2, 3])
        graph = ComputationGraph.leaf(data)

        assert graph.root is not None
        assert isinstance(graph.root, SourceNode)

    def test_add_node_immutability(self):
        """测试 add_node 返回新实例"""
        data = np.array([1, 2, 3])
        graph = ComputationGraph.leaf(data)

        op_node = OperationNode(operation="test", inputs=[graph.root])
        new_graph = graph.add_node(op_node)

        assert new_graph is not graph
        assert new_graph.root is op_node
        # 原图不受影响
        assert isinstance(graph.root, SourceNode)

    def test_graph_structure_traversal(self):
        """测试图结构遍历"""
        data = np.array([1, 2, 3])
        source = SourceNode(data, name="source")
        op1 = OperationNode(operation="op1", inputs=[source], name="op1")
        op2 = OperationNode(operation="op2", inputs=[op1], name="op2")

        graph = ComputationGraph(op2)

        nodes = graph.nodes
        assert len(nodes) == 3
        # nodes 应该是拓扑序 (source -> op1 -> op2)
        assert nodes[0] is source
        assert nodes[1] is op1
        assert nodes[2] is op2
