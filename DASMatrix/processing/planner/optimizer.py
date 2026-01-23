"""执行计划优化器。负责将逻辑计算图转换为物理执行计划，实施算子融合。"""

from typing import List, Optional, Set, cast

from ...core.computation_graph import (
    ComputationGraph,
    FusionNode,
    Node,
    NodeDomain,
    OperationNode,
)

# 不可融合的操作 - 需要独立执行
NON_FUSIBLE_OPS: Set[str] = {
    # 需要完整数据的操作
    "slice",
    "fft",
    "stft",
    "hilbert",
    "envelope",
    "normalize",
    # 滤波器操作 (需要 SciPy)
    "bandpass",
    "lowpass",
    "highpass",
    "notch",
    "median_filter",
}

# 可融合的简单逐元素操作
FUSIBLE_OPS: Set[str] = {
    "detrend",
    "demean",
    "abs",
    "scale",
}


class ExecutionPlanner:
    """执行计划生成器与优化器。"""

    def optimize(self, graph: ComputationGraph) -> ComputationGraph:
        """
        对计算图进行优化，主要是进行算子融合。

        策略:
        1. 遍历图节点 (拓扑序)。
        2. 识别连续的可融合 SIGNAL 域节点。
        3. 将它们合并为 FusionNode。

        Args:
            graph: 原始计算图

        Returns:
            ComputationGraph: 优化后的计算图 (包含 FusionNode)
        """
        if not graph.root:
            return graph

        # 1. 获取线性化的节点序列 (这里简化假设是单链结构，复杂图需要更复杂的拓扑遍历)
        # TODO: 支持多分支 DAG
        linear_nodes = self._linearize(graph.root)

        # 2. 执行融合
        optimized_nodes = self._fuse_signal_nodes(linear_nodes)

        # 3.以此构建新图，返回新的 root
        # 注意：这里简化处理，假设 optimized_nodes 的最后一个是 root
        if not optimized_nodes:
            return ComputationGraph(None)

        # 重建引用关系 (链式)
        # 这一步比较 tricky，因为我们改变了图结构。
        # 简化的做法：自底向上重建。

        # 这里的实现是一个简化的 Pass：
        # 我们假设主要是单链结构（Time-Series Pipe），这覆盖了 DSL 90% 的场景。

        current_node = optimized_nodes[0]  # Source
        for i in range(1, len(optimized_nodes)):
            next_node = optimized_nodes[i]
            # 重新连接 input
            next_node.inputs = [current_node]
            current_node = next_node

        return ComputationGraph(current_node)

    def _linearize(self, root: Node) -> List[Node]:
        """将单链图线性化为列表 (Source -> Op1 -> Op2 ... -> Root)"""
        nodes = []
        curr: Optional[Node] = root
        while curr:
            nodes.append(curr)
            if curr.inputs:
                curr = curr.inputs[0]  # 仅跟随第一个输入，处理单链
            else:
                curr = None
        return list(reversed(nodes))

    def _is_fusible(self, node: Node) -> bool:
        """判断节点是否可以融合。"""
        if not isinstance(node, OperationNode):
            return False
        if node.domain != NodeDomain.SIGNAL:
            return False
        # 只有在 FUSIBLE_OPS 中的操作才能融合
        return node.operation in FUSIBLE_OPS

    def _fuse_signal_nodes(self, nodes: List[Node]) -> List[Node]:
        """融合连续的可融合 Signal 节点。"""
        fused_list: List[Node] = []
        pending_fusion: List[OperationNode] = []

        for node in nodes:
            if self._is_fusible(node):
                # 这是一个可以融合的信号操作
                pending_fusion.append(cast(OperationNode, node))
            else:
                # 遇到不可融合节点，先结算之前的融合
                if pending_fusion:
                    if len(pending_fusion) > 1:
                        # 只有多个可融合操作时才创建融合节点
                        fused_node = FusionNode(pending_fusion, name=f"FusedOp_{len(pending_fusion)}")
                        fused_list.append(fused_node)
                    else:
                        # 单个可融合操作直接添加
                        fused_list.append(pending_fusion[0])
                    pending_fusion = []

                # 添加当前节点
                fused_list.append(node)

        # 处理末尾的 pending
        if pending_fusion:
            if len(pending_fusion) > 1:
                fused_node = FusionNode(pending_fusion, name=f"FusedOp_{len(pending_fusion)}")
                fused_list.append(fused_node)
            else:
                fused_list.append(pending_fusion[0])

        return fused_list
