"""Polars 后端实现，负责元数据查询与过滤。"""

from typing import List

import polars as pl


class PolarsBackend:
    """Polars 元数据后端。"""

    def __init__(self, meta_df: pl.DataFrame):
        self.meta = meta_df.lazy()

    def filter(self, expr: str) -> "PolarsBackend":
        """执行 SQL-like 过滤字符串"""
        # 这里应该有一个简单的 parser 将 string 转为 polars expr
        # 简化演示：直接假设传入的是 polars expression 对象
        # 实际 DSL 实现需要 dsl.py 解析器的支持
        raise NotImplementedError("String expression filtering not implemented yet")

    def select_channels(self, condition: pl.Expr) -> List[int]:
        """根据条件筛选通道索引"""
        # 触发计算，获取符合条件的 channel index
        res = self.meta.filter(condition).select("channel_index").collect()
        return res["channel_index"].to_list()
