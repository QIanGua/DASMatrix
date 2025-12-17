# DASFrame

`DASFrame` 是 DASMatrix 的核心数据处理类，提供流畅的链式 API 进行 DAS 信号处理。

## 概述

DASFrame 采用延迟计算（Lazy Evaluation）模式：所有操作不会立即执行，而是构建计算图，直到调用 `collect()` 或可视化方法时才真正计算。

```python
from DASMatrix import df

# 链式处理
result = (
    df(data, fs=1000)
    .detrend()
    .bandpass(1, 100)
    .normalize()
    .collect()  # 触发计算
)
```

---

## API 参考

::: DASMatrix.api.DASFrame
