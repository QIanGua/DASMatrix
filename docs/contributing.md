# 贡献指南

欢迎为 DASMatrix 做出贡献！本文档介绍代码规范和贡献流程。

---

## 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/yourusername/DASMatrix.git
cd DASMatrix

# 安装开发依赖
uv sync

# 运行测试
just test

# 运行类型检查
just typecheck

# 格式化代码
just format
```

---

## Docstring 规范

DASMatrix 使用 **Google Style** docstring，统一使用**中文**描述。

### 基本格式

```python
def function_name(arg1: Type1, arg2: Type2 = default) -> ReturnType:
    """简短的一行描述。

    可选的详细描述，解释函数的具体行为、算法原理或注意事项。
    可以是多行。

    Args:
        arg1: 参数1的描述
        arg2: 参数2的描述，默认为 default

    Returns:
        返回值的描述

    Raises:
        ValueError: 何时抛出此异常
        TypeError: 何时抛出此异常

    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        expected_output
    """
```

### 类的 Docstring

```python
class ClassName:
    """类的简短描述。

    类的详细描述（可选），说明用途、设计理念等。

    Attributes:
        attr1: 属性1的描述
        attr2: 属性2的描述

    Example:
        >>> obj = ClassName(param1, param2)
        >>> obj.method()
    """

    def __init__(self, param1: Type1, param2: Type2):
        """初始化类实例。

        Args:
            param1: 参数1的描述
            param2: 参数2的描述
        """
```

### 规范要点

| 项目 | 要求 |
|------|------|
| 语言 | 统一使用中文 |
| 风格 | Google Style |
| 类型注解 | 必须提供 |
| Args | 所有参数必须说明 |
| Returns | 必须说明返回值 |
| Raises | 如有异常，必须说明 |
| Example | 推荐但非必须 |

---

## 代码规范

### 格式化

使用 `ruff` 进行代码格式化和 lint：

```bash
just format  # 格式化
just lint    # 检查问题
```

### 类型检查

使用 `mypy` 进行类型检查：

```bash
just typecheck
```

### 测试

使用 `pytest` 运行测试：

```bash
just test
```

如果测试在 `collecting ...` 阶段卡住（通常是 Matplotlib 首次构建字体缓存），
可以使用可写缓存目录：

```bash
MPLCONFIGDIR=/tmp/mplcache MPLBACKEND=Agg just test
```

如果在 macOS 上 `uv` 因系统代理读取崩溃（`system-configuration` 报错），
可确保系统代理已正确配置，或先设置 Wi‑Fi 代理（示例使用本地代理）：

```bash
sudo networksetup -setwebproxy "Wi-Fi" 127.0.0.1 7890
sudo networksetup -setsecurewebproxy "Wi-Fi" 127.0.0.1 7890
sudo networksetup -setsocksfirewallproxy "Wi-Fi" 127.0.0.1 7890
```

恢复代理设置：

```bash
sudo networksetup -setwebproxystate "Wi-Fi" off
sudo networksetup -setsecurewebproxystate "Wi-Fi" off
sudo networksetup -setsocksfirewallproxystate "Wi-Fi" off
```

---

## 提交流程

1. Fork 仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 编写代码和测试
4. 确保所有检查通过：

   ```bash
   just lint
   just typecheck
   just test
   ```

5. 提交 Pull Request
