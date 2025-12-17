# 数据读取

DASMatrix 支持多种 DAS 数据格式的读取。

## 支持的格式

- **DAT** - 二进制 DAT 文件
- **HDF5** - HDF5 格式文件

## 快速使用

```python
from DASMatrix.acquisition import DASReader, DataType
from DASMatrix.config import SamplingConfig

# 配置
config = SamplingConfig(fs=10000, channels=512)

# DAT 文件读取
reader = DASReader(config, data_type=DataType.DAT)
data = reader.ReadRawData("path/to/file.dat")

# HDF5 文件读取
reader = DASReader(config, data_type=DataType.H5)
data = reader.ReadRawData("path/to/file.h5")
```

---

## API 参考

### DASReader

::: DASMatrix.acquisition.DASReader

### DataType

::: DASMatrix.acquisition.DataType

### SamplingConfig

::: DASMatrix.config.SamplingConfig
