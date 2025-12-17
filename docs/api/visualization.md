# 可视化

DASMatrix 提供科学期刊级别的数据可视化功能，遵循 Nature/Science 出版风格。

## 特性

- 🎨 **高对比度配色** - 色盲友好的调色板
- 📐 **专业排版** - Arial 字体，标准化刻度
- 📊 **多种图表类型** - 波形图、频谱图、时频图、瀑布图

## 快速使用

```python
from DASMatrix.visualization import DASVisualizer

# 创建可视化器
viz = DASVisualizer(output_path="./output", sampling_frequency=10000)

# 波形图
viz.WaveformPlot(data[:, 0], title="Waveform")

# 频谱图
viz.SpectrumPlot(data[:, 0], title="Spectrum")

# 瀑布图
viz.WaterfallPlot(data, title="Waterfall")
```

---

## API 参考

### PlotBase

::: DASMatrix.visualization.das_visualizer.PlotBase

### SpectrumPlot

::: DASMatrix.visualization.das_visualizer.SpectrumPlot

### WaveformPlot

::: DASMatrix.visualization.das_visualizer.WaveformPlot

### WaterfallPlot

::: DASMatrix.visualization.das_visualizer.WaterfallPlot

### FKPlot

::: DASMatrix.visualization.das_visualizer.FKPlot
