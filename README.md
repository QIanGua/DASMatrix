# DASMatrix

<div align="center">
  <img src="./DASMatrix-Logo.jpg" alt="DASMatrix Logo" width="200"/>
  <h3>Distributed Acoustic Sensing Data Processing and Analysis Framework</h3>
  
  [![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![CI Status](https://github.com/yourusername/DASMatrix/workflows/CI/badge.svg)](https://github.com/yourusername/DASMatrix/actions)
  [![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://yourusername.github.io/DASMatrix)
  [![中文文档](https://img.shields.io/badge/docs-中文-red.svg)](README_cn.md)
</div>

---

## 📖 Project Overview

DASMatrix is a high-performance Python framework specifically designed for Distributed Acoustic Sensing (DAS) data processing and analysis. This framework provides a comprehensive toolkit for reading, processing, analyzing, and visualizing DAS data, suitable for research and applications in geophysics, structural health monitoring, and security surveillance.

### ✨ Core Features

- **🚀 High-Efficiency Data Reading**: Support for 12+ data formats (DAT, HDF5, PRODML, Silixa, Febus, Terra15, APSensing, ZARR, NetCDF, SEG-Y, MiniSEED, TDMS) with **Lazy Loading**
- **⚡ HPC Engine**: Built on **Xarray** and **Dask** for TB-level Out-of-Core processing with **Numba** JIT-optimized kernels and operator fusion
- **🔗 Fluent Chainable API**: Intuitive signal processing workflows through `DASFrame`
- **📊 Professional Signal Processing**: Comprehensive tools including spectral analysis, filtering, integration
- **📈 Scientific-Grade Visualization**: Multiple plot types including time-domain waveforms, spectra, spectrograms, waterfalls
- **📏 Unit System**: First-class physical unit support via **Pint** integration
- **🎲 Built-in Examples**: Easy generation of synthetic data (sine waves, chirps, events) for testing
- **🎯 High-Performance Design**: Vectorized and parallel computing optimizations for critical algorithms

## 🚀 Quick Start

### Installation

#### Option 1: Using uv (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/DASMatrix.git
cd DASMatrix

# Install with uv (automatically creates virtual environment)
uv sync
```

#### Option 2: Using pip
```bash
# Clone the repository
git clone https://github.com/yourusername/DASMatrix.git
cd DASMatrix

# Install with pip
pip install -e .
```

### Basic Usage

#### 1. Modern API with DASFrame (Recommended)
```python
from DASMatrix import df

# Create DASFrame with lazy loading
frame = df.read("data.h5")

# Build processing pipeline
processed = (
    frame
    .detrend(axis="time")   # Remove trend
    .bandpass(1, 500)       # Bandpass filter
    .normalize()            # Normalize
)

# Advanced Analysis (Modern STFT API)
stft_frame = frame.stft(nperseg=1024, noverlap=512)

# Execute computation (Parallelized via Dask)
result = processed.collect()

# Scientific Visualization (with Auto-Decimation Protection)
processed.plot_heatmap(title="HPC Waterfall", max_samples=2000)
```

#### 2. Legacy API
```python
from DASMatrix.acquisition import DASReader, DataType
from DASMatrix.config import SamplingConfig

# Configure sampling parameters
config = SamplingConfig(
    fs=10000,      # Sampling frequency 10kHz
    channels=512,  # 512 channels
    wn=5.0,        # 5Hz high-pass filter
    byte_order="big"
)

# Read data
reader = DASReader(config, DataType.DAT)
raw_data = reader.ReadRawData("path/to/data.dat")
```

#### 3. Visualization Example
```python
from DASMatrix.visualization import DASVisualizer
import matplotlib.pyplot as plt

# Create visualizer
visualizer = DASVisualizer(
    output_path="./output",
    sampling_frequency=config.fs
)

# Time-domain waveform
visualizer.WaveformPlot(
    data[:, 100],          # Channel 100 data
    time_range=(0, 10),    # Show 0-10 seconds
    amplitude_range=(-0.5, 0.5),
    title="Waveform Plot",
    file_name="waveform_ch100"
)

# Spectrum plot
visualizer.SpectrumPlot(
    data[:, 100],
    title="Spectrum Plot",
    db_range=(-80, 0),
    file_name="spectrum_ch100"
)

# Spectrogram
visualizer.SpectrogramPlot(
    data[:, 100],
    freq_range=(0, 500),
    time_range=(0, 10),
    cmap="inferno",
    file_name="spectrogram_ch100"
)

# Waterfall plot (time-channel)
visualizer.WaterfallPlot(
    data,
    title="Waterfall Plot",
    colorbar_label="Amplitude",
    value_range=(-0.5, 0.5),
    file_name="waterfall"
)

plt.show()
```

## 📚 Documentation

- **[Full Documentation](https://yourusername.github.io/DASMatrix)**: Complete API reference and tutorials
- **[Examples](examples/)**: Practical usage examples
- **[API Reference](https://yourusername.github.io/DASMatrix/api/)**: Detailed API documentation
- **[中文文档](README_cn.md)**: Chinese documentation

## 🏗️ Project Structure

```text
DASMatrix/
├── acquisition/           # Data acquisition module
│   ├── formats/          # Format plugins
│   └── das_reader.py     # DAS data reader class
├── api/                   # Core API
│   ├── dasframe.py       # DASFrame (Xarray/Dask Backend)
│   └── df.py            # Functional API entry points
├── config/                # Configuration module
│   ├── sampling_config.py # Sampling configuration
│   └── visualization_config.py  # Visualization configuration
├── processing/            # Data processing module
│   ├── das_processor.py  # DAS data processor class
│   ├── numba_filters.py  # Numba-optimized filters
│   └── engine.py         # Computation graph engine
├── visualization/         # Visualization module
│   └── das_visualizer.py # DAS visualization class
├── units.py               # Unit system (Pint-based)
├── examples.py            # Example data generation
└── utils/                 # Utility functions
    └── time.py           # Time conversion tools
```


## 🚀 High Performance Computing (HPC)

DASMatrix is engineered for massive DAS datasets:
- **Zero-Copy Loading**: Utilizing `np.memmap` for binary formats to index TBs of data in milliseconds.
*   **Kernel Fusion**: Multiple operations (e.g., `demean -> filter -> abs`) are fused into a single machine-code loop via Numba, minimizing memory traffic.
*   **Lazy Computation Graph**: Every operation returns a lazy `DASFrame`. Real computation only happens when you explicitly `collect()` or `plot()`.
*   **Auto-Decimation**: Interactive visualization of huge datasets is protected by automatic downsampling to keep your UI responsive.

## 🔧 Development

### Development Setup
```bash
# Install development dependencies
uv sync --dev

# Run tests
just test

# Run tests
just test

# Run performance benchmarks
just benchmark

# Code quality checks
just check-all

# Quick fixes
just fix-all
```

### Code Quality Tools
- **Ruff**: Linting and formatting
- **MyPy**: Type checking
- **Pre-commit hooks**: Automatic code quality checks
- **GitHub Actions**: CI/CD pipeline

## 🤝 Contributing

We welcome contributions, issues, and suggestions! Please participate in project development through GitHub Issues and Pull Requests.

### Contributing Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/DASMatrix&type=Date)](https://star-history.com/#yourusername/DASMatrix&Date)

---

**[🇨🇳 中文文档](README_cn.md)** | **[🇺🇸 English](README.md)**