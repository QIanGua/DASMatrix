# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DASMatrix is a high-performance Python framework for processing and analyzing Distributed Acoustic Sensing (DAS) data. The project is primarily developed in Chinese but designed for scientific computing applications in geophysics, structural health monitoring, and security surveillance.

### Key Architecture Components

- **Lazy Loading & Out-of-Core Processing**: Built on Xarray and Dask for handling large datasets
- **Chainable API**: Modern fluent interface through `DASFrame` for signal processing workflows
- **Plugin System**: Extensible format support through `FormatRegistry` and format plugins
- **Dual Processing Engines**: Both new DASFrame API and legacy DASProcessor for backward compatibility

## Common Commands

### Development Environment Setup
```bash
# Install using uv (recommended)
uv sync

# Alternative: standard pip install
pip install -e .
```

### Testing
```bash
# Run all tests
just test
# Or directly with uv
uv run pytest tests

# Run specific test file
uv run pytest tests/unit/test_dasframe.py

# Performance benchmarks
uv run python tests/performance/benchmark_large_scale.py
```

### Code Quality

The project uses a **role-separation approach** between git hooks and manual commands to avoid duplication:

#### Git Hooks (Automatic)
- **pre-commit**: Auto-fixes code style (`ruff --fix` + `ruff format`)
- **pre-push**: Quick validation checks only (no modifications)

#### Manual Commands (Development)
```bash
# Quick development fixes
just fix-all     # Auto-fix linting + format (fast)

# Comprehensive validation (CI/release)
just check-all   # Full checks with fixes + tests

# Individual tools
just lint        # Check only (no fixes)
just lint-fix    # Check and auto-fix
just format      # Format code
just typecheck   # Type checking with mypy via ty

# Git hook simulation
just quick-check # Same as pre-push hook (validation only)

# Clean artifacts
just clean
```

#### Recommended Workflow
1. **During development**: `just fix-all` to auto-fix issues
2. **Before committing**: Git hooks auto-run (no action needed)
3. **Before release/CI**: `just check-all` for comprehensive validation
4. **For manual validation**: `just quick-check` (same as pre-push)

### Documentation
```bash
# Build documentation
just docs

# Serve docs with hot-reload for development
just docs-serve

# Deploy to GitHub Pages
just docs-deploy
```

### CI/CD Pipeline

The project uses GitHub Actions with optimized workflows:

#### Automated Workflows
- **CI (`ci.yml`)**: Runs on push/PR with parallel jobs:
  - Code quality: lint, format, type check
  - Multi-platform testing: Python 3.9-3.12 on Ubuntu/Windows/macOS
  - Security scanning: safety, bandit
  - Integration testing: examples validation
  - Package building with artifact upload

- **Documentation (`docs.yml`)**: Auto-deploys to GitHub Pages on main branch
- **Release (`release.yml`)**: Automated PyPI publishing on version tags
- **Benchmarks (`benchmark.yml`)**: Weekly performance monitoring

#### Key Features
- **Concurrency control**: Cancels redundant runs automatically
- **Smart caching**: Uses uv dependency caching for faster builds
- **Coverage reporting**: Integrated with Codecov
- **Security scanning**: Continuous vulnerability assessment
- **Multi-platform support**: Ensures compatibility across OS

#### Manual Triggers
```bash
# Simulate CI locally
just check-all

# Test examples before CI
for example in examples/*.py; do uv run python "$example"; done

# Build documentation locally
just docs
```

## Code Architecture

### Core Data Processing Pipeline

1. **Data Acquisition (`DASMatrix.acquisition`)**
   - `DASReader`: Legacy unified reader with `DataType` enum (DAT, H5, SEGY, MINISEED)
   - `FormatRegistry`: New plugin-based format detection and reading system
   - Format plugins in `acquisition/formats/`: Extensible support for new data formats

2. **Core API (`DASMatrix.api`)**
   - `DASFrame`: Primary data container built on Xarray/Dask for lazy evaluation
   - `df.py`: Functional API entry points (`read()`, `from_array()`, `stream()`)
   - Automatic format detection and metadata extraction

3. **Signal Processing (`DASMatrix.processing`)**
   - `DASProcessor`: Legacy immediate-execution processing engine
   - `engine.py`: New computation graph engine for deferred execution
   - `numba_filters.py`: High-performance compiled filters

4. **Visualization (`DASMatrix.visualization`)**
   - `DASVisualizer`: Scientific plotting for waveforms, spectra, spectrograms, waterfalls
   - Supports multiple output formats and customizable styling

5. **Configuration (`DASMatrix.config`)**
   - `SamplingConfig`: Data acquisition parameters (fs, channels, byte_order, filtering)
   - `VisualizationConfig`: Plot styling and output settings

### API Design Patterns

**Recommended Modern API (DASFrame)**:
```python
from DASMatrix.api import df

# Lazy loading with automatic format detection
frame = df.read("data.h5")

# Chainable processing (builds computation graph)
result = (frame
    .detrend(axis="time")
    .bandpass(1, 500)
    .normalize()
)

# Execute computation and return numpy array
data = result.collect()
```

**Legacy API (maintained for backward compatibility)**:
```python
from DASMatrix import DASReader, DASProcessor, SamplingConfig, DataType

config = SamplingConfig(fs=10000, channels=512)
reader = DASReader(config, DataType.H5)
processor = DASProcessor(config)

raw_data = reader.ReadRawData(path)
processed = processor.ProcessDifferential(raw_data)
```

### Data Format Plugin System

New data formats can be added by implementing the `FormatPlugin` interface in `acquisition/formats/`. The `FormatRegistry` automatically discovers and loads plugins, supporting:

- Automatic format detection by file extension and content
- Lazy loading capabilities with Dask
- Metadata scanning without full data loading
- Consistent API across different formats

### Testing Strategy

- **Unit tests**: `tests/unit/` - Individual component testing
- **Performance tests**: `tests/performance/` - Large-scale benchmarking
- **Integration tests**: Via examples in `examples/` directory
- **Visualization verification**: `tests/verify_plot_rms.py` for plot output validation

### Pre-commit Hooks

The project uses pre-commit with:
- Ruff for linting and formatting (replaces flake8, black, isort)
- `just check-all` runs on pre-push to ensure all quality checks pass

### Documentation Generation

- MkDocs with Material theme
- Automatic API documentation via mkdocstrings
- Chinese language support
- Auto-generated reference pages from docstrings

## Development Notes

- **Language**: Primary development in Chinese, but APIs and code comments should be clear for international developers
- **Dependencies**: Heavy use of scientific Python stack (NumPy, SciPy, Matplotlib, Xarray, Dask)
- **Performance**: Critical paths optimized with Numba compilation and vectorized operations
- **Memory Management**: Designed for out-of-core processing of large DAS datasets
- **Extensibility**: Plugin architecture allows easy addition of new data formats and processing algorithms

## Key Files for Understanding

- `DASMatrix/__init__.py` - Main exports and public API
- `DASMatrix/api/dasframe.py` - Core DASFrame implementation
- `DASMatrix/api/df.py` - Functional API entry points
- `DASMatrix/acquisition/formats/__init__.py` - Format plugin system
- `examples/quick_start.py` - Basic usage patterns
- `tests/unit/test_dasframe.py` - Core functionality tests