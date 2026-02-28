# attriblink

Multi-period attribution linking for portfolio returns.

## Overview

Attribution linking is a technique used in investment performance analysis to decompose portfolio returns across multiple periods while preserving additivity. This package provides implementations of linking methods, starting with the Carino method.

## Installation

```bash
pip install attriblink
```

## Usage

```python
import pandas as pd
import numpy as np
from attriblink import link

# Create sample data
portfolio_returns = pd.Series([0.02, 0.03, 0.015], index=pd.date_range("2024-01-01", periods=3, freq="ME"))
benchmark_returns = pd.Series([0.015, 0.02, 0.01], index=pd.date_range("2024-01-01", periods=3, freq="ME"))

# Attribution effects (e.g., allocation, selection, interaction effects)
effects = pd.DataFrame({
    "allocation": [0.005, 0.008, 0.003],
    "selection": [0.002, 0.005, 0.004],
    "interaction": [0.001, 0.002, 0.001]
}, index=portfolio_returns.index)

# Link effects using Carino method
linked_effects = link(effects, portfolio_returns, benchmark_returns, method="carino")
print(linked_effects)
print(f"Sum of linked effects: {linked_effects.sum():.6f}")
print(f"Total excess return: {(portfolio_returns - benchmark_returns).sum():.6f}")
```

## API

### `link(effects, portfolio_returns, benchmark_returns, method='carino')`

Links attribution effects across multiple periods.

**Parameters:**
- `effects` (pd.DataFrame): DataFrame where each column is an attribution effect (e.g., allocation, selection). Index must align with return series.
- `portfolio_returns` (pd.Series): Portfolio returns for each period.
- `benchmark_returns` (pd.Series): Benchmark returns for each period.
- `method` (str): Linking method to use. Currently only "carino" is supported.

**Returns:**
- `pd.Series`: Linked effects for each attribution source. Sum equals total excess return.

**Raises:**
- `AttributionError`: If inputs are invalid or misaligned.

## Development

```bash
# Install dependencies (requires uv)
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Run tests
pytest

# Build package
python -m build
```

## License

MIT License - see LICENSE file for details.
