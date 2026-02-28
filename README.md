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

### `link(effects, portfolio_returns, benchmark_returns, method='carino', check_effects_sum=True, strict=False)`

Links attribution effects across multiple periods.

**Parameters:**
- `effects` (pd.DataFrame): DataFrame where each column is an attribution effect (e.g., allocation, selection). Index must align with return series.
- `portfolio_returns` (pd.Series): Portfolio returns for each period.
- `benchmark_returns` (pd.Series): Benchmark returns for each period.
- `method` (str): Linking method to use. Currently only "carino" is supported.
- `check_effects_sum` (bool): If True, validates that period-by-period effects sum to period-by-period excess returns. Default is True.
- `strict` (bool): If True and `check_effects_sum` is True, raises `EffectsSumMismatchError` when effects don't sum to excess. If False, issues a UserWarning but continues. Default is False.

**Returns:**
- `AttributionResult`: An object containing linked effects and attribution data.

**Raises:**
- `AttributionError`: If inputs are invalid or misaligned.
- `EffectsSumMismatchError`: If effects don't sum to excess return and `strict=True`.

**Validation Behavior:**
By default, the function validates that each period's effects sum to that period's excess return (portfolio - benchmark). This helps catch attribution errors early. Use `check_effects_sum=False` to disable this check for legacy data or when using custom scaling.

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
