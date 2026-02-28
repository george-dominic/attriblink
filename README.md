# attriblink

[![PyPI Version](https://img.shields.io/pypi/v/attriblink)](https://pypi.org/project/attriblink/)
[![Python Versions](https://img.shields.io/pypi/pyversions/attriblink)](https://pypi.org/project/attriblink/)
[![Tests](https://github.com/george-dominic/attriblink/actions/workflows/test.yml/badge.svg)](https://github.com/george-dominic/attriblink/actions/workflows/test.yml)

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
from attriblink import link

# Quarterly portfolio and benchmark returns
portfolio_returns = pd.Series(
    [0.025, 0.035, -0.012, 0.048],
    index=pd.date_range("2025-01-01", periods=4, freq="ME")
)
benchmark_returns = pd.Series(
    [0.018, 0.028, -0.015, 0.038],
    index=portfolio_returns.index
)

# Attribution effects from Brinson-Fachler
effects = pd.DataFrame({
    "allocation":    [0.005, 0.006, 0.002, 0.008],
    "selection":     [0.003, 0.002, -0.001, 0.004],
    "interaction":   [0.001, 0.001, 0.000, 0.002]
}, index=portfolio_returns.index)

# Link effects using Carino method
result = link(effects, portfolio_returns, benchmark_returns)

# View results
print(result.summary())

# Access individual effects
print(f"Allocation: {result['allocation']:.4%}")
print(f"Selection:  {result['selection']:.4%}")

# k-factor interpretation
print(f"k-factor: {result.k_factor:.4f}")
# k > 1: volatile excess returns, k < 1: consistent excess
```

## Understanding the k-Factor

The k-factor is a smoothing coefficient that scales attribution effects to achieve geometric additivity:

- **k = 1.0**: No adjustment needed (arithmetic = geometric)
- **k > 1**: Volatile excess returns — effects scaled up
- **k < 1**: Consistent excess returns — effects scaled down

The sum of linked effects always equals the cumulative excess return.


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
uv run pytest

```

## License

MIT License - see LICENSE file for details.
