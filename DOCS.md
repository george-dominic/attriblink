# attriblink Documentation

Comprehensive documentation for the attriblink library covering the Carino method, implementation details, and best practices for multi-period performance attribution.

---

## Table of Contents

1. [Mathematical Reference](#mathematical-reference)
2. [Effect Validation Policy](#effect-validation-policy)
3. [K-Factor Interpretation](#k-factor-interpretation)
4. [Limitations: When NOT to Use Carino](#limitations-when-not-to-use-carino)
5. [Examples](#examples)
6. [Changelog](#changelog)

---

## Mathematical Reference

### The Attribution Linking Problem

When performing multi-period performance attribution, a fundamental challenge arises: **arithmetic attribution effects do not naturally compound**. If you simply sum period-by-period allocation, selection, and interaction effects, the total will not equal the cumulative excess return between portfolio and benchmark.

This violates the fundamental **additivity principle**: the sum of all attribution effects must equal the total excess return.

### Carino Method Formula

The Carino method (Carino, 1999) resolves this by introducing a **smoothing factor (k-factor)** that scales all attribution effects proportionally:

#### The k-Factor Formula

$$k = \frac{\ln(1 + CER)}{\sum_{t=1}^{n} \ln(1 + ER_t)}$$

Where:
- $CER$ = Cumulative Excess Return (geometric) = $R_p - R_b$
- $R_p$ = Geometric cumulative portfolio return = $\prod_{t=1}^{n}(1 + r_{p,t}) - 1$
- $R_b$ = Geometric cumulative benchmark return = $\prod_{t=1}^{n}(1 + r_{b,t}) - 1$
- $ER_t$ = Period excess return = $r_{p,t} - r_{b,t}$
- $n$ = Number of periods

#### Linked Effects Formula

$$Effect_{linked} = k \times \sum_{t=1}^{n} Effect_t$$

#### The Key Invariant

$$\sum_{j} Effect_{linked,j} = CER$$

This ensures that the sum of all linked attribution effects **exactly equals** the geometric cumulative excess return.

### Detailed Formula Derivation

**Step 1: Compute geometric cumulative returns**

$$R_p = \prod_{t=1}^{n}(1 + r_{p,t}) - 1$$
$$R_b = \prod_{t=1}^{n}(1 + r_{b,t}) - 1$$

**Step 2: Compute cumulative excess return (CER)**

$$CER = R_p - R_b$$

**Step 3: Compute period-by-period excess returns**

$$ER_t = r_{p,t} - r_{b,t} \quad \text{for } t = 1, 2, \ldots, n$$

**Step 4: Compute log-linked excess returns**

$$\ln(ER_t + 1) = \ln(1 + ER_t)$$

**Step 5: Sum the log-linked excess returns**

$$S = \sum_{t=1}^{n} \ln(1 + ER_t)$$

**Step 6: Compute k-factor**

$$k = \frac{\ln(1 + CER)}{S}$$

**Step 7: Scale each attribution effect**

For each attribution source $j$:
$$Effect_{linked,j} = k \times \sum_{t=1}^{n} Effect_{j,t}$$

### Mathematical Intuition

The Carino method bridges **arithmetic** (period-by-period) and **geometric** (cumulative) return calculations:

- **Arithmetic linking**: Simply sum period effects → fails to match geometric excess return
- **Geometric linking**: Compound period effects → mathematically complex, may produce negative effects
- **Carino smoothing**: Scales arithmetic effects by $k$ → preserves sign and relative magnitude while ensuring additivity

### Period-Level k-Factor (Advanced)

For more granular analysis, a period-level k-factor can be computed:

$$k_t = \frac{\ln(1 + r_{p,t}) - \ln(1 + r_{b,t})}{r_{p,t} - r_{b,t}}$$

This is useful when you want to see how the smoothing factor varies across periods. When portfolio and benchmark returns are equal ($r_{p,t} = r_{b,t}$):

$$k_t = \frac{1}{1 + r_{p,t}}$$

### Reference

- Carino, D. R. (1999). "Combining Attribution Effects Over Time." *The Journal of Performance Measurement*, Summer, pp. 5-14.
- Christopherson, J. A., Carino, D. R., & Ferson, W. E. (2009). *Portfolio Performance Measurement and Benchmarking*. McGraw-Hill, Chapter 19.
- Bacon, C. (2004). *Practical Portfolio Performance Measurement and Attribution*. Wiley, pp. 191-193.

---

## Effect Validation Policy

### Default Behavior

By default, `attriblink.link()` **validates** that period-by-period effects sum to period-by-period excess returns:

```python
result = link(effects, portfolio_returns, benchmark_returns, 
               check_effects_sum=True, strict=False)
```

For each period $t$:
$$\sum_{j} Effect_{j,t} = r_{p,t} - r_{b,t}$$

This validation ensures your attribution model is internally consistent before applying the Carino linking.

### Why Validation Matters

1. **Catch calculation errors early**: If effects don't sum to excess in a single period, the attribution model has a bug
2. **Prevent garbage-in-garbage-out**: Linking bad data produces bad results
3. **Maintain credibility**: Linked effects that don't match actual returns undermine trust in attribution analysis

### Handling Validation Failures

| Parameter | Behavior |
|-----------|----------|
| `check_effects_sum=True, strict=False` | Issues a `UserWarning`, continues with calculation |
| `check_effects_sum=True, strict=True` | Raises `EffectsSumMismatchError`, stops execution |
| `check_effects_sum=False` | Skips validation entirely, proceeds with linking |

### When to Disable Validation

You may need to disable validation in these scenarios:

1. **Legacy data**: Historical attribution where original calculation methodology is unknown
2. **Custom scaling**: When you pre-scale effects using a different methodology
3. **Alternative attribution models**: Some geometric attribution models produce effects that intentionally don't sum to arithmetic excess

### Scaling Behavior

The library handles two types of scaling:

1. **k-factor scaling**: Applied automatically via the Carino formula
2. **Residual scaling**: Applied as a final adjustment to ensure exact additivity

```python
# The library ensures exact additivity:
linked_sum = result.linked_effects.sum()
cumulative_excess = (1 + portfolio_returns).prod() - (1 + benchmark_returns).prod() - 1

assert abs(linked_sum - cumulative_excess) < 1e-10  # Always true
```

---

## K-Factor Interpretation

### What the k-Factor Tells You

The k-factor is a **smoothing coefficient** that adjusts raw attribution effects to achieve geometric additivity. It bridges the gap between arithmetic and geometric return calculation.

### K > 1: Interpretation

**When k > 1** (e.g., k = 1.15):

| Scenario | Meaning |
|----------|---------|
| **Volatile excess returns** | Period-by-period excess returns varied significantly (some positive, some negative) |
| **Compounding asymmetry** | Geometric linking would overweight negative periods; Carino scales up to compensate |
| **Typical range** | 1.0 to ~1.5 for moderate volatility |

**Practical example**: If k = 1.15, the raw sum of attribution effects is scaled up by 15%. This typically occurs when:
- Portfolio and benchmark returns have opposite signs in some periods
- Large return dispersion across periods

### K < 1: Interpretation

**When k < 1** (e.g., k = 0.88):

| Scenario | Meaning |
|----------|---------|
| **Consistent excess returns** | Portfolio consistently outperformed (or underperformed) benchmark |
| **Compounding benefit/harm** | Geometric compounding works in your favor; less scaling needed |
| **Typical range** | ~0.5 to 1.0 for consistently positive/negative excess |

**Practical example**: If k = 0.88, the raw sum is scaled down by 12%. This typically occurs when:
- Excess returns are consistently positive (or consistently negative)
- Geometric compounding naturally aligns with arithmetic sum

### K = 1: Interpretation

**When k = 1**:

| Scenario | Meaning |
|----------|---------|
| **Single period** | No linking needed; arithmetic = geometric |
| **Zero cumulative excess** | CER ≈ 0; no adjustment necessary |
| **Log-linear returns** | Period excess returns follow a specific pattern |

### Interpreting Period-Level k_t

The period-level k-factor reveals **within-period dynamics**:

- **k_t ≈ 1**: Period excess return is small; arithmetic ≈ geometric
- **k_t > 1**: Large positive or negative excess in that period
- **k_t < 1**: Moderate excess with compounding benefits

### Worked Example

```python
import pandas as pd
from attriblink import link

# Quarterly data with volatile excess returns
portfolio = pd.Series([0.05, -0.02, 0.08, 0.03], 
                       index=pd.date_range("2024-01-01", periods=4, freq="QE"))
benchmark = pd.Series([0.03, 0.01, 0.05, 0.02],
                       index=portfolio.index)

effects = pd.DataFrame({
    "allocation": [0.015, -0.025, 0.020, 0.008],
    "selection":  [0.005, -0.005, 0.010, 0.002],
}, index=portfolio.index)

result = link(effects, portfolio, benchmark)
print(f"k-factor: {result.k_factor:.4f}")  # Likely > 1 due to volatility
```

---

## Limitations: When NOT to Use Carino

### 1. Single-Period Attribution

Carino is designed for **multi-period** linking. For single periods:
- k = 1 by definition (no linking needed)
- Use standard arithmetic attribution directly

```python
# Don't use Carino for single period
if len(periods) == 1:
    # Just use raw effects
    linked = raw_effects
```

### 2. Near-Zero Cumulative Excess

When $CER \approx 0$:
- k-factor becomes unstable (division by near-zero)
- The library handles this by setting k = 1.0 automatically
- But interpretation becomes problematic

**Warning sign**: Cumulative excess return < 0.001 (0.1%)

### 3. Highly Asymmetric Return Distributions

Carino assumes relatively smooth return patterns. For extreme scenarios:
- Very large positive returns in some periods
- Very large negative returns in others
- Consider Menchero or GRAP methods as alternatives

### 4. When You Need Geometric Attribution Effects

Carino produces **arithmetic-scaled** effects. If you need:
- Geometric attribution effects (compounded through time)
- Consider: Frongello method, GRAP method

### 5. Currency Attribution

Carino doesn't handle **currency effects** directly:
- Requires separate currency attribution model
- Then link currency effects using Carino

### 6. Benchmark Timing Differences

If your benchmark and portfolio have different rebalancing dates:
- Carino assumes aligned periods
- May produce misleading results for misaligned portfolios

### 7. When Exact Period Effects Must Be Preserved

Carino scales **all periods equally** by the same k-factor:
- If you need to show exact per-period contributions
- Consider: disclosure tables showing both raw and linked effects

### Summary Table

| Use Carino When | Don't Use Carino When |
|-----------------|----------------------|
| Multi-period (≥2 periods) | Single period |
| Moderate return dispersion | Extreme return asymmetry |
| Standard Brinson-Fachler or Brinson-Hood-Beebower | Currency attribution |
| Arithmetic attribution effects | Need geometric effects |
| Aligned rebalancing dates | Misaligned portfolio/benchmark |

---

## Examples

### Example 1: Basic Quarterly Attribution

```python
import pandas as pd
import numpy as np
from attriblink import link

# Quarterly portfolio and benchmark returns (2024)
portfolio_returns = pd.Series(
    [0.025, 0.035, -0.012, 0.048],
    index=pd.date_range("2024-01-01", periods=4, freq="QE")
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

# Link effects
result = link(effects, portfolio_returns, benchmark_returns)

# View results
print(result.summary())
print(f"\nLinked effects: {result.linked_effects.to_dict()}")
print(f"k-factor: {result.k_factor:.4f}")

# Verify additivity
total_excess = (1 + portfolio_returns).prod() - (1 + benchmark_returns).prod() - 1
print(f"\nSum of linked effects: {result.linked_effects.sum():.6f}")
print(f"Cumulative excess return: {total_excess:.6f}")
```

### Example 2: Multiple Attribution Effects with Interaction Term

```python
import pandas as pd
from attriblink import link

# Monthly data
portfolio = pd.Series(
    [0.02, 0.03, 0.015, 0.025, 0.01, 0.035],
    index=pd.date_range("2024-01-01", periods=6, freq="ME")
)
benchmark = pd.Series(
    [0.015, 0.025, 0.01, 0.02, 0.008, 0.028],
    index=portfolio.index
)

# Extended attribution model with multiple effects
effects = pd.DataFrame({
    "country_allocation":    [0.003, 0.004, 0.002, 0.003, 0.001, 0.005],
    "currency_effect":      [0.001, 0.002, 0.001, 0.002, 0.001, 0.001],
    "sector_allocation":    [0.002, 0.001, 0.001, 0.002, 0.001, 0.003],
    "security_selection":   [0.004, 0.003, 0.002, 0.003, 0.002, 0.004],
    "interaction":          [0.001, 0.001, 0.000, 0.001, 0.000, 0.001]
}, index=portfolio.index)

result = link(effects, portfolio, benchmark)

# Access individual linked effects
for effect_name in result.effect_columns:
    print(f"{effect_name}: {result[effect_name]:.4%}")

# Full summary
print(result.summary())
```

### Example 3: Handling Validation Warnings

```python
import pandas as pd
import numpy as np
from attriblink import link, EffectsSumMismatchError
import warnings

portfolio = pd.Series([0.02, 0.03], index=pd.date_range("2024-01-01", periods=2, freq="ME"))
benchmark = pd.Series([0.015, 0.025], index=portfolio.index)

# Effects that DON'T sum to excess return (intentional error)
effects = pd.DataFrame({
    "allocation": [0.005, 0.010],  # Should be 0.005, 0.005
    "selection":  [0.002, 0.003]   # Should be 0.002, 0.002
}, index=portfolio.index)

# Default: warn but continue
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = link(effects, portfolio, benchmark, strict=False)
    if w:
        print(f"Warning: {w[0].message}")

# Strict mode: raise error
try:
    result = link(effects, portfolio, benchmark, strict=True)
except EffectsSumMismatchError as e:
    print(f"Error: {e}")

# Disable validation entirely (use carefully!)
result = link(effects, portfolio, benchmark, check_effects_sum=False)
```

### Example 4: Real-World Institutional Portfolio

```python
import pandas as pd
from attriblink import link

# Realistic scenario: Global equity portfolio, monthly for 3 years
dates = pd.date_range("2021-01-01", "2023-12-31", freq="ME")

# Simulated realistic returns (annualized ~7% excess)
np.random.seed(42)
portfolio_returns = pd.Series(
    0.07/12 + np.random.normal(0, 0.04, 36),
    index=dates
)
benchmark_returns = pd.Series(
    0.07/12 + np.random.normal(0, 0.04, 36),
    index=dates
)

# Attribution effects (simulated Brinson-Fachler)
effects = pd.DataFrame({
    "regional_allocation": np.random.normal(0.002, 0.003, 36),
    "country_allocation":   np.random.normal(0.001, 0.002, 36),
    "sector_allocation":   np.random.normal(0.001, 0.002, 36),
    "security_selection":  np.random.normal(0.003, 0.004, 36),
    "currency_effect":     np.random.normal(0.000, 0.001, 36),
}, index=dates)

# Adjust effects to sum to excess (validation requirement)
excess_returns = portfolio_returns - benchmark_returns
effects = effects.div(effects.sum(axis=1), axis=0).multiply(excess_returns, axis=0)

result = link(effects, portfolio_returns, benchmark_returns)

# Multi-year summary
print("=" * 60)
print("GLOBAL EQUITY PORTFOLIO - 3 YEAR ATTRIBUTION")
print("=" * 60)
print(result.summary())

# Key metrics
print(f"\nKey Insights:")
print(f"  - k-factor: {result.k_factor:.4f}")
print(f"  - Total excess return: {(1+portfolio_returns).prod() - (1+benchmark_returns).prod() - 1:.2%}")
print(f"  - Best contributor: {result.linked_effects.idxmax()} ({result.linked_effects.max():.2%})")
print(f"  - Worst contributor: {result.linked_effects.idxmin()} ({result.linked_effects.min():.2%})")
```

### Example 5: Extracting k-Factor for Analysis

```python
import pandas as pd
import numpy as np
from attriblink.methods.carino import get_k_factor

# Analyze k-factor across different return scenarios

def analyze_k_factor(portfolio, benchmark):
    """Analyze what the k-factor tells us about returns."""
    k = get_k_factor(portfolio.values, benchmark.values)
    
    excess = portfolio - benchmark
    cumulative_excess = (1 + portfolio).prod() - (1 + benchmark).prod() - 1
    
    return {
        'k_factor': k,
        'cumulative_excess': cumulative_excess,
        'mean_period_excess': excess.mean(),
        'std_period_excess': excess.std(),
        'excess_volatility': excess.std() * np.sqrt(12)  # Annualized
    }

# Scenario 1: Consistent outperformance
portfolio1 = pd.Series([0.02, 0.025, 0.03, 0.028])
benchmark1 = pd.Series([0.015, 0.018, 0.02, 0.022])

# Scenario 2: Volatile returns
portfolio2 = pd.Series([0.05, -0.02, 0.08, -0.03])
benchmark2 = pd.Series([0.03, 0.01, 0.05, 0.00])

# Scenario 3: Mixed
portfolio3 = pd.Series([0.02, 0.03, -0.01, 0.04])
benchmark3 = pd.Series([0.015, 0.02, 0.005, 0.025])

for i, (p, b) in enumerate([(portfolio1, benchmark1), 
                            (portfolio2, benchmark2), 
                            (portfolio3, benchmark3)], 1):
    result = analyze_k_factor(p, b)
    print(f"\nScenario {i}:")
    print(f"  k-factor: {result['k_factor']:.4f}")
    print(f"  Cumulative excess: {result['cumulative_excess']:.2%}")
    print(f"  Excess volatility: {result['excess_volatility']:.2%}")
    print(f"  → {'k < 1: Consistent' if result['k_factor'] < 1 else 'k > 1: Volatile'}")
```

### Example 6: Basic link() with Decimal Input

```python
import pandas as pd
from attriblink import link

# Simple monthly data with decimal returns (e.g., 0.02 = 2%)
portfolio_returns = pd.Series(
    [0.02, 0.03, 0.015],
    index=pd.date_range("2024-01-01", periods=3, freq="ME")
)
benchmark_returns = pd.Series(
    [0.015, 0.025, 0.01],
    index=portfolio_returns.index
)

# Attribution effects (decimal format)
effects = pd.DataFrame({
    "allocation": [0.004, 0.003, 0.003],
    "selection":  [0.002, 0.003, 0.003]
}, index=portfolio.index)

# Link effects (default: unit="decimal")
result = link(effects, portfolio_returns, benchmark_returns)

print(result.summary())
print(f"\nLinked effects: {result.linked_effects.to_dict()}")
```

### Example 7: link() with BPS Input

```python
import pandas as pd
from attriblink import link

# Returns in basis points (e.g., 200 = 2%)
portfolio_returns = pd.Series(
    [200, 300, 150],  # 2%, 3%, 1.5%
    index=pd.date_range("2024-01-01", periods=3, freq="ME")
)
benchmark_returns = pd.Series(
    [150, 250, 100],  # 1.5%, 2.5%, 1%
    index=portfolio_returns.index
)

# Attribution effects in basis points
effects = pd.DataFrame({
    "allocation": [40, 30, 35],
    "selection":  [20, 30, 25]
}, index=portfolio_returns.index)

# Link with bps unit - library handles conversion automatically
result = link(effects, portfolio_returns, benchmark_returns, unit="bps")

print(result.summary())
print(f"\nLinked effects (decimal): {result.linked_effects.to_dict()}")
# Linked effects are converted back to input unit (bps)
```

### Example 8: link() with Percent Input

```python
import pandas as pd
from attriblink import link

# Returns in percent (e.g., 2 = 2%)
portfolio_returns = pd.Series(
    [2.0, 3.0, 1.5],
    index=pd.date_range("2024-01-01", periods=3, freq="ME")
)
benchmark_returns = pd.Series(
    [1.5, 2.5, 1.0],
    index=portfolio_returns.index
)

# Attribution effects in percent
effects = pd.DataFrame({
    "allocation": [0.4, 0.3, 0.35],
    "selection":  [0.2, 0.3, 0.25]
}, index=portfolio_returns.index)

# Link with percent unit
result = link(effects, portfolio_returns, benchmark_returns, unit="percent")

print(result.summary())
```

### Example 9: link_batch() with Long-Format DataFrame

```python
import pandas as pd
from attriblink import link_batch

# Long-format DataFrame with multiple funds
data = pd.DataFrame({
    'FUND_ID': ['FUND_A', 'FUND_A', 'FUND_A', 'FUND_B', 'FUND_B', 'FUND_B'],
    'DATE': ['2024-01-31', '2024-02-29', '2024-03-31',
             '2024-01-31', '2024-02-29', '2024-03-31'],
    'allocation': [0.005, 0.008, 0.004, 0.003, 0.006, 0.005],
    'selection': [0.003, 0.005, 0.002, 0.002, 0.004, 0.003],
    'portfolio': [0.02, 0.03, 0.015, 0.018, 0.025, 0.02],
    'benchmark': [0.015, 0.02, 0.012, 0.012, 0.018, 0.014],
})

# Apply linking to each fund group
result = link_batch(
    data,
    group_by='FUND_ID',
    date_col='DATE',
    effects_cols=['allocation', 'selection'],
    portfolio_col='portfolio',
    benchmark_col='benchmark',
)

print(result)
# Output:
#          DATE FUND_ID  portfolio_return  benchmark_return  active_return  allocation  selection
# 0  2024-01-31  FUND_A             0.020             0.015            0.005    0.010298    0.006865
# 1  2024-02-29  FUND_A             0.030             0.020            0.010    0.010298    0.006865
# 2  2024-03-31  FUND_A             0.015             0.012            0.003    0.010298    0.006865
# 3  2024-01-31  FUND_B             0.018             0.012            0.006    0.007078    0.005299
# 4  2024-02-29  FUND_B             0.025             0.018            0.007    0.007078    0.005299
# 5  2024-03-31  FUND_B             0.020             0.014            0.006    0.007078    0.005299
```

### Example 10: link_batch() with BPS and Validation Disabled

```python
import pandas as pd
from attriblink import link_batch

# Long-format data with multiple funds in basis points
data = pd.DataFrame({
    'FUND_ID': ['FUND_A', 'FUND_A', 'FUND_B', 'FUND_B'],
    'DATE': ['2024-01-31', '2024-02-29', '2024-01-31', '2024-02-29'],
    'allocation': [50, 80, 40, 60],
    'selection': [30, 50, 20, 40],
    'portfolio': [200, 300, 180, 250],
    'benchmark': [150, 200, 120, 180],
})

# Apply linking with BPS unit and validation disabled
result = link_batch(
    data,
    group_by='FUND_ID',
    date_col='DATE',
    effects_cols=['allocation', 'selection'],
    portfolio_col='portfolio',
    benchmark_col='benchmark',
    unit='bps',
    check_effects_sum=False,  # Skip validation for demo data
)

print(result)
# Each fund gets its own linked effects computed separately
```

---

## Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

### [0.1.0] - 2024-02-25

#### Added
- Initial release of `attriblink` library
- `link()` function implementing Carino multi-period attribution linking
- Carino k-factor computation and formula documentation
- Effect validation with `check_effects_sum` parameter
- Strict mode via `strict` parameter (`EffectsSumMismatchError`)
- `AttributionResult` class with:
  - `.linked_effects`: Linked attribution effects
  - `.k_factor`: Carino smoothing factor
  - `.summary()`: Formatted summary table
  - `.data`: Full DataFrame with period breakdown
  - Dictionary-like access to effects (`result['allocation']`)
- `compute_geometric_cumulative_return()` utility
- `get_k_factor()` standalone function for k-factor analysis
- Input validation for effects, returns, and alignment
- Edge case handling (single period, near-zero excess)

#### Known Issues
- Limited to Carino method only (no Menchero, GRAP, Frongello)
- No built-in support for currency attribution
- No geometric attribution alternatives

---

## API Reference

### `link()`

```python
def link(
    effects: pd.DataFrame,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    method: str = "carino",
    unit: str = "decimal",
    check_effects_sum: bool = True,
    strict: bool = False,
) -> AttributionResult
```

### `link_batch()`

```python
def link_batch(
    data: pd.DataFrame,
    group_by: str,
    date_col: str,
    effects_cols: list[str],
    portfolio_col: str,
    benchmark_col: str,
    unit: str = "decimal",
    method: str = "carino",
    check_effects_sum: bool = True,
) -> pd.DataFrame
```

### `AttributionResult`

| Property/Method | Description |
|-----------------|-------------|
| `.linked_effects` | Series of linked effects (one per column) |
| `.k_factor` | Carino smoothing factor |
| `.data` | Full DataFrame with all periods and totals |
| `.summary()` | Formatted string summary table |
| `.portfolio_returns` | Original portfolio returns |
| `.benchmark_returns` | Original benchmark returns |
| `.effects` | Original effects DataFrame |
| `.date_range` | Tuple of (start, end) dates |
| `.num_periods` | Number of periods |
| `.effect_columns` | List of effect column names |
| `result[effect_name]` | Access individual linked effect |

---

## Further Reading

- CFA Institute. (2019). "Performance Attribution." *Financial Reporting and Analysis Series*.
- Cariño, D. R. (1999). "Combining Attribution Effects Over Time." *The Journal of Performance Measurement*.
- Menchero, J. G. (2000). "An Optimized Approach to Linking Attribution Effects Over Time." *The Journal of Performance Measurement*.
- Frongello, A. (2002). "A Methodology for Linking Decompensation." *The Journal of Performance Measurement*.

---

*Documentation generated for attriblink v0.1.0*
