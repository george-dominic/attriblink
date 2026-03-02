"""Math utilities for attriblink."""

from __future__ import annotations

import warnings
from decimal import Decimal
from enum import Enum

import numpy as np


# Default epsilon for near-zero comparisons
DEFAULT_EPSILON = 1e-10


class Unit(Enum):
    """Unit representation for attribution effects and returns.
    
    Attributes:
        DECIMAL: Decimal format (e.g., 0.02 for 2%)
        BPS: Basis points (e.g., 200 for 2%)
        PERCENT: Percentage format (e.g., 2 for 2%)
    """
    DECIMAL = "decimal"
    BPS = "bps"
    PERCENT = "percent"


def normalize_to_decimal(
    value: float | int | np.floating | np.integer,
    unit: Unit | str = Unit.DECIMAL,
) -> Decimal:
    """Normalize a value to decimal format based on the unit.
    
    Args:
        value: The value to normalize.
        unit: The unit of the value. Can be a Unit enum or string.
            - DECIMAL: no change (e.g., 0.02 stays 0.02)
            - BPS: divide by 10000 (e.g., 200 becomes 0.02)
            - PERCENT: divide by 100 (e.g., 2 becomes 0.02)
    
    Returns:
        The value normalized to decimal format as a Decimal.
    
    Raises:
        ValueError: If an invalid unit is provided.
    """
    # Convert string to Unit if needed
    if isinstance(unit, str):
        try:
            unit = Unit(unit.lower())
        except ValueError:
            raise ValueError(
                f"Invalid unit '{unit}'. Must be one of: "
                f"{', '.join(u.value for u in Unit)}"
            )
    
    # Convert to Decimal for precise calculations
    decimal_value = Decimal(str(value))
    
    if unit == Unit.DECIMAL:
        return decimal_value
    elif unit == Unit.BPS:
        return decimal_value / Decimal(10000)
    elif unit == Unit.PERCENT:
        return decimal_value / Decimal(100)
    else:
        raise ValueError(f"Unknown unit: {unit}")


def validate_decimal_unit(
    value: float | int | np.floating | np.integer,
    unit: Unit | str = Unit.DECIMAL,
) -> None:
    """Warn if a value > 1 is passed with unit='decimal'.
    
    This helps catch potential unit mistakes where the user
    passed a value in bps or percent but specified 'decimal'.
    
    Args:
        value: The value to check.
        unit: The unit of the value.
    """
    # Only validate for decimal unit
    if isinstance(unit, str):
        try:
            unit = Unit(unit.lower())
        except ValueError:
            return  # Skip validation for invalid unit
    
    if unit != Unit.DECIMAL:
        return
    
    # Check if value > 1 (likely a mistake)
    if abs(value) > 1:
        warnings.warn(
            f"Value {value} appears to be > 1 with unit='decimal'. "
            f"Did you mean unit='bps' (divide by 10000) or "
            f"unit='percent' (divide by 100)?",
            UserWarning,
            stacklevel=3
        )


def safe_log1p(x: np.ndarray, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Compute log(1 + x) with safeguards for small/negative values.

    Uses numpy.log1p for numerical stability with small x.
    Clips negative values that would cause issues.

    Args:
        x: Input array.
        epsilon: Small value for clipping.

    Returns:
        Log of (1 + x), safely computed.
    """
    # Clip to avoid log of negative numbers
    x_clipped = np.clip(x, -1 + epsilon, None)
    return np.log1p(x_clipped)


def safe_expm1(x: np.ndarray) -> np.ndarray:
    """Compute exp(x) - 1 with numerical stability for small x.

    Args:
        x: Input array.

    Returns:
        exp(x) - 1.
    """
    return np.expm1(x)


def is_near_zero(x: float, epsilon: float = DEFAULT_EPSILON) -> bool:
    """Check if a value is near zero within epsilon tolerance.

    Args:
        x: Value to check.
        epsilon: Tolerance threshold.

    Returns:
        True if abs(x) < epsilon.
    """
    return abs(x) < epsilon


def compute_excess_returns(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> np.ndarray:
    """Compute excess returns.

    Args:
        portfolio_returns: Array of portfolio returns.
        benchmark_returns: Array of benchmark returns.

    Returns:
        Array of excess returns.
    """
    return portfolio_returns - benchmark_returns


def compute_cumulative_return(returns: np.ndarray) -> float:
    """Compute cumulative return from period returns using simple addition.

    For multi-period linking, we use simple addition of excess returns
    rather than compound returns, as this is the standard approach in
    attribution linking.

    Args:
        returns: Array of period returns.

    Returns:
        Sum of period returns (cumulative return).
    """
    return np.sum(returns)


def compute_cumulative_excess_return(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> float:
    """Compute cumulative excess return.

    Args:
        portfolio_returns: Array of portfolio returns.
        benchmark_returns: Array of benchmark returns.

    Returns:
        Cumulative excess return.
    """
    excess = compute_excess_returns(portfolio_returns, benchmark_returns)
    return compute_cumulative_return(excess)
