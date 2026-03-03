"""Input validation for attriblink."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .exceptions import (
    AlignmentError,
    EffectsSumMismatchError,
    InvalidEffectsError,
    InvalidReturnsError,
)
from .utils.math import Unit, validate_decimal_unit


__all__ = [
    "validate_effects",
    "validate_returns",
    "validate_alignment",
    "validate_not_missing",
    "validate_effects_sum",
    "normalize_effects",
    "normalize_returns",
]


# Tolerance for floating-point comparisons
EPSILON = 1e-10


def validate_effects(effects: pd.DataFrame) -> None:
    """Validate effects DataFrame.

    Args:
        effects: DataFrame of attribution effects.

    Raises:
        InvalidEffectsError: If effects is not a DataFrame or contains invalid data.
    """
    if not isinstance(effects, pd.DataFrame):
        raise InvalidEffectsError(
            f"effects must be a pandas DataFrame, got {type(effects).__name__}"
        )

    if effects.empty:
        raise InvalidEffectsError("effects DataFrame cannot be empty")

    if effects.shape[1] == 0:
        raise InvalidEffectsError("effects DataFrame must have at least one column")

    # Check for non-numeric data
    # Use dtype kind check for pandas compatibility (including string dtypes)
    if effects.dtypes.apply(lambda x: x.kind not in ('i', 'u', 'f', 'c')).any():
        raise InvalidEffectsError("effects must contain only numeric values")

    # Check for infinite values
    if np.isinf(effects.values).any():
        raise InvalidEffectsError("effects cannot contain infinite values")

    # Check for all-NaN columns
    if effects.isna().all().any():
        raise InvalidEffectsError("effects cannot contain all-NaN columns")


def validate_returns(returns: pd.Series, name: str = "returns") -> None:
    """Validate return series.

    Args:
        returns: Series of returns.
        name: Name for error messages.

    Raises:
        InvalidReturnsError: If returns is not a Series or contains invalid data.
    """
    if not isinstance(returns, pd.Series):
        raise InvalidReturnsError(
            f"{name} must be a pandas Series, got {type(returns).__name__}"
        )

    if returns.empty:
        raise InvalidReturnsError(f"{name} cannot be empty")

    # Check for non-numeric data (use pandas API for compatibility)
    if returns.dtype.kind not in ('i', 'u', 'f', 'c'):  # integer, unsigned, float, complex
        raise InvalidReturnsError(f"{name} must contain numeric values")

    # Check for infinite values
    if np.isinf(returns.values).any():
        raise InvalidReturnsError(f"{name} cannot contain infinite values")


def validate_alignment(
    effects: pd.DataFrame,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> None:
    """Validate that all inputs are properly aligned.

    Args:
        effects: DataFrame of attribution effects.
        portfolio_returns: Portfolio return series.
        benchmark_returns: Benchmark return series.

    Raises:
        AlignmentError: If inputs are not properly aligned.
    """
    # Check index alignment
    if not effects.index.equals(portfolio_returns.index):
        raise AlignmentError(
            "effects index must match portfolio_returns index"
        )

    if not effects.index.equals(benchmark_returns.index):
        raise AlignmentError(
            "effects index must match benchmark_returns index"
        )

    # Check for duplicate indices
    if effects.index.has_duplicates:
        raise AlignmentError("effects index contains duplicate dates")

    if portfolio_returns.index.has_duplicates:
        raise AlignmentError("portfolio_returns index contains duplicate dates")

    if benchmark_returns.index.has_duplicates:
        raise AlignmentError("benchmark_returns index contains duplicate dates")


def validate_not_missing(
    effects: pd.DataFrame,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> None:
    """Validate that inputs don't contain NaN values.

    Args:
        effects: DataFrame of attribution effects.
        portfolio_returns: Portfolio return series.
        benchmark_returns: Benchmark return series.

    Raises:
        InvalidEffectsError: If effects contains NaN.
        InvalidReturnsError: If returns contain NaN.
    """
    if effects.isna().any().any():
        raise InvalidEffectsError("effects cannot contain NaN values")

    if portfolio_returns.isna().any():
        raise InvalidReturnsError("portfolio_returns cannot contain NaN values")

    if benchmark_returns.isna().any():
        raise InvalidReturnsError("benchmark_returns cannot contain NaN values")


# Tolerance for floating-point comparisons in effects sum validation
EFFECTS_SUM_EPSILON = 1e-6


def validate_effects_sum(
    effects: pd.DataFrame,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    strict: bool = False,
) -> None:
    """Validate that period-by-period effects sum to period-by-period excess returns.

    This checks the invariant: for each period t,
        sum_j(effect_j_t) == portfolio_returns_t - benchmark_returns_t

    Args:
        effects: DataFrame of attribution effects.
        portfolio_returns: Portfolio return series.
        benchmark_returns: Benchmark return series.
        strict: If True, raise EffectsSumMismatchError on mismatch.
                If False, issue a UserWarning.

    Raises:
        EffectsSumMismatchError: If effects don't sum to excess and strict=True.
    """
    # Calculate period-by-period excess return
    excess_returns = portfolio_returns - benchmark_returns

    # Calculate sum of effects for each period
    effects_sum_per_period = effects.sum(axis=1)

    # Check difference
    difference = (effects_sum_per_period - excess_returns).abs()

    # Use relative tolerance for comparison
    max_diff = difference.max()
    max_excess = excess_returns.abs().max()

    # For very small excess returns, use absolute tolerance
    if max_excess < EFFECTS_SUM_EPSILON:
        is_within_tolerance = max_diff < EFFECTS_SUM_EPSILON
    else:
        relative_diff = max_diff / max_excess
        is_within_tolerance = relative_diff < EFFECTS_SUM_EPSILON

    if not is_within_tolerance:
        total_effects = effects_sum_per_period.sum()
        total_excess = excess_returns.sum()
        diff_total = total_excess - total_effects

        error_msg = (
            f"Effects do not sum to excess return. "
            f"Sum of period effects: {total_effects:.10f}, "
            f"Sum of excess returns: {total_excess:.10f}, "
            f"Difference: {diff_total:.10f}"
        )

        if strict:
            raise EffectsSumMismatchError(error_msg)
        else:
            warnings.warn(error_msg, UserWarning)


def normalize_effects(
    effects: pd.DataFrame,
    unit: Unit | str = Unit.DECIMAL,
) -> pd.DataFrame:
    """Normalize effects DataFrame to decimal format based on unit.
    
    Args:
        effects: DataFrame of attribution effects.
        unit: Unit of the effects. Can be a Unit enum or string.
            - DECIMAL: no change
            - BPS: divide by 10000
            - PERCENT: divide by 100
    
    Returns:
        DataFrame with effects normalized to decimal format.
    """
    if isinstance(unit, str):
        try:
            unit = Unit(unit.lower())
        except ValueError:
            raise InvalidEffectsError(
                f"Invalid unit '{unit}'. Must be one of: "
                f"{', '.join(u.value for u in Unit)}"
            )
    
    if unit == Unit.DECIMAL:
        # Validate decimal values for potential unit mistakes
        for col in effects.columns:
            for idx, val in effects[col].items():
                validate_decimal_unit(val, unit)
        return effects.copy()
    
    # Validate bps/percent values for potential unit mistakes
    for col in effects.columns:
        for idx, val in effects[col].items():
            validate_decimal_unit(val, unit)
    
    # Normalize non-decimal units
    if unit == Unit.BPS:
        divisor = 10000
    elif unit == Unit.PERCENT:
        divisor = 100
    else:
        raise InvalidEffectsError(f"Unknown unit: {unit}")
    
    return effects / divisor


def normalize_returns(
    returns: pd.Series,
    unit: Unit | str = Unit.DECIMAL,
) -> pd.Series:
    """Normalize return series to decimal format based on unit.
    
    Args:
        returns: Series of returns.
        unit: Unit of the returns. Can be a Unit enum or string.
            - DECIMAL: no change
            - BPS: divide by 10000
            - PERCENT: divide by 100
    
    Returns:
        Series with returns normalized to decimal format.
    """
    if isinstance(unit, str):
        try:
            unit = Unit(unit.lower())
        except ValueError:
            raise InvalidReturnsError(
                f"Invalid unit '{unit}'. Must be one of: "
                f"{', '.join(u.value for u in Unit)}"
            )
    
    if unit == Unit.DECIMAL:
        # Validate decimal values for potential unit mistakes
        for idx, val in returns.items():
            validate_decimal_unit(val, unit)
        return returns.copy()
    
    # Validate bps/percent values for potential unit mistakes
    for idx, val in returns.items():
        validate_decimal_unit(val, unit)
    
    # Normalize non-decimal units
    if unit == Unit.BPS:
        divisor = 10000
    elif unit == Unit.PERCENT:
        divisor = 100
    else:
        raise InvalidReturnsError(f"Unknown unit: {unit}")
    
    return returns / divisor
