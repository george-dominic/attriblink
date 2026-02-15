"""Input validation for attriblink."""

import numpy as np
import pandas as pd

from .exceptions import (
    InvalidEffectsError,
    InvalidReturnsError,
    AlignmentError,
)


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
