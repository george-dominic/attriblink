"""Core functionality for attriblink."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd

from . import methods
from .exceptions import EffectsSumMismatchError, InvalidMethodError
from .result import AttributionResult
from .utils.math import Unit
from .validators import (
    normalize_effects,
    normalize_returns,
    validate_alignment,
    validate_effects,
    validate_effects_sum,
    validate_not_missing,
    validate_returns,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


AVAILABLE_METHODS = {"carino"}


__all__ = ["link", "AttributionResult"]


def link(
    effects: pd.DataFrame,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    method: str = "carino",
    unit: Unit | str = "decimal",
    check_effects_sum: bool = True,
    strict: bool = False,
) -> AttributionResult:
    """Link attribution effects across multiple periods.

    This function applies a linking method to convert period-by-period
    attribution effects into linked effects that sum exactly to the
    cumulative excess return.

    Args:
        effects: DataFrame where each column is an attribution effect
            (e.g., allocation, selection, interaction). Each row represents
            a time period. The index must align with the return series.
        portfolio_returns: Series of portfolio returns for each period.
        benchmark_returns: Series of benchmark returns for each period.
        method: Linking method to use. Currently only "carino" is supported.
        unit: Unit of the input effects and returns. Can be:
            - "decimal": values as decimals (e.g., 0.02 for 2%)
            - "bps": values in basis points (e.g., 200 for 2%)
            - "percent": values in percent (e.g., 2 for 2%)
            Default is "decimal" for backward compatibility.
        check_effects_sum: If True, validates that period-by-period effects
            sum to period-by-period excess returns. Default is True.
        strict: If True and check_effects_sum is True, raises an error when
            effects don't sum to excess. If False, issues a warning but continues.
            Default is False.

    Returns:
        AttributionResult: An object containing:
            - .linked_effects: Series of linked effects
            - .data: DataFrame with all attribution data (period values, totals, linked)
            - .summary(): Print a formatted summary table
            - .k_factor: The Carino k-factor used
            - .date_range: Tuple of (start_date, end_date)
            - .num_periods: Number of periods
            - .effect_columns: List of effect column names
            - Access linked effects via: result['effect_name']

    Raises:
        InvalidMethodError: If an unsupported method is specified.
        AttributionError: If inputs are invalid or misaligned.
        EffectsSumMismatchError: If effects don't sum to excess return and strict=True.

    Example:
        >>> import pandas as pd
        >>> from attriblink import link
        >>> portfolio = pd.Series([0.02, 0.03], index=pd.date_range("2024-01-01", periods=2, freq="ME"))
        >>> benchmark = pd.Series([0.015, 0.02], index=portfolio.index)
        >>> effects = pd.DataFrame({"allocation": [0.005, 0.008], "selection": [0.002, 0.005]}, index=portfolio.index)
        >>> result = link(effects, portfolio, benchmark, method="carino")
        >>> print(result.summary())
        
    Example with BPS input:
        >>> # Effects in basis points (e.g., 50 bps = 0.50%)
        >>> effects_bps = pd.DataFrame({"allocation": [50, 80], "selection": [20, 50]}, index=portfolio.index)
        >>> result = link(effects_bps, portfolio, benchmark, unit="bps")
    """
    # Validate method
    if method not in AVAILABLE_METHODS:
        raise InvalidMethodError(
            f"Unknown method '{method}'. Available methods: {AVAILABLE_METHODS}"
        )

    # Check if index is sorted chronologically
    if not effects.index.is_monotonic_increasing:
        warnings.warn(
            "effects index is not sorted chronologically. "
            "Data will be sorted automatically. "
            "To suppress this warning, sort your data before passing to link().",
            FutureWarning,
            stacklevel=2
        )
        # Sort all inputs by index
        effects = effects.sort_index()
        portfolio_returns = portfolio_returns.sort_index()
        benchmark_returns = benchmark_returns.sort_index()

    # Normalize inputs based on unit
    effects_normalized = normalize_effects(effects, unit)
    portfolio_normalized = normalize_returns(portfolio_returns, unit)
    benchmark_normalized = normalize_returns(benchmark_returns, unit)

    # Get conversion factor for output (convert decimal back to original unit)
    if isinstance(unit, str):
        unit = Unit(unit.lower())
    
    if unit == Unit.BPS:
        output_factor = 10000
    elif unit == Unit.PERCENT:
        output_factor = 100
    else:  # DECIMAL
        output_factor = 1

    # Validate normalized inputs
    validate_effects(effects_normalized)
    validate_returns(portfolio_normalized, "portfolio_returns")
    validate_returns(benchmark_normalized, "benchmark_returns")
    validate_alignment(effects_normalized, portfolio_normalized, benchmark_normalized)
    validate_not_missing(effects_normalized, portfolio_normalized, benchmark_normalized)

    # Validate effects sum to excess return if enabled
    if check_effects_sum:
        validate_effects_sum(
            effects_normalized, portfolio_normalized, benchmark_normalized, strict=strict
        )

    # Apply the linking method
    if method == "carino":
        linked_series, k_factor = methods.carino_link(
            effects_normalized, portfolio_normalized, benchmark_normalized, return_k=True
        )

    # Convert linked effects back to original unit
    linked_series = linked_series * output_factor

    return AttributionResult(
        linked_effects=linked_series,
        k_factor=k_factor,
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        effects=effects,
        unit=unit,
    )
