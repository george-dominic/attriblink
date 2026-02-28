"""Core functionality for attriblink."""

import warnings

import pandas as pd

from . import methods
from .exceptions import EffectsSumMismatchError, InvalidMethodError
from .result import AttributionResult
from .validators import (
    validate_alignment,
    validate_effects,
    validate_effects_sum,
    validate_not_missing,
    validate_returns,
)


AVAILABLE_METHODS = {"carino"}


def link(
    effects: pd.DataFrame,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    method: str = "carino",
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
    """
    # Validate method
    if method not in AVAILABLE_METHODS:
        raise InvalidMethodError(
            f"Unknown method '{method}'. Available methods: {AVAILABLE_METHODS}"
        )

    # Validate inputs
    validate_effects(effects)
    validate_returns(portfolio_returns, "portfolio_returns")
    validate_returns(benchmark_returns, "benchmark_returns")
    validate_alignment(effects, portfolio_returns, benchmark_returns)
    validate_not_missing(effects, portfolio_returns, benchmark_returns)

    # Validate effects sum to excess return if enabled
    if check_effects_sum:
        validate_effects_sum(effects, portfolio_returns, benchmark_returns, strict=strict)

    # Apply the linking method
    if method == "carino":
        linked_series, k_factor = methods.carino_link(
            effects, portfolio_returns, benchmark_returns, return_k=True
        )

    return AttributionResult(
        linked_effects=linked_series,
        k_factor=k_factor,
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        effects=effects,
    )
