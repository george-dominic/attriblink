"""Core functionality for attriblink."""

import pandas as pd

from . import methods
from .exceptions import InvalidMethodError
from .result import AttributionResult
from .validators import (
    validate_alignment,
    validate_effects,
    validate_not_missing,
    validate_returns,
)


AVAILABLE_METHODS = {"carino"}


def link(
    effects: pd.DataFrame,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    method: str = "carino",
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

    Example:
        >>> import pandas as pd
        >>> from attriblink import link
        >>> portfolio = pd.Series([0.02, 0.03], index=pd.date_range("2024-01-01", periods=2, freq="M"))
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
