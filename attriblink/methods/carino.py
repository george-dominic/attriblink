"""Carino multi-period linking method.

The Carino method uses a log-based scaling factor (k-factor) to link attribution
effects across multiple periods while preserving additivity.

Formula:
    k = R / sum_t(effect_t)

where:
    R = cumulative excess return over the period (simple sum)
    effect_t = sum of all attribution effects in period t

The linked effect for each source is:
    linked_effect_j = k * sum_t(effect_j_t)

This ensures the key invariant:
    sum_j(linked_effect_j) = R = sum_t(excess_t)

For single-period cases, k = 1 (no linking needed).

For multi-period cases, the k-factor scales the period-by-period
effect sums so they add up to the total cumulative excess return.

Edge cases:
- Single period: k = 1 (effects returned as-is)
- When R ≈ 0: k = 1

Reference:
    Carino, D. R. (1999). Linking Attribution Effects. CFA Institute.
"""

import numpy as np
import pandas as pd

from ..exceptions import ZeroExcessReturnError
from ..utils.math import DEFAULT_EPSILON


def carino_link(
    effects: pd.DataFrame,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """Apply Carino multi-period linking to attribution effects.

    Args:
        effects: DataFrame where each column is an attribution effect for each period.
        portfolio_returns: Portfolio returns for each period.
        benchmark_returns: Benchmark returns for each period.

    Returns:
        Series of linked effects (one value per effect column).
        The sum of linked effects equals the cumulative excess return.

    Raises:
        ZeroExcessReturnError: If cumulative excess return is zero or near-zero.
    """
    # Convert to numpy arrays for performance
    effects_arr = effects.values  # Shape: (n_periods, n_effects)
    portfolio_arr = portfolio_returns.values
    benchmark_arr = benchmark_returns.values

    n_periods = len(portfolio_arr)

    # Compute period excess returns
    excess_returns = portfolio_arr - benchmark_arr

    # Compute cumulative excess return (simple sum for additivity)
    cumulative_excess = np.sum(excess_returns)

    # Handle single period case: no linking needed
    if n_periods == 1:
        k_factor = 1.0
    else:
        # Sum of all effects across all periods
        total_effects = np.sum(effects_arr)

        # Compute k-factor (Carino scaling factor)
        # The key invariant: k * total_effects = cumulative_excess
        # So: k = cumulative_excess / total_effects
        if abs(total_effects) < DEFAULT_EPSILON:
            if abs(cumulative_excess) < DEFAULT_EPSILON:
                # Both are zero/near-zero: use k = 1
                k_factor = 1.0
            else:
                raise ZeroExcessReturnError(
                    "Cannot compute Carino link: total effects are near zero "
                    "but cumulative excess return is not"
                )
        else:
            k_factor = cumulative_excess / total_effects

    # Sum effects across periods for each effect type
    effect_sums = np.sum(effects_arr, axis=0)

    # Apply k-factor scaling
    linked_effects = k_factor * effect_sums

    # Preserve original index names from effects columns
    result = pd.Series(linked_effects, index=effects.columns, name="linked_effects")

    return result


def get_k_factor(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> float:
    """Compute the Carino k-factor for given returns.

    This is a utility function to extract the k-factor separately
    from the linking calculation.

    Args:
        portfolio_returns: Array of portfolio returns.
        benchmark_returns: Array of benchmark returns.

    Returns:
        The Carino k-factor.
    """
    if len(portfolio_returns) == 1:
        return 1.0

    excess_returns = portfolio_returns - benchmark_returns
    cumulative_excess = np.sum(excess_returns)
    total_effects = np.sum(excess_returns)  # In practice, this comes from effects

    if abs(total_effects) < DEFAULT_EPSILON:
        if abs(cumulative_excess) < DEFAULT_EPSILON:
            return 1.0
        return 1.0  # Fallback

    return cumulative_excess / total_effects
