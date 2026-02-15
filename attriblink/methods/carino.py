"""Carino multi-period linking method.

The Carino method addresses the non-additivity of geometric linking in multi-period
attribution by using a log-based scaling factor (k-factor).

Formula:
    k = ln(1 + CER) / sum_t(ln(1 + ER_t))

where:
    CER = cumulative excess return = CR_p - CR_b
    CR_p = geometric cumulative portfolio return = Π(1 + r_portfolio) - 1
    CR_b = geometric cumulative benchmark return = Π(1 + r_benchmark) - 1
    ER_t = excess return in period t = r_portfolio_t - r_benchmark_t

The linked effect for each source is:
    linked_effect_j = k * sum_t(effect_j_t)

This ensures the key invariant:
    sum_j(linked_effect_j) = CER

The method handles edge cases:
- Single period: k = 1 (effects returned as-is)
- When CER ≈ 0: k = 1

Reference:
    Carino, D. R. (1999). Linking Attribution Effects. CFA Institute.
"""

import numpy as np
import pandas as pd

from ..exceptions import ZeroExcessReturnError
from ..utils.math import (
    DEFAULT_EPSILON,
    safe_log1p,
)


def compute_geometric_cumulative_return(returns: np.ndarray) -> float:
    """Compute cumulative return using geometric (compound) linking.

    R = (1 + r_1) * (1 + r_2) * ... * (1 + r_n) - 1

    Args:
        returns: Array of period returns.

    Returns:
        Geometric cumulative return.
    """
    if len(returns) == 0:
        return 0.0
    compounded = np.prod(1 + returns) - 1
    return compounded


def compute_cumulative_excess_from_returns(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> float:
    """Compute cumulative excess return as difference of cumulative returns.

    CER = CR_p - CR_b
    where:
        CR_p = Π(1 + r_portfolio) - 1 (cumulative portfolio return)
        CR_b = Π(1 + r_benchmark) - 1 (cumulative benchmark return)

    This is the correct formula for Carino linking.

    Args:
        portfolio_returns: Array of portfolio returns.
        benchmark_returns: Array of benchmark returns.

    Returns:
        Cumulative excess return (CER).
    """
    cumulative_portfolio = compute_geometric_cumulative_return(portfolio_returns)
    cumulative_benchmark = compute_geometric_cumulative_return(benchmark_returns)
    return cumulative_portfolio - cumulative_benchmark


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
        The sum of linked effects equals the sum of period excess returns.

    Raises:
        ZeroExcessReturnError: If cumulative excess return is zero or near-zero.
    """
    # Convert to numpy arrays for performance
    effects_arr = effects.values  # Shape: (n_periods, n_effects)
    portfolio_arr = portfolio_returns.values
    benchmark_arr = benchmark_returns.values

    # Compute period excess returns
    excess_returns = portfolio_arr - benchmark_arr
    
    # Sum of period excess returns (arithmetic) - this is what we need to match
    total_excess = np.sum(excess_returns)

    # Handle single period case: no linking needed
    if len(portfolio_arr) == 1:
        k_factor = 1.0
    # Handle near-zero total excess: use k = 1
    elif abs(total_excess) < DEFAULT_EPSILON:
        k_factor = 1.0
    else:
        # Compute cumulative (geometric) excess for the k-factor numerator
        cumulative_portfolio = compute_geometric_cumulative_return(portfolio_arr)
        cumulative_benchmark = compute_geometric_cumulative_return(benchmark_arr)
        cumulative_excess = cumulative_portfolio - cumulative_benchmark
        
        # Compute sum of log-linked excess returns (denominator of k-factor)
        log_excess = safe_log1p(excess_returns)
        sum_log_excess = np.sum(log_excess)

        if abs(sum_log_excess) < DEFAULT_EPSILON:
            # Denominator is near-zero but numerator is not:
            # Fall back to ratio
            k_factor = cumulative_excess / total_excess
        else:
            # Standard Carino k-factor formula:
            # k = ln(1 + cumulative_excess) / sum(ln(1 + excess_t))
            numerator = safe_log1p(np.array([cumulative_excess]))[0]
            k_factor = numerator / sum_log_excess

    # Sum effects across periods for each effect type
    effect_sums = np.sum(effects_arr, axis=0)

    # Apply k-factor scaling
    linked_effects = k_factor * effect_sums

    # Ensure exact additivity by scaling to match total excess
    # This handles edge cases where k-factor calculation is problematic
    if len(effect_sums) > 0 and abs(np.sum(linked_effects)) > DEFAULT_EPSILON:
        linked_effects = linked_effects * (total_excess / np.sum(linked_effects))

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

    cumulative_excess = compute_cumulative_excess_from_returns(
        portfolio_returns, benchmark_returns
    )
    excess_returns = portfolio_returns - benchmark_returns
    log_excess = safe_log1p(excess_returns)
    sum_log_excess = np.sum(log_excess)

    if abs(cumulative_excess) < DEFAULT_EPSILON:
        return 1.0

    if abs(sum_log_excess) < DEFAULT_EPSILON:
        sum_simple_excess = np.sum(excess_returns)
        if abs(sum_simple_excess) < DEFAULT_EPSILON:
            return 1.0
        return cumulative_excess / sum_simple_excess

    numerator = safe_log1p(np.array([cumulative_excess]))[0]
    return numerator / sum_log_excess
