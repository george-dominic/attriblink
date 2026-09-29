"""Carino multi-period linking method.

The Carino method addresses the non-additivity of geometric linking in multi-period
attribution by using a log-based scaling factor (k-factor).

Formula:
    k_t = (ln(1 + r_p,t) - ln(1 + r_b,t)) / (r_p,t - r_b,t)
    K = (ln(1 + R_p) - ln(1 + R_b)) / (R_p - R_b)
    linked_effect_j = sum_t(effect_j,t * k_t / K)

R_p and R_b are compounded portfolio and benchmark returns. When returns
are equal, the coefficient's limit is 1 / (1 + return). Reconciled period
effects therefore sum to R_p - R_b after linking, within floating-point
precision. No residual rescaling is applied.

Reference:
    Carino, D. R. (1999). Combining Attribution Effects Over Time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import pandas as pd

from ..utils.math import safe_log1p

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_geometric_cumulative_return(returns: NDArray[np.float64]) -> float:
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
    return float(compounded)


def compute_cumulative_excess_from_returns(
    portfolio_returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
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


@overload
def carino_link(
    effects: pd.DataFrame,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    return_k: bool = False,
) -> pd.Series: ...


@overload
def carino_link(
    effects: pd.DataFrame,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    return_k: bool = True,
) -> tuple[pd.Series, float]: ...


def carino_link(
    effects: pd.DataFrame,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    return_k: bool = False,
) -> pd.Series | tuple[pd.Series, float]:
    """Apply Carino multi-period linking to attribution effects.

    For reconciled period effects, the sum equals the cumulative excess return
    (geometric active return), consistent with the Carino definition
    of CER in this module.

    Args:
        effects: DataFrame where each column is an attribution effect for each period.
        portfolio_returns: Portfolio returns for each period.
        benchmark_returns: Benchmark returns for each period.
        return_k: If True, return a tuple (linked_effects, k_factor).

    Returns:
        Series of linked effects (one value per effect column).
        Reconciled period effects sum to the cumulative excess return.
        If return_k=True, returns (linked_effects, k_factor) tuple.

    Raises:
        ValueError: If portfolio or benchmark returns are <= -1.
    """
    # Convert to numpy arrays for performance
    effects_arr = effects.values  # Shape: (n_periods, n_effects)
    portfolio_arr = portfolio_returns.values
    benchmark_arr = benchmark_returns.values

    period_k = _log_return_coefficient(portfolio_arr, benchmark_arr)
    k_factor = get_k_factor(portfolio_arr, benchmark_arr)
    linked_effects = np.sum(effects_arr * (period_k / k_factor)[:, None], axis=0)

    # Preserve original index names from effects columns
    result = pd.Series(linked_effects, index=effects.columns, name="linked_effects")

    if return_k:
        return result, k_factor
    return result


def get_k_factor(
    portfolio_returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
) -> float:
    """Compute the Carino k-factor for given returns.

    This is a utility function to extract the k-factor separately
    from the linking calculation.

    Args:
        portfolio_returns: Array of portfolio returns.
        benchmark_returns: Array of benchmark returns.

    Returns:
        The cumulative log-return coefficient K (not a common effect multiplier).
    """
    # Validate the log domain for every period, including single-period inputs.
    safe_log1p(portfolio_returns)
    safe_log1p(benchmark_returns)
    cumulative_portfolio = compute_geometric_cumulative_return(portfolio_returns)
    cumulative_benchmark = compute_geometric_cumulative_return(benchmark_returns)
    return float(_log_return_coefficient(
        np.array([cumulative_portfolio]), np.array([cumulative_benchmark])
    )[0])


def _log_return_coefficient(
    portfolio_returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute log-return divided differences, including the equal-return limit."""
    safe_log1p(portfolio_returns)
    safe_log1p(benchmark_returns)
    excess = portfolio_returns - benchmark_returns
    # log1p of the relative difference avoids subtracting nearly equal logs.
    log_difference = safe_log1p(excess / (1 + benchmark_returns))
    return np.divide(
        log_difference, excess,
        out=1.0 / (1 + benchmark_returns), where=excess != 0,
    )
