"""Math utilities for attriblink."""

import numpy as np


# Default epsilon for near-zero comparisons
DEFAULT_EPSILON = 1e-10


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
