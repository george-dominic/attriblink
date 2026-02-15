"""Utility functions for attriblink."""

from .math import (
    compute_cumulative_excess_return,
    compute_cumulative_return,
    compute_excess_returns,
    is_near_zero,
    safe_expm1,
    safe_log1p,
)

__all__ = [
    "safe_log1p",
    "safe_expm1",
    "is_near_zero",
    "compute_excess_returns",
    "compute_cumulative_return",
    "compute_cumulative_excess_return",
]
