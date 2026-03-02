"""Utility functions for attriblink."""

from .math import (
    DEFAULT_EPSILON,
    Unit,
    compute_cumulative_excess_return,
    compute_cumulative_return,
    compute_excess_returns,
    is_near_zero,
    normalize_to_decimal,
    safe_expm1,
    safe_log1p,
    validate_decimal_unit,
)

__all__ = [
    "Unit",
    "safe_log1p",
    "safe_expm1",
    "is_near_zero",
    "compute_excess_returns",
    "compute_cumulative_return",
    "compute_cumulative_excess_return",
    "normalize_to_decimal",
    "validate_decimal_unit",
    "DEFAULT_EPSILON",
]
