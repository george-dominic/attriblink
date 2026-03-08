"""attriblink - Multi-period attribution linking for portfolio returns."""

from __future__ import annotations

from .batch import link_batch
from .core import link
from .exceptions import (
    AlignmentError,
    AttributionError,
    EffectsSumMismatchError,
    InvalidEffectsError,
    InvalidMethodError,
    InvalidReturnsError,
    ZeroExcessReturnError,
)
from .result import AttributionResult
from .utils.math import Unit

__all__ = [
    "link",
    "link_batch",
    "AttributionResult",
    "AttributionError",
    "AlignmentError",
    "EffectsSumMismatchError",
    "InvalidEffectsError",
    "InvalidMethodError",
    "InvalidReturnsError",
    "ZeroExcessReturnError",
    "Unit",
]
