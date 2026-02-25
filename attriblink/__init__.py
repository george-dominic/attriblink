"""attriblink - Multi-period attribution linking for portfolio returns."""

from .core import link
from .exceptions import (
    AlignmentError,
    AttributionError,
    InvalidEffectsError,
    InvalidMethodError,
    InvalidReturnsError,
    ZeroExcessReturnError,
)
from .result import AttributionResult

__version__ = "0.2.0"

__all__ = [
    "link",
    "AttributionResult",
    "AttributionError",
    "AlignmentError",
    "InvalidEffectsError",
    "InvalidMethodError",
    "InvalidReturnsError",
    "ZeroExcessReturnError",
]
