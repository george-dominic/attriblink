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

__version__ = "0.1.0"

__all__ = [
    "link",
    "AttributionError",
    "AlignmentError",
    "InvalidEffectsError",
    "InvalidMethodError",
    "InvalidReturnsError",
    "ZeroExcessReturnError",
]
