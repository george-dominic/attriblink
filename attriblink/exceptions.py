"""Custom exceptions for attriblink."""


class AttributionError(Exception):
    """Base exception for attribution errors."""
    pass


class InvalidEffectsError(AttributionError):
    """Raised when effects DataFrame is invalid."""
    pass


class InvalidReturnsError(AttributionError):
    """Raised when return series is invalid."""
    pass


class AlignmentError(AttributionError):
    """Raised when inputs are not properly aligned."""
    pass


class InvalidMethodError(AttributionError):
    """Raised when an invalid linking method is specified."""
    pass


class ZeroExcessReturnError(AttributionError):
    """Raised when excess return is zero or near-zero and cannot be linked."""
    pass


class EffectsSumMismatchError(AttributionError):
    """Raised when effects don't sum to excess return per period."""
    pass
