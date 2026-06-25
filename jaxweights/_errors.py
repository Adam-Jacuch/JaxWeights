class JaxWeightsError(Exception):
    """Base exception class for all jaxweights errors."""
    pass


class ContextError(JaxWeightsError):
    """Raised when calling jaxweights operations outside of a transformed function context."""
    pass


class DuplicateParameterError(JaxWeightsError):
    """Raised when duplicate parameter names (explicit, anonymous, or collisons) are detected in the same scope."""
    pass


class ReservedNameError(JaxWeightsError):
    """Raised when an explicit parameter or scope name violates validation rules."""
    pass


class ShapeMismatchError(JaxWeightsError):
    """Raised when the requested parameter shape does not match the stored parameter shape."""
    pass


class DTypeMismatchError(JaxWeightsError):
    """Raised when the requested parameter dtype does not match the stored parameter dtype."""
    pass


class MissingParameterError(JaxWeightsError):
    """Raised when a parameter is requested during apply but is missing from the stored parameters."""
    pass


class TraceMismatchError(JaxWeightsError):
    """Raised when the trace of parameter requests during apply differs from the init trace/spec."""
    pass


class SpecError(JaxWeightsError):
    """Raised when apply is called with an invalid param representation (such as a plain dict)."""
    pass
