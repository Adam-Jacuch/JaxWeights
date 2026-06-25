from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp

from jaxweights._context import Context, get_context, use_context, validate_name
from jaxweights._bundle import ParamBundle, ParamSpec, RequestSpec
from jaxweights._errors import (
    ContextError,
    DuplicateParameterError,
    ShapeMismatchError,
    DTypeMismatchError,
    MissingParameterError,
    TraceMismatchError,
    SpecError,
)


def parse_name_and_shape(args: Tuple[Any, ...]) -> Tuple[Optional[str], Tuple[int, ...]]:
    """Parses positional arguments to extract name (optional) and shape.
    
    Accepts:
        - zeros(10) -> (None, (10,))
        - zeros((2, 3)) -> (None, (2, 3))
        - zeros(2, 3) -> (None, (2, 3))
        - zeros("w", 10) -> ("w", (10,))
        - zeros("w", (2, 3)) -> ("w", (2, 3))
        - zeros("w", 2, 3) -> ("w", (2, 3))
    """
    if not args:
        raise ValueError("At least one shape dimension or a parameter name must be provided.")

    if isinstance(args[0], str):
        name = args[0]
        shape_args = args[1:]
        if not shape_args:
            raise ValueError("Shape must be provided alongside explicit name.")
        if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list)):
            shape = tuple(shape_args[0])
        else:
            shape = tuple(shape_args)
    else:
        name = None
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            shape = tuple(args[0])
        else:
            shape = tuple(args)

    for dim in shape:
        # Check type
        if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)):
            raise ValueError(f"Shape dimensions must be integers, got {type(dim).__name__}: {dim}")
        if dim < 0:
            raise ValueError(f"Shape dimensions must be non-negative, got {dim}")

    return name, shape


def param(
    name: Optional[str],
    shape: Tuple[int, ...],
    init_fn: Callable[[Any], jax.Array],
    kind: str,
    dtype: Any = jnp.float32,
    config: Optional[Dict[str, Any]] = None,
    needs_rng: bool = False,
) -> jax.Array:
    """Core function to request a parameter.
    
    Checks the current context. In 'init' mode, creates the parameter.
    In 'apply' mode, retrieves and validates the parameter.
    """
    ctx = get_context()
    
    # 1. Validate explicit name if provided
    if name is not None:
        validate_name(name, is_scope=False)
        
    # 2. Determine current scope prefix
    scope_prefix = "/".join(ctx.scope_stack)
    
    # 3. Compute full name
    if name is None:
        if ctx.auto_names == "error":
            raise ContextError("Anonymous parameters are not allowed when auto_names='error'.")
        elif ctx.auto_names == "warn":
            if ctx.on_warn_anonymous is not None:
                ctx.on_warn_anonymous()
        
        # Generate anonymous name in the current scope
        counter = ctx.anonymous_counters.get(scope_prefix, 0)
        ctx.anonymous_counters[scope_prefix] = counter + 1
        local_name = f"~{counter}"
    else:
        local_name = name
        
    full_name = f"{scope_prefix}/{local_name}" if scope_prefix else local_name
    dtype_obj = jnp.dtype(dtype)
    
    # Format config to sorted tuple of pairs for hashable / frozen storage
    hashable_config = tuple(sorted((str(k), v) for k, v in (config or {}).items()))

    # Build the request specification
    request_spec = RequestSpec(
        name=full_name,
        kind=kind,
        shape=shape,
        dtype=dtype_obj,
        config=hashable_config,
    )

    # 4. Handle based on active mode
    if ctx.mode == "init":
        # Check for duplicate parameter name collision
        if any(req.name == full_name for req in ctx.trace):
            raise DuplicateParameterError(
                f"Duplicate parameter name '{full_name}' detected in scope '{scope_prefix}'."
            )

        # Record request trace
        ctx.trace.append(request_spec)

        # Draw and execute initializer (only splitting key if needs_rng is True)
        if needs_rng:
            if ctx.rng_key is None:
                raise ValueError("RNG key is required for random parameter initialization.")
            ctx.rng_key, subkey = jax.random.split(ctx.rng_key)
        else:
            subkey = None
            
        value = init_fn(subkey)
        ctx.params[full_name] = value
        return value

    elif ctx.mode == "apply":
        # Record request trace
        ctx.trace.append(request_spec)

        idx = len(ctx.trace) - 1
        if ctx.init_trace is None:
            raise TraceMismatchError("Initialization trace/spec is missing during apply.")

        if idx >= len(ctx.init_trace):
            raise TraceMismatchError(
                f"Trace mismatch detected during apply.\n"
                f"Apply requested more parameters than recorded in the spec.\n"
                f"Extra parameter: '{full_name}'\n"
                f"Anonymous parameters should not be placed inside data-dependent control flow "
                f"or variable-length loops unless the structure is static. Suggest using explicit names."
            )

        expected_spec = ctx.init_trace[idx]
        if expected_spec != request_spec:
            diff_msg = []
            if expected_spec.name != request_spec.name:
                diff_msg.append(f"Expected name: '{expected_spec.name}', got: '{request_spec.name}'")
            if expected_spec.kind != request_spec.kind:
                diff_msg.append(f"Expected kind: '{expected_spec.kind}', got: '{request_spec.kind}'")
            if expected_spec.shape != request_spec.shape:
                diff_msg.append(f"Expected shape: {expected_spec.shape}, got: {request_spec.shape}")
            if expected_spec.dtype != request_spec.dtype:
                diff_msg.append(f"Expected dtype: {expected_spec.dtype}, got: {request_spec.dtype}")
            if expected_spec.config != request_spec.config:
                diff_msg.append(f"Expected config: {dict(expected_spec.config)}, got: {dict(request_spec.config)}")
                
            raise TraceMismatchError(
                f"Trace mismatch detected during apply at parameter index {idx}.\n"
                f"Differences:\n" + "\n".join(f" - {m}" for m in diff_msg)
            )

        # Retrieve and validate parameter from the bundle values
        if full_name not in ctx.params:
            raise MissingParameterError(
                f"Parameter '{full_name}' is missing from the stored parameter bundle."
            )

        value = ctx.params[full_name]
        if value.shape != shape:
            raise ShapeMismatchError(
                f"Shape mismatch for parameter '{full_name}': expected {shape}, got {value.shape}"
            )
        if jnp.dtype(value.dtype) != dtype_obj:
            raise DTypeMismatchError(
                f"Dtype mismatch for parameter '{full_name}': expected {dtype_obj}, got {value.dtype}"
            )

        return value
    else:
        raise ValueError(f"Invalid context mode: {ctx.mode}")


class Transformed:
    """Wrapper class returned by transform."""
    def __init__(self, fn: Callable[..., Any], auto_names: str = "allow"):
        self.fn = fn
        self.auto_names = auto_names
        self._has_warned = False

    def init(self, key: Any, *args: Any, **kwargs: Any) -> ParamBundle:
        ctx = Context(
            mode="init",
            params={},
            rng_key=key,
            auto_names=self.auto_names,
            on_warn_anonymous=self._get_warn_callback(),
        )
        with use_context(ctx):
            _ = self.fn(*args, **kwargs)
            
        spec = ParamSpec(requests=tuple(ctx.trace))
        return ParamBundle(values=ctx.params, spec=spec)

    def apply(self, bundle: ParamBundle, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(bundle, ParamBundle):
            raise SpecError(
                f"apply must be called with a ParamBundle instance, got: {type(bundle).__name__}."
            )
            
        ctx = Context(
            mode="apply",
            params=bundle.values,
            init_trace=bundle.spec.requests,
            auto_names=self.auto_names,
            on_warn_anonymous=self._get_warn_callback(),
        )
        with use_context(ctx):
            out = self.fn(*args, **kwargs)

        if len(ctx.trace) != len(bundle.spec.requests):
            raise TraceMismatchError(
                f"Trace mismatch detected during apply.\n"
                f"Expected {len(bundle.spec.requests)} parameters, but only {len(ctx.trace)} were requested.\n"
                f"Anonymous parameters should not be placed inside data-dependent control flow "
                f"or variable-length loops unless the structure is static. Suggest using explicit names."
            )
        return out

    def pure(self) -> Callable[..., Any]:
        return self.apply

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            "Transformed objects cannot be called directly. Use '.init(key, ...)' or '.apply(bundle, ...)' instead."
        )

    def _get_warn_callback(self) -> Callable[[], None]:
        def callback():
            if not self._has_warned:
                import warnings
                warnings.warn(
                    f"Anonymous parameter used in transformed function '{self.fn.__name__}'. "
                    f"Consider using explicit names to make your function safe against trace mismatches.",
                    UserWarning,
                    stacklevel=4,
                )
                self._has_warned = True
        return callback


def transform(fn: Optional[Callable[..., Any]] = None, *, auto_names: str = "allow") -> Any:
    """Decorator/wrapper to transform a function for explicit parameter management."""
    if auto_names not in ("allow", "warn", "error"):
        raise ValueError(f"Invalid auto_names mode: {auto_names}")

    if fn is None:
        def decorator(f: Callable[..., Any]) -> Transformed:
            return Transformed(f, auto_names=auto_names)
        return decorator
    else:
        if not callable(fn):
            raise ValueError("First argument to transform must be callable or None.")
        return Transformed(fn, auto_names=auto_names)
