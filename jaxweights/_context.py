import contextlib
import contextvars
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple

from jaxweights._errors import ContextError, ReservedNameError
from jaxweights._bundle import RequestSpec


def validate_name(name: str, is_scope: bool = False):
    if not isinstance(name, str):
        raise ValueError(f"{'Scope' if is_scope else 'Parameter'} name must be a string, got {type(name).__name__}.")
    if not name:
        raise ValueError(f"{'Scope' if is_scope else 'Parameter'} name cannot be empty.")
    if "/" in name:
        raise ValueError(f"{'Scope' if is_scope else 'Parameter'} name cannot contain '/'. Got: '{name}'")
    if name in (".", ".."):
        raise ValueError(f"{'Scope' if is_scope else 'Parameter'} name cannot be '.' or '..'. Got: '{name}'")
    if name.startswith("~"):
        raise ReservedNameError(f"{'Scope' if is_scope else 'Parameter'} name cannot start with '~'. Got: '{name}'")


@dataclass
class Context:
    mode: str  # 'init' or 'apply'
    params: Dict[str, Any]  # parameter dictionary
    rng_key: Optional[Any] = None  # JAX PRNGKey, active during 'init'
    init_trace: Optional[Tuple[RequestSpec, ...]] = None  # trace from 'init' when in 'apply' mode
    scope_stack: List[str] = field(default_factory=list)
    anonymous_counters: Dict[str, int] = field(default_factory=dict)
    trace: List[RequestSpec] = field(default_factory=list)
    auto_names: str = "allow"
    on_warn_anonymous: Optional[Callable[[], None]] = None


_current_context = contextvars.ContextVar("jaxweights_context", default=None)


def get_context() -> Context:
    ctx = _current_context.get()
    if ctx is None:
        raise ContextError("jaxweights operations must be called inside a @jw.transform function.")
    return ctx


def has_context() -> bool:
    return _current_context.get() is not None


@contextlib.contextmanager
def use_context(ctx: Context):
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)


@contextlib.contextmanager
def scope(name: str):
    validate_name(name, is_scope=True)
    ctx = get_context()
    ctx.scope_stack.append(name)
    try:
        yield
    finally:
        ctx.scope_stack.pop()
