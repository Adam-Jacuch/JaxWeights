from typing import Any, Callable, Tuple, Union, Dict, Optional
import numpy as np
import jax
import jax.numpy as jnp

from jaxweights._core import parse_name_and_shape, param
from jaxweights._context import validate_name


def compute_fans(shape: Tuple[int, ...]) -> Tuple[float, float]:
    """Computes fan_in and fan_out for a given parameter shape."""
    if len(shape) == 0:
        return 1.0, 1.0
    if len(shape) == 1:
        return float(shape[0]), float(shape[0])
    if len(shape) == 2:
        return float(shape[0]), float(shape[1])
    
    # Convolution kernel convention: (*spatial_dims, in_channels, out_channels)
    in_channels = shape[-2]
    out_channels = shape[-1]
    receptive_field_size = 1
    for dim in shape[:-2]:
        receptive_field_size *= dim
    fan_in = float(in_channels * receptive_field_size)
    fan_out = float(out_channels * receptive_field_size)
    return fan_in, fan_out


def validate_dims(dims: Tuple[Any, ...], name: str) -> Tuple[int, ...]:
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims_tuple = tuple(dims[0])
    else:
        dims_tuple = tuple(dims)
        
    if not dims_tuple:
        raise ValueError(f"{name} must not be empty.")

    for dim in dims_tuple:
        if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)):
            raise ValueError(f"{name} dimensions must be integers, got {type(dim).__name__}: {dim}")
        if dim < 0:
            raise ValueError(f"{name} dimensions must be non-negative, got {dim}")
            
    return dims_tuple


def parse_item(item: Any) -> Tuple[Optional[str], Tuple[int, ...]]:
    name = None
    if isinstance(item, tuple):
        if len(item) > 0 and isinstance(item[0], str):
            name = item[0]
            batch_items = item[1:]
        else:
            batch_items = item
    else:
        batch_items = (item,)

    if not batch_items:
        raise ValueError("Must specify batch dimensions or [:] slice.")

    batch_dims = []
    if len(batch_items) == 1 and isinstance(batch_items[0], slice):
        sl = batch_items[0]
        if sl.start is not None or sl.stop is not None or sl.step is not None:
            raise ValueError(f"Only the plain [:] slice is supported, got {sl}")
        batch_dims = ()
    else:
        for x in batch_items:
            if isinstance(x, slice):
                raise ValueError(f"Only plain [:] is valid as a single item, got slice: {x}")
            if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
                raise ValueError(f"Batch dimensions must be integers, got {type(x).__name__}: {x}")
            if x < 0:
                raise ValueError(f"Batch dimensions must be non-negative, got {x}")
            batch_dims.append(int(x))
        batch_dims = tuple(batch_dims)

    if name is not None:
        validate_name(name, is_scope=False)

    return name, batch_dims


def compute_grouped_fans(in_dims: Tuple[int, ...], out_dims: Tuple[int, ...]) -> Tuple[float, float]:
    fan_in = 1
    for d in in_dims:
        fan_in *= d
    fan_out = 1
    for d in out_dims:
        fan_out *= d
    return float(fan_in), float(fan_out)


class GroupedFanWithIn:
    def __init__(
        self,
        batch_dims: Tuple[int, ...],
        name: Optional[str],
        in_dims: Tuple[int, ...],
        kind: str,
    ):
        self._batch_dims = batch_dims
        self._name = name
        self._in_dims = in_dims
        self._kind = kind

    def __call__(self, *out_dims: Any, dtype: Any = jnp.float32) -> jax.Array:
        out_dims_tuple = validate_dims(out_dims, "out_dims")
        
        # Determine canonical kind mapping
        canonical_kinds = {
            "lecun_normal": "lecun_normal",
            "lecun_uniform": "lecun_uniform",
            "xavier_normal": "xavier_normal",
            "xavier_uniform": "xavier_uniform",
            "glorot_normal": "xavier_normal",
            "xavier": "xavier_normal",
            "glorot_uniform": "xavier_uniform",
            "he_normal": "he_normal",
            "kaiming_normal": "he_normal",
            "he": "he_normal",
            "he_uniform": "he_uniform",
            "kaiming_uniform": "he_uniform",
        }
        canon_kind = canonical_kinds.get(self._kind, self._kind)
        
        fan_in, fan_out = compute_grouped_fans(self._in_dims, out_dims_tuple)
        shape = self._batch_dims + self._in_dims + out_dims_tuple
        
        if canon_kind == "lecun_normal":
            stddev = np.sqrt(1.0 / fan_in)
            init_fn = lambda key: stddev * jax.random.normal(key, shape, dtype=dtype)
        elif canon_kind == "lecun_uniform":
            limit = np.sqrt(3.0 / fan_in)
            init_fn = lambda key: jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=dtype)
        elif canon_kind == "xavier_normal":
            stddev = np.sqrt(2.0 / (fan_in + fan_out))
            init_fn = lambda key: stddev * jax.random.normal(key, shape, dtype=dtype)
        elif canon_kind == "xavier_uniform":
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            init_fn = lambda key: jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=dtype)
        elif canon_kind == "he_normal":
            stddev = np.sqrt(2.0 / fan_in)
            init_fn = lambda key: stddev * jax.random.normal(key, shape, dtype=dtype)
        elif canon_kind == "he_uniform":
            limit = np.sqrt(6.0 / fan_in)
            init_fn = lambda key: jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=dtype)
        else:
            raise ValueError(f"Unknown initializer kind: {self._kind}")
            
        config = {
            "fan_mode": "grouped",
            "batch_dims": tuple(int(d) for d in self._batch_dims),
            "in_dims": tuple(int(d) for d in self._in_dims),
            "out_dims": tuple(int(d) for d in out_dims_tuple),
            "fan_in": float(fan_in),
            "fan_out": float(fan_out),
        }
        
        return param(
            self._name,
            shape,
            init_fn,
            canon_kind,
            dtype=dtype,
            config=config,
            needs_rng=True,
        )


class GroupedFanWithBatch:
    def __init__(
        self,
        batch_dims: Tuple[int, ...],
        name: Optional[str],
        kind: str,
    ):
        self._batch_dims = batch_dims
        self._name = name
        self._kind = kind

    def __call__(self, *in_dims: Any) -> GroupedFanWithIn:
        in_dims_tuple = validate_dims(in_dims, "in_dims")
        return GroupedFanWithIn(
            batch_dims=self._batch_dims,
            name=self._name,
            in_dims=in_dims_tuple,
            kind=self._kind,
        )


class Initializer:
    def __init__(self, legacy_fn: Callable[..., jax.Array], kind: str):
        self._legacy_fn = legacy_fn
        self._kind = kind
        self.__doc__ = legacy_fn.__doc__

    def __call__(self, *args: Any, dtype: Any = jnp.float32, **kwargs: Any) -> jax.Array:
        return self._legacy_fn(*args, dtype=dtype, **kwargs)

    def __getitem__(self, item: Any) -> GroupedFanWithBatch:
        name, batch_dims = parse_item(item)
        return GroupedFanWithBatch(batch_dims=batch_dims, name=name, kind=self._kind)


def zeros(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Initialize parameter to all zeros."""
    name, shape = parse_name_and_shape(args)
    return param(
        name,
        shape,
        lambda key: jnp.zeros(shape, dtype=dtype),
        "zeros",
        dtype=dtype,
        config={},
        needs_rng=False,
    )


def zeroes(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Alias for zeros."""
    name, shape = parse_name_and_shape(args)
    return param(
        name,
        shape,
        lambda key: jnp.zeros(shape, dtype=dtype),
        "zeros",
        dtype=dtype,
        config={},
        needs_rng=False,
    )


def ones(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Initialize parameter to all ones."""
    name, shape = parse_name_and_shape(args)
    return param(
        name,
        shape,
        lambda key: jnp.ones(shape, dtype=dtype),
        "ones",
        dtype=dtype,
        config={},
        needs_rng=False,
    )


def constant(*args: Any, value: float = 0.0, dtype: Any = jnp.float32) -> jax.Array:
    """Initialize parameter to a constant value."""
    name, shape = parse_name_and_shape(args)
    return param(
        name,
        shape,
        lambda key: jnp.full(shape, value, dtype=dtype),
        "constant",
        dtype=dtype,
        config={"value": float(value)},
        needs_rng=False,
    )


def normal(*args: Any, mean: float = 0.0, stddev: float = 1.0, dtype: Any = jnp.float32) -> jax.Array:
    """Initialize parameter from a normal distribution."""
    name, shape = parse_name_and_shape(args)
    init_fn = lambda key: mean + stddev * jax.random.normal(key, shape, dtype=dtype)
    return param(
        name,
        shape,
        init_fn,
        "normal",
        dtype=dtype,
        config={"mean": float(mean), "stddev": float(stddev)},
        needs_rng=True,
    )


def norm(*args: Any, mean: float = 0.0, stddev: float = 1.0, dtype: Any = jnp.float32) -> jax.Array:
    """Alias for normal."""
    name, shape = parse_name_and_shape(args)
    init_fn = lambda key: mean + stddev * jax.random.normal(key, shape, dtype=dtype)
    return param(
        name,
        shape,
        init_fn,
        "normal",
        dtype=dtype,
        config={"mean": float(mean), "stddev": float(stddev)},
        needs_rng=True,
    )


def truncated_normal(
    *args: Any,
    mean: float = 0.0,
    stddev: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
    dtype: Any = jnp.float32,
) -> jax.Array:
    """Initialize parameter from a truncated normal distribution."""
    name, shape = parse_name_and_shape(args)
    init_fn = lambda key: mean + stddev * jax.random.truncated_normal(
        key, lower=lower, upper=upper, shape=shape, dtype=dtype
    )
    return param(
        name,
        shape,
        init_fn,
        "truncated_normal",
        dtype=dtype,
        config={"mean": float(mean), "stddev": float(stddev), "lower": float(lower), "upper": float(upper)},
        needs_rng=True,
    )


def uniform(*args: Any, minval: float = 0.0, maxval: float = 1.0, dtype: Any = jnp.float32) -> jax.Array:
    """Initialize parameter from a uniform distribution."""
    name, shape = parse_name_and_shape(args)
    init_fn = lambda key: jax.random.uniform(key, shape, minval=minval, maxval=maxval, dtype=dtype)
    return param(
        name,
        shape,
        init_fn,
        "uniform",
        dtype=dtype,
        config={"minval": float(minval), "maxval": float(maxval)},
        needs_rng=True,
    )


def _lecun_normal_legacy(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """LeCun normal initializer."""
    name, shape = parse_name_and_shape(args)
    fan_in, _ = compute_fans(shape)
    stddev = np.sqrt(1.0 / fan_in)
    init_fn = lambda key: stddev * jax.random.normal(key, shape, dtype=dtype)
    return param(
        name,
        shape,
        init_fn,
        "lecun_normal",
        dtype=dtype,
        config={},
        needs_rng=True,
    )


lecun_normal = Initializer(_lecun_normal_legacy, "lecun_normal")


def _lecun_uniform_legacy(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """LeCun uniform initializer."""
    name, shape = parse_name_and_shape(args)
    fan_in, _ = compute_fans(shape)
    limit = np.sqrt(3.0 / fan_in)
    init_fn = lambda key: jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=dtype)
    return param(
        name,
        shape,
        init_fn,
        "lecun_uniform",
        dtype=dtype,
        config={},
        needs_rng=True,
    )


lecun_uniform = Initializer(_lecun_uniform_legacy, "lecun_uniform")


def _xavier_normal_legacy(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Xavier/Glorot normal initializer."""
    name, shape = parse_name_and_shape(args)
    fan_in, fan_out = compute_fans(shape)
    stddev = np.sqrt(2.0 / (fan_in + fan_out))
    init_fn = lambda key: stddev * jax.random.normal(key, shape, dtype=dtype)
    return param(
        name,
        shape,
        init_fn,
        "xavier_normal",
        dtype=dtype,
        config={},
        needs_rng=True,
    )


xavier_normal = Initializer(_xavier_normal_legacy, "xavier_normal")


def _glorot_normal_legacy(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Alias for xavier_normal."""
    return xavier_normal(*args, dtype=dtype)


glorot_normal = Initializer(_glorot_normal_legacy, "glorot_normal")


def _xavier_legacy(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Alias for xavier_normal."""
    return xavier_normal(*args, dtype=dtype)


xavier = Initializer(_xavier_legacy, "xavier")


def _xavier_uniform_legacy(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Xavier/Glorot uniform initializer."""
    name, shape = parse_name_and_shape(args)
    fan_in, fan_out = compute_fans(shape)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    init_fn = lambda key: jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=dtype)
    return param(
        name,
        shape,
        init_fn,
        "xavier_uniform",
        dtype=dtype,
        config={},
        needs_rng=True,
    )


xavier_uniform = Initializer(_xavier_uniform_legacy, "xavier_uniform")


def _glorot_uniform_legacy(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Alias for xavier_uniform."""
    return xavier_uniform(*args, dtype=dtype)


glorot_uniform = Initializer(_glorot_uniform_legacy, "glorot_uniform")


def _he_normal_legacy(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """He/Kaiming normal initializer."""
    name, shape = parse_name_and_shape(args)
    fan_in, _ = compute_fans(shape)
    stddev = np.sqrt(2.0 / fan_in)
    init_fn = lambda key: stddev * jax.random.normal(key, shape, dtype=dtype)
    return param(
        name,
        shape,
        init_fn,
        "he_normal",
        dtype=dtype,
        config={},
        needs_rng=True,
    )


he_normal = Initializer(_he_normal_legacy, "he_normal")


def _kaiming_normal_legacy(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Alias for he_normal."""
    return he_normal(*args, dtype=dtype)


kaiming_normal = Initializer(_kaiming_normal_legacy, "kaiming_normal")


def _he_legacy(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Alias for he_normal."""
    return he_normal(*args, dtype=dtype)


he = Initializer(_he_legacy, "he")


def _he_uniform_legacy(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """He/Kaiming uniform initializer."""
    name, shape = parse_name_and_shape(args)
    fan_in, _ = compute_fans(shape)
    limit = np.sqrt(6.0 / fan_in)
    init_fn = lambda key: jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=dtype)
    return param(
        name,
        shape,
        init_fn,
        "he_uniform",
        dtype=dtype,
        config={},
        needs_rng=True,
    )


he_uniform = Initializer(_he_uniform_legacy, "he_uniform")


def _kaiming_uniform_legacy(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Alias for he_uniform."""
    return he_uniform(*args, dtype=dtype)


kaiming_uniform = Initializer(_kaiming_uniform_legacy, "kaiming_uniform")


def orthogonal(*args: Any, gain: float = 1.0, dtype: Any = jnp.float32) -> jax.Array:
    """Orthogonal initializer."""
    name, shape = parse_name_and_shape(args)
    if len(shape) < 2:
        raise ValueError(f"Orthogonal initializer requires at least 2 dimensions, got shape {shape}")
        
    def init_fn(key: Any) -> jax.Array:
        num_rows = int(np.prod(shape[:-1]))
        num_cols = int(shape[-1])
        
        transposed = num_rows < num_cols
        if transposed:
            matrix_shape = (num_cols, num_rows)
        else:
            matrix_shape = (num_rows, num_cols)
            
        z = jax.random.normal(key, matrix_shape, dtype=dtype)
        q, r = jnp.linalg.qr(z)
        
        d = jnp.diag(r)
        q = q * jnp.sign(d)
        
        if transposed:
            q = q.T
            
        return gain * q.reshape(shape)

    return param(
        name,
        shape,
        init_fn,
        "orthogonal",
        dtype=dtype,
        config={"gain": float(gain)},
        needs_rng=True,
    )


def identity(*args: Any, gain: float = 1.0, dtype: Any = jnp.float32) -> jax.Array:
    """Identity initializer (for 2D square matrices)."""
    name, shape = parse_name_and_shape(args)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"Identity initializer requires a square 2D shape, got shape {shape}")
    init_fn = lambda key: gain * jnp.eye(shape[0], dtype=dtype)
    return param(
        name,
        shape,
        init_fn,
        "identity",
        dtype=dtype,
        config={"gain": float(gain)},
        needs_rng=False,
    )
