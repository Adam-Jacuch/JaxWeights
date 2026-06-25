from typing import Any, Callable, Tuple, Union, Dict
import numpy as np
import jax
import jax.numpy as jnp

from jaxweights._core import parse_name_and_shape, param


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


def lecun_normal(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
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


def lecun_uniform(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
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


def xavier_normal(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
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


def glorot_normal(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Alias for xavier_normal."""
    return xavier_normal(*args, dtype=dtype)


def xavier_uniform(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
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


def glorot_uniform(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Alias for xavier_uniform."""
    return xavier_uniform(*args, dtype=dtype)


def he_normal(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
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


def kaiming_normal(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Alias for he_normal."""
    return he_normal(*args, dtype=dtype)


def he(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Alias for he_normal."""
    return he_normal(*args, dtype=dtype)


def he_uniform(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
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


def kaiming_uniform(*args: Any, dtype: Any = jnp.float32) -> jax.Array:
    """Alias for he_uniform."""
    return he_uniform(*args, dtype=dtype)


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
