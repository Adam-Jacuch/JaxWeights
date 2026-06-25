import jax
import jax.numpy as jnp
import pytest
import jaxweights as jw


def test_scope_keys():
    @jw.transform
    def mlp(x):
        with jw.scope("layer1"):
            x = x @ jw.zeros("w", x.shape[-1], 32) + jw.ones("b", 32)
        with jw.scope("layer2"):
            x = x @ jw.zeros("w", 32, 10) + jw.ones("b", 10)
        return x

    key = jax.random.key(0)
    x = jnp.ones((2, 16))
    bundle = mlp.init(key, x)
    
    assert "layer1/w" in bundle.values
    assert "layer1/b" in bundle.values
    assert "layer2/w" in bundle.values
    assert "layer2/b" in bundle.values


def test_independent_anonymous_counters_per_scope():
    @jw.transform
    def mlp(x):
        with jw.scope("layer1"):
            x = x @ jw.ones(x.shape[-1], 32) + jw.zeros(32)
        with jw.scope("layer2"):
            x = x @ jw.ones(32, 10) + jw.zeros(10)
        return x

    key = jax.random.key(0)
    x = jnp.ones((2, 16))
    bundle = mlp.init(key, x)
    
    assert "layer1/~0" in bundle.values
    assert "layer1/~1" in bundle.values
    assert "layer2/~0" in bundle.values
    assert "layer2/~1" in bundle.values


def test_nested_scopes():
    @jw.transform
    def nested(x):
        with jw.scope("outer"):
            with jw.scope("inner"):
                w = jw.zeros("w", x.shape[-1], 10)
                b = jw.zeros(10)
        return x @ w + b

    key = jax.random.key(0)
    x = jnp.ones((2, 5))
    bundle = nested.init(key, x)
    
    assert "outer/inner/w" in bundle.values
    assert "outer/inner/~0" in bundle.values
