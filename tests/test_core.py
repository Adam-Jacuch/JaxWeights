import jax
import jax.numpy as jnp
import pytest
import jaxweights as jw


def test_basic_explicit_init_apply():
    @jw.transform
    def forward(x):
        w = jw.zeros("w", x.shape[-1], 64)
        b = jw.ones("b", 64)
        return x @ w + b

    key = jax.random.key(0)
    x = jnp.ones((4, 16))
    bundle = forward.init(key, x)
    
    assert "w" in bundle.values
    assert "b" in bundle.values
    assert bundle.values["w"].shape == (16, 64)
    assert bundle.values["b"].shape == (64,)
    
    y = forward.apply(bundle, x)
    assert y.shape == (4, 64)
    assert jnp.allclose(y, 1.0)


def test_basic_anonymous_init_apply():
    @jw.transform
    def forward(x):
        w = jw.ones(x.shape[-1], 32)
        b = jw.zeros(32)
        return x @ w + b

    key = jax.random.key(0)
    x = jnp.ones((2, 8))
    bundle = forward.init(key, x)
    
    assert "~0" in bundle.values
    assert "~1" in bundle.values
    assert bundle.values["~0"].shape == (8, 32)
    assert bundle.values["~1"].shape == (32,)
    
    y = forward.apply(bundle, x)
    assert y.shape == (2, 32)
    assert jnp.allclose(y, 8.0)


def test_mixed_explicit_anonymous():
    @jw.transform
    def forward(x):
        w = jw.ones("w", x.shape[-1], 16)
        b = jw.zeros(16)
        return x @ w + b

    key = jax.random.key(0)
    x = jnp.ones((2, 4))
    bundle = forward.init(key, x)
    
    assert "w" in bundle.values
    assert "~0" in bundle.values
    assert bundle.values["w"].shape == (4, 16)
    assert bundle.values["~0"].shape == (16,)


def test_pure_callable():
    @jw.transform
    def forward(x):
        return x @ jw.ones("w", x.shape[-1], 8)

    key = jax.random.key(0)
    x = jnp.ones((2, 4))
    bundle = forward.init(key, x)
    
    pure_fn = forward.pure()
    y_apply = forward.apply(bundle, x)
    y_pure = pure_fn(bundle, x)
    assert jnp.array_equal(y_apply, y_pure)


def test_shape_parser_variadic_and_tuple():
    from jaxweights._core import parse_name_and_shape
    
    assert parse_name_and_shape((10,)) == (None, (10,))
    assert parse_name_and_shape(((2, 3),)) == (None, (2, 3))
    assert parse_name_and_shape((2, 3)) == (None, (2, 3))
    assert parse_name_and_shape(((),)) == (None, ())
    
    assert parse_name_and_shape(("w", 10)) == ("w", (10,))
    assert parse_name_and_shape(("w", (2, 3))) == ("w", (2, 3))
    assert parse_name_and_shape(("w", 2, 3)) == ("w", (2, 3))
    assert parse_name_and_shape(("w", ())) == ("w", ())


def test_shape_parser_rejects_invalid():
    from jaxweights._core import parse_name_and_shape
    
    with pytest.raises(ValueError):
        parse_name_and_shape(())
    with pytest.raises(ValueError):
        parse_name_and_shape(("w",))
    with pytest.raises(ValueError):
        parse_name_and_shape(("w", -1))
    with pytest.raises(ValueError):
        parse_name_and_shape(("w", 2.5))
    with pytest.raises(ValueError):
        parse_name_and_shape((-1,))
    with pytest.raises(ValueError):
        parse_name_and_shape((2.5,))
    with pytest.raises(ValueError):
        parse_name_and_shape((True,))
