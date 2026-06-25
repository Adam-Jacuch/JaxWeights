import jax
import jax.numpy as jnp
import pytest
import warnings
import jaxweights as jw


def test_calling_outside_transform():
    with pytest.raises(jw.ContextError):
        jw.zeros(10)
    with pytest.raises(jw.ContextError):
        with jw.scope("layer"):
            pass


def test_auto_names_error():
    @jw.transform(auto_names="error")
    def bad_model(x):
        w = jw.zeros(10)
        return x

    key = jax.random.key(0)
    with pytest.raises(jw.ContextError):
        bad_model.init(key, None)


def test_auto_names_warn():
    @jw.transform(auto_names="warn")
    def model(x):
        w1 = jw.zeros(10)
        w2 = jw.zeros(10)
        return x

    key = jax.random.key(0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bundle = model.init(key, None)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "Anonymous parameter used in transformed function" in str(w[0].message)


def test_invalid_auto_names_mode():
    with pytest.raises(ValueError):
        @jw.transform(auto_names="invalid")
        def model(x):
            pass


def test_duplicate_parameter_name():
    @jw.transform
    def bad_model(x):
        w1 = jw.zeros("w", 5)
        w2 = jw.zeros("w", 5)
        return x

    key = jax.random.key(0)
    with pytest.raises(jw.DuplicateParameterError):
        bad_model.init(key, None)


def test_any_fullname_collision():
    @jw.transform
    def collision_model(x):
        jw.zeros("foo", 10)
        jw.ones("foo", 10)
        return x
        
    with pytest.raises(jw.DuplicateParameterError):
        collision_model.init(jax.random.key(0), None)


def test_reserved_explicit_names():
    @jw.transform
    def bad_model1(x):
        return jw.zeros("~0", 10)
        
    with pytest.raises(jw.ReservedNameError):
        bad_model1.init(jax.random.key(0), None)

    @jw.transform
    def bad_model2(x):
        return jw.zeros("w/b", 10)
        
    with pytest.raises(ValueError):
        bad_model2.init(jax.random.key(0), None)

    @jw.transform
    def bad_model3(x):
        return jw.zeros("", 10)
        
    with pytest.raises(ValueError):
        bad_model3.init(jax.random.key(0), None)


def test_invalid_scope_names():
    @jw.transform
    def bad_model1(x):
        with jw.scope(""):
            pass
        return x
        
    with pytest.raises(ValueError):
        bad_model1.init(jax.random.key(0), None)

    @jw.transform
    def bad_model2(x):
        with jw.scope("a/b"):
            pass
        return x
        
    with pytest.raises(ValueError):
        bad_model2.init(jax.random.key(0), None)

    @jw.transform
    def bad_model3(x):
        with jw.scope("~layer"):
            pass
        return x
        
    with pytest.raises(jw.ReservedNameError):
        bad_model3.init(jax.random.key(0), None)


def test_missing_parameter():
    @jw.transform
    def model(x):
        return jw.zeros("w", 5)

    key = jax.random.key(0)
    bundle = model.init(key, None)
    
    bad_bundle = jw.ParamBundle(values={}, spec=bundle.spec)
    with pytest.raises(jw.MissingParameterError):
        model.apply(bad_bundle, None)


def test_shape_mismatch():
    @jw.transform
    def model(x):
        return jw.zeros("w", 5)

    key = jax.random.key(0)
    bundle = model.init(key, None)
    
    bad_values = {"w": jnp.zeros((10,))}
    bad_bundle = jw.ParamBundle(values=bad_values, spec=bundle.spec)
    with pytest.raises(jw.ShapeMismatchError):
        model.apply(bad_bundle, None)


def test_dtype_mismatch():
    @jw.transform
    def model(x):
        return jw.zeros("w", 5, dtype=jnp.float32)

    key = jax.random.key(0)
    bundle = model.init(key, None)
    
    bad_values = {"w": jnp.zeros((5,), dtype=jnp.int32)}
    bad_bundle = jw.ParamBundle(values=bad_values, spec=bundle.spec)
    with pytest.raises(jw.DTypeMismatchError):
        model.apply(bad_bundle, None)


def test_trace_mismatch_control_flow():
    @jw.transform
    def model(x, cond=True):
        if cond:
            return jw.zeros("w", 5)
        else:
            return jw.zeros("b", 5)

    key = jax.random.key(0)
    bundle = model.init(key, None, cond=True)
    
    with pytest.raises(jw.TraceMismatchError) as exc_info:
        model.apply(bundle, None, cond=False)
    assert "Expected name: 'w', got: 'b'" in str(exc_info.value)


def test_trace_mismatch_config_change():
    def make_model(stddev):
        @jw.transform
        def model(x):
            return x @ jw.normal("w", x.shape[-1], 4, stddev=stddev)
        return model

    x = jnp.ones((2, 3))
    m1 = make_model(0.01)
    bundle = m1.init(jax.random.key(0), x)

    m2 = make_model(1.0)
    with pytest.raises(jw.TraceMismatchError) as exc_info:
        m2.apply(bundle, x)
    assert "Expected config: {'mean': 0.0, 'stddev': 0.01}, got: {'mean': 0.0, 'stddev': 1.0}" in str(exc_info.value)
