import jax
import jax.numpy as jnp
import pytest
import jaxweights as jw


def test_init_returns_bundle():
    @jw.transform
    def model(x):
        return x @ jw.zeros("w", x.shape[-1], 4)

    key = jax.random.key(0)
    x = jnp.ones((2, 3))
    bundle = model.init(key, x)
    
    assert isinstance(bundle, jw.ParamBundle)
    assert "w" in bundle.values
    assert bundle.values["w"].shape == (3, 4)
    assert len(bundle.spec.requests) == 1
    req = bundle.spec.requests[0]
    assert req.name == "w"
    assert req.kind == "zeros"
    assert req.shape == (3, 4)


def test_bundle_is_pytree():
    @jw.transform
    def model(x):
        return x @ jw.zeros("w", x.shape[-1], 4)

    key = jax.random.key(0)
    x = jnp.ones((2, 3))
    bundle = model.init(key, x)
    
    leaves = jax.tree_util.tree_leaves(bundle)
    assert len(leaves) == 1
    assert jnp.array_equal(leaves[0], bundle.values["w"])


def test_fresh_object_old_bundle():
    def make_model():
        @jw.transform
        def model(x):
            return x @ jw.he("w", x.shape[-1], 4)
        return model

    key = jax.random.key(0)
    x = jnp.ones((2, 3))
    m1 = make_model()
    bundle = m1.init(key, x)

    m2 = make_model()
    y = m2.apply(bundle, x)
    assert y.shape == (2, 4)


def test_reinit_does_not_break_old_bundles():
    @jw.transform
    def model(x):
        return x @ jw.he("w", x.shape[-1], 4)

    key = jax.random.key(0)
    b1 = model.init(key, jnp.ones((2, 3)))
    b2 = model.init(key, jnp.ones((2, 5)))

    y1 = model.apply(b1, jnp.ones((2, 3)))
    y2 = model.apply(b2, jnp.ones((2, 5)))

    assert y1.shape == (2, 4)
    assert y2.shape == (2, 4)


def test_apply_with_plain_dict_raises():
    @jw.transform
    def model(x):
        return x @ jw.zeros("w", x.shape[-1], 4)

    key = jax.random.key(0)
    x = jnp.ones((2, 3))
    bundle = model.init(key, x)
    
    with pytest.raises(jw.SpecError):
        model.apply(bundle.values, x)


def test_extra_values_in_bundle_values():
    @jw.transform
    def model(x):
        return x @ jw.zeros("w", x.shape[-1], 4)

    key = jax.random.key(0)
    x = jnp.ones((2, 3))
    bundle = model.init(key, x)
    
    extended_values = dict(bundle.values)
    extended_values["extra_param"] = jnp.ones((10,))
    new_bundle = jw.ParamBundle(values=extended_values, spec=bundle.spec)
    
    y = model.apply(new_bundle, x)
    assert y.shape == (2, 4)
