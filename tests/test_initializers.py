import jax
import jax.numpy as jnp
import numpy as np
import pytest
import jaxweights as jw


def test_rng_consumption():
    @jw.transform
    def a():
        w1 = jw.normal("w1", 10, 10)
        w2 = jw.normal("w2", 10, 10)
        return w1, w2

    @jw.transform
    def b():
        w1 = jw.normal("w1", 10, 10)
        z = jw.zeros("z", 10)
        w2 = jw.normal("w2", 10, 10)
        return w1, z, w2

    key = jax.random.key(0)
    ba = a.init(key)
    bb = b.init(key)

    assert jnp.array_equal(ba.values["w1"], bb.values["w1"])
    assert jnp.array_equal(ba.values["w2"], bb.values["w2"])


def test_random_determinism_and_shapes():
    @jw.transform
    def model():
        p1 = jw.normal("p1", 10, 10)
        p2 = jw.norm("p2", (10, 10))
        p3 = jw.uniform("p3", 5, minval=1.0, maxval=2.0)
        p4 = jw.truncated_normal("p4", 5)
        return p1, p2, p3, p4

    key = jax.random.key(0)
    bundle1 = model.init(key)
    bundle2 = model.init(key)
    
    for k in bundle1.values:
        assert jnp.array_equal(bundle1.values[k], bundle2.values[k])
        
    bundle3 = model.init(jax.random.key(1))
    assert not jnp.array_equal(bundle1.values["p1"], bundle3.values["p1"])
    
    assert bundle1.values["p1"].shape == (10, 10)
    assert bundle1.values["p1"].dtype == jnp.float32
    assert bundle1.values["p3"].shape == (5,)
    assert jnp.all(bundle1.values["p3"] >= 1.0) & jnp.all(bundle1.values["p3"] <= 2.0)


def test_constants():
    @jw.transform
    def model():
        p1 = jw.zeros("p1", 2, 3)
        p2 = jw.zeroes("p2", (2, 3))
        p3 = jw.ones("p3", 4)
        p4 = jw.constant("p4", 5, value=3.5)
        return p1, p2, p3, p4

    key = jax.random.key(0)
    bundle = model.init(key)
    
    assert jnp.all(bundle.values["p1"] == 0.0)
    assert jnp.all(bundle.values["p2"] == 0.0)
    assert jnp.all(bundle.values["p3"] == 1.0)
    assert jnp.all(bundle.values["p4"] == 3.5)


def test_variance_scaling():
    @jw.transform
    def model():
        p_lecun = jw.lecun_normal("lecun", 1000, 1000)
        p_lecun_u = jw.lecun_uniform("lecun_u", 1000, 1000)
        p_xavier = jw.xavier_normal("xavier", 1000, 1000)
        p_glorot_u = jw.glorot_uniform("glorot_u", 1000, 1000)
        p_he = jw.he("he", 1000, 1000)
        p_kaiming_u = jw.kaiming_uniform("kaiming_u", 1000, 1000)
        return p_lecun, p_lecun_u, p_xavier, p_glorot_u, p_he, p_kaiming_u

    key = jax.random.key(42)
    bundle = model.init(key)
    
    assert jnp.abs(jnp.std(bundle.values["lecun"]) - 0.0316) < 0.005
    assert jnp.all(bundle.values["lecun_u"] >= -0.055) & jnp.all(bundle.values["lecun_u"] <= 0.055)
    assert jnp.abs(jnp.std(bundle.values["xavier"]) - 0.0316) < 0.005
    assert jnp.all(bundle.values["glorot_u"] >= -0.055) & jnp.all(bundle.values["glorot_u"] <= 0.055)
    assert jnp.abs(jnp.std(bundle.values["he"]) - 0.0447) < 0.005
    assert jnp.all(bundle.values["kaiming_u"] >= -0.078) & jnp.all(bundle.values["kaiming_u"] <= 0.078)


def test_orthogonal():
    @jw.transform
    def model():
        p1 = jw.orthogonal("p1", 10, 10)
        p2 = jw.orthogonal("p2", 5, 10)
        p3 = jw.orthogonal("p3", 10, 5)
        return p1, p2, p3

    key = jax.random.key(0)
    bundle = model.init(key)
    
    p1 = bundle.values["p1"]
    assert jnp.allclose(p1 @ p1.T, jnp.eye(10), atol=1e-5)
    
    p2 = bundle.values["p2"]
    assert jnp.allclose(p2 @ p2.T, jnp.eye(5), atol=1e-5)
    
    p3 = bundle.values["p3"]
    assert jnp.allclose(p3.T @ p3, jnp.eye(5), atol=1e-5)
    
    @jw.transform
    def bad_model():
        return jw.orthogonal("p", 10)
        
    with pytest.raises(ValueError):
        bad_model.init(key)


def test_identity():
    @jw.transform
    def model():
        p = jw.identity("p", 5, 5, gain=2.0)
        return p

    key = jax.random.key(0)
    bundle = model.init(key)
    assert jnp.allclose(bundle.values["p"], 2.0 * jnp.eye(5))
    
    @jw.transform
    def bad_model1():
        return jw.identity("p", 5, 6)
        
    with pytest.raises(ValueError):
        bad_model1.init(key)

    @jw.transform
    def bad_model2():
        return jw.identity("p", 5)
        
    with pytest.raises(ValueError):
        bad_model2.init(key)
