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


def test_grouped_semantics_basic():
    # 1. Direct behavior still works
    @jw.transform
    def model_direct():
        p1 = jw.xavier_normal("p1", 128)
        p2 = jw.xavier_normal("p2", 3, 128, 8, 16)
        return p1, p2

    key = jax.random.key(0)
    bundle_direct = model_direct.init(key)
    assert bundle_direct.values["p1"].shape == (128,)
    assert bundle_direct.values["p2"].shape == (3, 128, 8, 16)

    # 2. Grouped QKV packed transformer projection
    @jw.transform
    def model_qkv():
        return jw.xavier_normal[3](128)(8, 16)

    bundle_qkv = model_qkv.init(key)
    p_qkv = bundle_qkv.values["~0"]
    assert p_qkv.shape == (3, 128, 8, 16)
    # expected stddev close to sqrt(2 / (128 + 8 * 16)) -> sqrt(2 / 256) -> sqrt(1 / 128) -> 0.088388
    expected_std_qkv = np.sqrt(2.0 / (128 + 8 * 16))
    assert np.abs(np.std(p_qkv) - expected_std_qkv) < 0.01

    # 3. Grouped no-batch output projection
    @jw.transform
    def model_proj():
        return jw.xavier_normal[:](8, 16)(128)

    bundle_proj = model_proj.init(key)
    p_proj = bundle_proj.values["~0"]
    assert p_proj.shape == (8, 16, 128)
    expected_std_proj = np.sqrt(2.0 / (8 * 16 + 128))
    assert np.abs(np.std(p_proj) - expected_std_proj) < 0.01


def test_grouped_variance_scaling():
    key = jax.random.key(0)

    # He Normal
    @jw.transform
    def model_he():
        return jw.he_normal[:](128)(256)

    bundle_he = model_he.init(key)
    p_he = bundle_he.values["~0"]
    assert p_he.shape == (128, 256)
    expected_std_he = np.sqrt(2.0 / 128)
    assert np.abs(np.std(p_he) - expected_std_he) < 0.01

    # LeCun Normal
    @jw.transform
    def model_lecun():
        return jw.lecun_normal[:](128)(256)

    bundle_lecun = model_lecun.init(key)
    p_lecun = bundle_lecun.values["~0"]
    assert p_lecun.shape == (128, 256)
    expected_std_lecun = np.sqrt(1.0 / 128)
    assert np.abs(np.std(p_lecun) - expected_std_lecun) < 0.01

    # Xavier Uniform
    @jw.transform
    def model_xavier_u():
        return jw.xavier_uniform[:](128)(256)

    bundle_xavier_u = model_xavier_u.init(key)
    p_xavier_u = bundle_xavier_u.values["~0"]
    assert p_xavier_u.shape == (128, 256)
    limit_xavier_u = np.sqrt(6.0 / (128 + 256))
    assert np.all(p_xavier_u >= -limit_xavier_u) and np.all(p_xavier_u <= limit_xavier_u)

    # He Uniform
    @jw.transform
    def model_he_u():
        return jw.he_uniform[:](128)(256)

    bundle_he_u = model_he_u.init(key)
    p_he_u = bundle_he_u.values["~0"]
    assert p_he_u.shape == (128, 256)
    limit_he_u = np.sqrt(6.0 / 128)
    assert np.all(p_he_u >= -limit_he_u) and np.all(p_he_u <= limit_he_u)

    # LeCun Uniform
    @jw.transform
    def model_lecun_u():
        return jw.lecun_uniform[:](128)(256)

    bundle_lecun_u = model_lecun_u.init(key)
    p_lecun_u = bundle_lecun_u.values["~0"]
    assert p_lecun_u.shape == (128, 256)
    limit_lecun_u = np.sqrt(3.0 / 128)
    assert np.all(p_lecun_u >= -limit_lecun_u) and np.all(p_lecun_u <= limit_lecun_u)


def test_grouped_multiple_packed_and_named():
    key = jax.random.key(0)

    # Multiple packed dims
    @jw.transform
    def model_multi():
        return jw.xavier_normal[2, 3](128)(64)

    bundle_multi = model_multi.init(key)
    p_multi = bundle_multi.values["~0"]
    assert p_multi.shape == (2, 3, 128, 64)

    # Named grouped parameter with batch
    @jw.transform
    def model_named_batch():
        return jw.xavier_normal["w", 3](128)(64)

    bundle_named_batch = model_named_batch.init(key)
    assert "w" in bundle_named_batch.values
    assert bundle_named_batch.values["w"].shape == (3, 128, 64)

    # Named grouped parameter no batch
    @jw.transform
    def model_named_nobatch():
        return jw.xavier_normal["w", :](128)(64)

    bundle_named_nobatch = model_named_nobatch.init(key)
    assert "w" in bundle_named_nobatch.values
    assert bundle_named_nobatch.values["w"].shape == (128, 64)


def test_grouped_init_apply_and_trace_mismatch():
    key = jax.random.key(0)

    @jw.transform
    def model(step):
        if step == 0:
            return jw.xavier_normal[:](128)(64)
        else:
            # Different output dimension to trigger shape mismatch
            return jw.xavier_normal[:](128)(128)

    # Trace recording on init
    bundle = model.init(key, 0)
    assert bundle.values["~0"].shape == (128, 64)

    # Success apply with identical dims
    out = model.apply(bundle, 0)
    assert out.shape == (128, 64)

    # Failure with different shape (trace mismatch / shape mismatch)
    with pytest.raises(jw.TraceMismatchError):
        model.apply(bundle, 1)


def test_grouped_aliases_and_kinds():
    # Glorot normal/uniform, Kaiming normal/uniform, He normal
    key = jax.random.key(0)
    
    @jw.transform
    def model_aliases():
        g1 = jw.glorot_normal[3](128)(64)
        g2 = jw.glorot_uniform[:](128)(64)
        k1 = jw.kaiming_normal[3](128)(64)
        k2 = jw.kaiming_uniform[:](128)(64)
        h1 = jw.he[3](128)(64)
        return g1, g2, k1, k2, h1

    bundle = model_aliases.init(key)
    assert bundle.values["~0"].shape == (3, 128, 64)
    assert bundle.values["~1"].shape == (128, 64)
    assert bundle.values["~2"].shape == (3, 128, 64)
    assert bundle.values["~3"].shape == (128, 64)
    assert bundle.values["~4"].shape == (3, 128, 64)

    # Verify kinds in the registered trace
    # (glorot_normal -> xavier_normal, glorot_uniform -> xavier_uniform,
    # kaiming_normal -> he_normal, kaiming_uniform -> he_uniform, he -> he_normal)
    trace = bundle.spec.requests
    assert trace[0].kind == "xavier_normal"
    assert trace[1].kind == "xavier_uniform"
    assert trace[2].kind == "he_normal"
    assert trace[3].kind == "he_uniform"
    assert trace[4].kind == "he_normal"


def test_grouped_validation_errors():
    # Invalid slice types
    with pytest.raises(ValueError):
        jw.xavier_normal[1:3]

    with pytest.raises(ValueError):
        jw.xavier_normal[::2]

    # Bool slice/batch dimensions
    with pytest.raises(ValueError):
        jw.xavier_normal[True]

    with pytest.raises(ValueError):
        jw.xavier_normal[False]

    # Negative dimensions in batch dims
    with pytest.raises(ValueError):
        jw.xavier_normal[-3]

    # Float/string dimensions in batch dims
    with pytest.raises(ValueError):
        jw.xavier_normal[1.5]

    # Empty dimensions validation
    @jw.transform
    def model_empty_in():
        return jw.xavier_normal[:]()(64)

    @jw.transform
    def model_empty_out():
        return jw.xavier_normal[:](128)()

    @jw.transform
    def model_invalid_in():
        return jw.xavier_normal[:](True)(64)

    @jw.transform
    def model_negative_in():
        return jw.xavier_normal[:](-128)(64)

    key = jax.random.key(0)
    with pytest.raises(ValueError):
        model_empty_in.init(key)
    with pytest.raises(ValueError):
        model_empty_out.init(key)
    with pytest.raises(ValueError):
        model_invalid_in.init(key)
    with pytest.raises(ValueError):
        model_negative_in.init(key)
