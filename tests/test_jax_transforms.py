import jax
import jax.numpy as jnp
import pytest
import jaxweights as jw


def test_jax_grad_and_value_and_grad():
    @jw.transform
    def model(x):
        w = jw.normal("w", x.shape[-1], 1)
        b = jw.zeros("b", 1)
        return x @ w + b

    key = jax.random.key(0)
    x = jnp.array([[2.0, 3.0]])
    bundle = model.init(key, x)
    
    loss_fn = lambda bnd, x_val: jnp.sum(model.apply(bnd, x_val) ** 2)
    
    val, grads = jax.value_and_grad(loss_fn)(bundle, x)
    assert val.shape == ()
    
    assert isinstance(grads, jw.ParamBundle)
    assert grads.spec == bundle.spec
    assert "w" in grads.values
    assert "b" in grads.values
    assert grads.values["w"].shape == (2, 1)
    assert grads.values["b"].shape == (1,)


def test_jax_jit():
    @jw.transform
    def model(x):
        w = jw.normal("w", x.shape[-1], 4)
        return x @ w

    key = jax.random.key(0)
    x = jnp.ones((2, 8))
    bundle = model.init(key, x)
    
    jitted_apply = jax.jit(model.apply)
    y1 = model.apply(bundle, x)
    y2 = jitted_apply(bundle, x)
    assert jnp.allclose(y1, y2)


def test_jax_vmap():
    @jw.transform
    def model(x):
        w = jw.normal("w", x.shape[-1], 4)
        return x @ w

    key = jax.random.key(0)
    x_batched = jnp.ones((5, 2, 8))
    bundle = model.init(key, x_batched[0])
    
    vmapped_apply = jax.vmap(model.apply, in_axes=(None, 0))
    y = vmapped_apply(bundle, x_batched)
    assert y.shape == (5, 2, 4)
