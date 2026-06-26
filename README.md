# JaxWeights 🏋️

`jaxweights` is a tiny, noninvasive JAX parameter management layer. It lets you request parameters inline inside ordinary JAX functions, then transforms those functions into explicit parameter bundles (`ParamBundle`) + pure functional `apply` functions.

---

## What JaxWeights Is
* A lightweight utility to manage parameter initialization, request traces, and validations.
* Pure and noninvasive: it does not force you into a specific neural-network module/layer hierarchy.
* Fully compatible with standard JAX transformations (`jax.jit`, `jax.grad`, `jax.vmap`).
* Completely stateless: `apply` is a pure function of function inputs and a `ParamBundle`. It does not rely on hidden mutable state in the transformed object.

## What JaxWeights Is NOT
* A full-blown neural-network module system (like Flax, Haiku, or Equinox).
* It does NOT provide layers, stateful features (like BatchNorm tracking), optimizers, serialization, or training loops.

---

## Installation for Local Development

Install `jaxweights` in editable mode inside your virtual environment:

```bash
git clone https://github.com/yourusername/jaxweights.git
cd jaxweights
pip install -e .
```

---

## Usage Examples

### 1. Basic Example with `ParamBundle`

```python
import jax
import jax.numpy as jnp
import jaxweights as jw

@jw.transform
def forward(x):
    # Generates anonymous parameters internally: "~0" and "~1"
    w = jw.he(x.shape[-1], 64)
    b = jw.zeros(64)
    return jax.nn.relu(x @ w + b)

key = jax.random.key(0)
x = jnp.ones((8, 16))

# Initialize returns a ParamBundle containing values and spec
bundle = forward.init(key, x)
print(type(bundle))          # <class "jaxweights._bundle.ParamBundle">
print(bundle.values.keys())  # dict_keys(["~0", "~1"])

# Pure apply using the ParamBundle
y = forward.apply(bundle, x)
```

### 2. Explicit Parameter Names and Scopes

```python
@jw.transform
def mlp(x):
    with jw.scope("layer1"):
        x = jax.nn.relu(x @ jw.he("w", x.shape[-1], 64) + jw.zeros("b", 64))
    with jw.scope("layer2"):
        x = x @ jw.he("w", 64, 10) + jw.zeros("b", 10)
    return x

bundle = mlp.init(key, x)
# Keys are scoped and deterministic: "layer1/w", "layer1/b", "layer2/w", "layer2/b"
print(bundle.values.keys())
```

---

## JAX Transforms and PyTree Registration

`ParamBundle` is registered as a native JAX PyTree node. The trainable arrays in `values` are dynamic leaf nodes, whereas `spec` is treated as static auxiliary metadata.

### Automatic Differentiation (`jax.grad` / `jax.value_and_grad`)
Gradients computed on a `ParamBundle` preserve the bundle structure perfectly:

```python
def loss(b, x_val):
    return jnp.mean(mlp.apply(b, x_val) ** 2)

# grads is also a ParamBundle!
value, grads = jax.value_and_grad(loss)(bundle, x)
assert isinstance(grads, jw.ParamBundle)
assert grads.spec == bundle.spec
```

### Compilation (`jax.jit`)
```python
fast_apply = jax.jit(mlp.apply)
y = fast_apply(bundle, x)
```

### Vectorization (`jax.vmap`)
```python
vmapped_apply = jax.vmap(mlp.apply, in_axes=(None, 0))
batched_x = jnp.ones((4, 8, 16))
y = vmapped_apply(bundle, batched_x)
```

---

## Name and Scope Validation Rules

To prevent naming bugs and trace mismatches, explicit and scope names are strictly validated:
* Must be strings.
* Must be nonempty.
* Cannot contain `/`.
* Cannot be `.` or `..`.
* Cannot start with `~` (which is reserved for anonymous names only).

Violations raise a `ReservedNameError` or a `ValueError`. Any full-name collisions during `init` raise a `DuplicateParameterError`.

---

## RNG Consumption Policy

Deterministic initializers do not consume/split RNG key material. This guarantees that inserting or removing constant nodes (like `zeros`) does not shift the initialization of random matrices down the stream.

* **Random (Consumes RNG):** `normal`, `norm`, `truncated_normal`, `uniform`, `lecun_normal`, `lecun_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `orthogonal`.
* **Deterministic (No RNG):** `zeros`, `zeroes`, `ones`, `constant`, `identity`.

---

## Trace and Configuration Safety

During `init`, `jaxweights` compiles a static request spec of all parameter characteristics (names, kinds, shapes, dtypes, and configuration hyperparameters).
During `apply`, the function execution is re-traced and verified against the bundle spec. Alterations to shape, dtype, kinds, or initializer hyperparameters (e.g. changing `stddev=0.01` to `1.0`) will immediately trigger a `TraceMismatchError`.

---

## Available Initializers

* `zeros(*shape, dtype=jnp.float32)` (Alias: `zeroes`)
* `ones(*shape, dtype=jnp.float32)`
* `constant(*shape, value=0.0, dtype=jnp.float32)`
* `normal(*shape, mean=0.0, stddev=1.0, dtype=jnp.float32)` (Alias: `norm`)
* `truncated_normal(*shape, mean=0.0, stddev=1.0, lower=-2.0, upper=2.0, dtype=jnp.float32)`
* `uniform(*shape, minval=0.0, maxval=1.0, dtype=jnp.float32)`
* `lecun_normal(*shape, dtype=jnp.float32)`
* `lecun_uniform(*shape, dtype=jnp.float32)`
* `xavier_normal(*shape, dtype=jnp.float32)` (Alias: `glorot_normal`)
* `xavier_uniform(*shape, dtype=jnp.float32)` (Alias: `glorot_uniform`)
* `he_normal(*shape, dtype=jnp.float32)` (Alias: `kaiming_normal`, `he`)
* `he_uniform(*shape, dtype=jnp.float32)` (Alias: `kaiming_uniform`)
* `orthogonal(*shape, gain=1.0, dtype=jnp.float32)`
* `identity(*shape, gain=1.0, dtype=jnp.float32)`

---

## Grouped Fan Initializers

In addition to legacy shape-based initializers, `jaxweights` supports explicit grouped linear fan semantics using compact indexing syntax for all variance scaling/fan-based initializers (LeCun, Xavier/Glorot, He/Kaiming).

### Syntax

```python
# Grouped fan mode, no packed/batch dims
# shape == in_dims + out_dims
# fan_in == prod(in_dims), fan_out == prod(out_dims)
initializer[:](in_dims...)(out_dims...)

# Packed grouped fan mode, with packed/batch dims
# shape == batch_dims + in_dims + out_dims
# fan_in == prod(in_dims), fan_out == prod(out_dims)
initializer[batch_dims...](in_dims...)(out_dims...)

# Named variants
initializer['w', :](in_dims...)(out_dims...)
initializer['w', batch_dims...](in_dims...)(out_dims...)
```

### Examples

```python
# QKV packed transformer projection:
w = jw.xavier_normal[3](cfg.dim)(cfg.heads, hd)
# shape == (3, cfg.dim, cfg.heads, hd)
# fan_in == cfg.dim
# fan_out == cfg.heads * hd

# Attention output projection:
w = jw.xavier_normal[:](cfg.heads, hd)(cfg.dim)
# shape == (cfg.heads, hd, cfg.dim)
# fan_in == cfg.heads * hd
# fan_out == cfg.dim
```

### Transformer Usage

```python
q, k, v = jnp.einsum(
    "bsd,qdhk->qbhsk",
    x,
    jw.xavier_normal[3](cfg.dim)(cfg.heads, hd),
)

y = jnp.einsum(
    "bhsk,hkd->bsd",
    y,
    jw.xavier_normal[:](cfg.heads, hd)(cfg.dim),
)
```

---

## Caveats and Design Rules

1. **Always Initialize Outside `jit`**: Run `model.init(key, ...)` eagerly outside of any compiled contexts, compile/execute `model.apply(bundle, ...)` within `jit`.
2. **Stateless Decorator**: `apply` does not rely on previous calls to `init`. You can recreate transformed models on the fly; they will run successfully as long as their parameter sequence matches the bundle spec.
3. **No Direct Dict Params**: `apply` does not accept raw dictionaries of parameters. You must pass the complete `ParamBundle`. Raw parameter dictionaries alone are not verified.
4. **No Mutable State**: `jaxweights` focuses purely on parameter management. It does not handle running state (like moving averages in BatchNorm).
