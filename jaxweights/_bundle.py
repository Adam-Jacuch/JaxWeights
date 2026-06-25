from dataclasses import dataclass
from typing import Any, Dict, Tuple
import jax

@dataclass(frozen=True)
class RequestSpec:
    name: str
    kind: str
    shape: Tuple[int, ...]
    dtype: Any
    config: Tuple[Tuple[str, Any], ...]


@dataclass(frozen=True)
class ParamSpec:
    requests: Tuple[RequestSpec, ...]


@dataclass(frozen=True)
class ParamBundle:
    values: Dict[str, jax.Array]
    spec: ParamSpec


def bundle_flatten(bundle: ParamBundle):
    return (bundle.values,), bundle.spec


def bundle_unflatten(spec: ParamSpec, children: Tuple[Dict[str, jax.Array]]) -> ParamBundle:
    return ParamBundle(values=children[0], spec=spec)


jax.tree_util.register_pytree_node(
    ParamBundle,
    bundle_flatten,
    bundle_unflatten
)
