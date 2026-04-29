from importlib import import_module

_REGISTRY = {
    "fvg": "patterns.fair_value_gap",
    "order_block": "patterns.order_block",
}


def load(pattern: str):
    if pattern not in _REGISTRY:
        raise ValueError(f"Unknown pattern '{pattern}'. Available: {list(_REGISTRY)}")
    return import_module(_REGISTRY[pattern])
