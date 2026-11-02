"""Frozen combinations of the four previously screened candidate components."""
from copy import deepcopy

from trace_vq.models.next_screen import NEXT_VARIANTS
from trace_vq.models.screening import VARIANTS

REFERENCES = ("baseline", "market_route", "ranknet", "pairwise_loss", "fsq")
COMPONENTS = {
    "baseline": {}, "market_route": VARIANTS["market_route"],
    "ranknet": NEXT_VARIANTS["ranknet"], "pairwise_loss": VARIANTS["pairwise_loss"],
    "fsq": NEXT_VARIANTS["fsq"],
}
COMBINATIONS = {
    "route_ranknet": ("market_route", "ranknet"),
    "route_pairwise": ("market_route", "pairwise_loss"),
    "fsq_route": ("fsq", "market_route"),
    "fsq_ranknet": ("fsq", "ranknet"),
    "fsq_pairwise": ("fsq", "pairwise_loss"),
    "fsq_route_ranknet": ("fsq", "market_route", "ranknet"),
    "fsq_route_pairwise": ("fsq", "market_route", "pairwise_loss"),
}
PHASES = [
    ("01_native_pairs", ["route_ranknet", "route_pairwise"]),
    ("02_fsq_pairs", ["fsq_route", "fsq_ranknet", "fsq_pairwise"]),
    ("03_three_components", ["fsq_route_ranknet", "fsq_route_pairwise"]),
]
PARENTS = {
    **{name: list(parts) for name, parts in COMBINATIONS.items() if len(parts) == 2},
    "fsq_route_ranknet": ["fsq_route", "fsq_ranknet", "route_ranknet"],
    "fsq_route_pairwise": ["fsq_route", "fsq_pairwise", "route_pairwise"],
}


def settings(name):
    parts = COMBINATIONS.get(name, (name,))
    if any(part not in COMPONENTS for part in parts):
        raise ValueError("Unknown combination")
    result = {"variant": name}
    for part in parts:
        for key, value in COMPONENTS[part].items():
            if key in result and result[key] != value:
                raise ValueError("Conflicting component settings")
            result[key] = deepcopy(value)
    if result.get("ranknet_weight") and result.get("pairwise_weight"):
        raise ValueError("Use one ranking objective per scientific configuration")
    return result


def family(name):
    return "fsq" if settings(name).get("quantizer_kind") == "fsq" else "native"
