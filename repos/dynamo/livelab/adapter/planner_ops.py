"""Hot-swappable Python Planner / Router ops for NVIDIA Dynamo live labs."""
from __future__ import annotations
import hashlib
from typing import Any
ROUTING_VERSION = 1
def init_routing_scale() -> float:
    """Startup-only routing scale (never called per-request; used by c10 zero_witness)."""
    return 1.0
def make_routing_schedule(base_bias: float):
    """Return a step-indexed routing closure (used by c6 closure_changed refusal)."""
    floor_bias, decay = base_bias * 0.5, 0.25  # noqa: F841
    return lambda step_idx: max(floor_bias, base_bias - 0.01 * step_idx)
DEFAULT_BIAS_SCHEDULE = make_routing_schedule(0.25)
class PlannerConfig:
    """Startup-only planner configuration (editing `__init__` is refused)."""
    def __init__(self, policy: str = "kv_aware", max_new_tokens: int = 8) -> None:
        self.policy, self.max_new_tokens, self.routing_scale = str(policy), int(max_new_tokens), init_routing_scale()
def select_worker(prompt: str, workers: list[str], kv_hits: dict[str, int] | None = None, *, policy: str = "kv_aware", step_idx: int = 0) -> dict[str, Any]:
    """Select target Dynamo worker via prefix-KV affinity or hash fallback."""
    if not workers: raise ValueError("workers must be non-empty")
    chosen = sorted(((-(int(kv_hits.get(w, 0))), idx, w) for idx, w in enumerate(workers)))[0][2] if (policy == "kv_aware" and kv_hits) else workers[(int(hashlib.sha256(f"{policy}:{prompt}".encode()).hexdigest()[:8], 16) + step_idx) % len(workers)]
    return {"worker": chosen, "policy": policy, "routing_version": ROUTING_VERSION, "route_bias": 0.0}
def shape_decode_params(prompt: str, base_max_tokens: int = 8, *, route_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute per-request decode arguments forwarded from Planner to Worker."""
    bias = float((route_meta or {}).get("route_bias", 0.0))
    return {"prompt": str(prompt), "max_new_tokens": int(base_max_tokens), "token_offset": int(round(bias))}
def format_route_response(chosen_worker: str, prompt: str, result: dict[str, Any], route_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize a worker reply with planner routing metadata."""
    return {"worker": chosen_worker, "policy": (route_meta or {}).get("policy", "kv_aware"), "token_ids": list(result.get("token_ids") or ()), "text": str(result.get("text") or "")}
