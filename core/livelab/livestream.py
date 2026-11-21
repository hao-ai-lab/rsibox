"""Livestreamer cross-checking and read-only run dashboard harness for live-lab runs.

Provides:
- `PricedBrief`: structured pre-execution brief (`candidate_id`, `predicted_route`,
  `expected_overhead_seconds`, `noise_floor_pct`, `cheapest_probe`).
- `cross_check_step_record(record, brief=None)`: independent automated reviewer checks
  attached to each live step event (`zero_witness_pass`, `unaccepted_finding_ignored`,
  `silent_tier_escalation`, `unpaired_fork_throughput_claim`).
- `build_dashboard_summary(records, lock_holders=None)`: builds a structured read-only
  run dashboard view (`engine_status`, `restart_cost_ladder`, `hill_climb_ledger`,
  `standing_replay_summary`, `yield_and_spend`).
"""
from dataclasses import asdict, dataclass
from typing import Any


ROUTE_TIER_WEIGHTS = {
    "r0": 0,
    "r0 swap": 0,
    "swap": 0,
    "live": 0,
    "live_swap": 0,
    "in_place": 0,
    "hot_swap": 0,
    "r1": 1,
    "r1 recapture": 1,
    "recapture": 1,
    "graph_recapture": 1,
    "recompile": 1,
    "compile_reset": 1,
    "r2": 2,
    "r2 drain": 1,
    "drain": 1,
    "worker_respawn": 2,
    "fork": 2,
    "forked": 2,
    "zygote": 2,
    "zygote_refresh": 2,
    "fork_refresh": 2,
    "r3": 3,
    "r3 cache": 1,
    "cache": 1,
    "cache_reset": 1,
    "code_restart": 3,
    "warm_restart": 3,
    "engine_restart": 3,
    "r4": 4,
    "r4 rebuild": 1,
    "engine_core_rebuild": 1,
    "cold_restart": 4,
    "full_restart": 4,
    "container_restart": 4,
    "r5": 5,
    "soft_restart": 5,
    "r6": 6,
    "weight_reload": 6,
    "r7": 7,
    "native_reload": 7,
    "r8": 8,
    "r8 snapshot": 8,
    "snapshot": 8,
    "r9": 9,
    "r9 respawn": 9,
    "respawn": 9,
    # the kernel's own names (INTERFACE-v1.md mechanisms, swap.py routes)
    "r10": 0,          # hand repair cell: in process, ms per call
    "r10 repair": 0,
    "repair": 0,
    "t1": 1,           # rebuild the objects in place (constructor edits, new fields)
}

# Explicit ladder provenance for conflict detection between livestream's 5-tier coarse
# restart ladder (r2=worker_respawn, r3=code_restart, r4=cold_restart) and INTERFACE-v1's
# mechanism table (R2=drain, R3=cache, R4=engine-core rebuild, R1=recapture at ~5s).
_CONFLICTING_RUNGS = {
    "r2": ("worker_respawn", "drain"),
    "r3": ("code_restart", "cache"),
    "r4": ("cold_restart", "engine_core_rebuild"),
}
_LIVESTREAM_RUNG_TAGS = {
    "worker_respawn": ("r2", 2),
    "fork": ("r2", 2),
    "forked": ("r2", 2),
    "zygote": ("r2", 2),
    "zygote_refresh": ("r2", 2),
    "fork_refresh": ("r2", 2),
    "r2 worker_respawn": ("r2", 2),
    "code_restart": ("r3", 3),
    "warm_restart": ("r3", 3),
    "engine_restart": ("r3", 3),
    "r3 code_restart": ("r3", 3),
    "cold_restart": ("r4", 4),
    "full_restart": ("r4", 4),
    "container_restart": ("r4", 4),
    "r4 cold_restart": ("r4", 4),
}
_INTERFACE_RUNG_TAGS = {
    # INTERFACE-v1.md cost order:
    # R0/R10 (0) < T1 (1) < R2 drain 0.1ms (1) < R3 cache 1-2ms (2) < R4 rebuild 8-35ms (3)
    # < R5 soft restart 1.4-2.2s (4) < R1 recapture 4.5-7.8s (5) < R6 (6) < R7 (7) < R8 (8) < R9 (9)
    "t1": ("t1", 1),
    "drain": ("r2", 1),
    "r2 drain": ("r2", 1),
    "cache": ("r3", 2),
    "r3 cache": ("r3", 2),
    "cache_reset": ("r3", 2),
    "engine_core_rebuild": ("r4", 3),
    "r4 rebuild": ("r4", 3),
    "r5": ("r5", 4),
    "soft_restart": ("r5", 4),
    "r1 recapture": ("r1", 5),
    "recapture": ("r1", 5),
    "r6": ("r6", 6),
    "weight_reload": ("r6", 6),
    "r7": ("r7", 7),
    "native_reload": ("r7", 7),
    "r8": ("r8", 8),
    "r8 snapshot": ("r8", 8),
    "snapshot": ("r8", 8),
    "r9": ("r9", 9),
    "r9 respawn": ("r9", 9),
    "respawn": ("r9", 9),
}


@dataclass(frozen=True)
class PricedBrief:
    """Structured pre-execution brief pricing a candidate experiment before execution."""

    candidate_id: str
    predicted_route: str
    expected_overhead_seconds: float = 0.0
    noise_floor_pct: float = 0.0
    cheapest_probe: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PricedBrief":
        return cls(
            candidate_id=str(data.get("candidate_id", "")),
            predicted_route=str(data.get("predicted_route", "R0")),
            expected_overhead_seconds=float(data.get("expected_overhead_seconds", 0.0)),
            noise_floor_pct=float(data.get("noise_floor_pct", 0.0)),
            cheapest_probe=str(data.get("cheapest_probe", "")),
        )


_UNIVERSAL_WEIGHTS = {
    "r0": ("r0", 0, 0),
    "r0 swap": ("r0", 0, 0),
    "swap": ("r0", 0, 0),
    "live": ("r0", 0, 0),
    "live_swap": ("r0", 0, 0),
    "in_place": ("r0", 0, 0),
    "hot_swap": ("r0", 0, 0),
    "r10": ("r10", 0, 0),
    "r10 repair": ("r10", 0, 0),
    "repair": ("r10", 0, 0),
    "t1": ("t1", 1, 1),
    "r1": ("r1", 1, 5),
    "graph_recapture": ("r1", 1, 5),
    "recompile": ("r1", 1, 5),
    "compile_reset": ("r1", 1, 5),
}
_IFACE_BARE_WEIGHT = {"r2": 1, "r3": 2, "r4": 3}


def _resolve_route_spec(
    route_name: Any, context: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Resolve `route_name` into ladder provenance, rung key, and comparable tier weights."""
    if route_name is None:
        return None
    norm = " ".join(str(route_name).strip().lower().split())
    if not norm:
        return None
    head = norm.split()[0]

    def _from_token(tok: str) -> dict[str, Any] | None:
        if tok in _LIVESTREAM_RUNG_TAGS:
            rung, w = _LIVESTREAM_RUNG_TAGS[tok]
            return {"raw": route_name, "norm": norm, "ladder": "livestream",
                    "rung": rung, "weight": w, "iface_weight": None}
        if tok in _INTERFACE_RUNG_TAGS:
            rung, w = _INTERFACE_RUNG_TAGS[tok]
            return {"raw": route_name, "norm": norm, "ladder": "interface",
                    "rung": rung, "weight": ROUTE_TIER_WEIGHTS.get(tok, w), "iface_weight": w}
        if tok in _UNIVERSAL_WEIGHTS:
            rung, w, iw = _UNIVERSAL_WEIGHTS[tok]
            return {"raw": route_name, "norm": norm, "ladder": "both",
                    "rung": rung, "weight": w, "iface_weight": iw}
        if tok in _CONFLICTING_RUNGS:
            pmode = str((context or {}).get("process_mode") or
                        (context or {}).get("launch_mode") or "").strip().lower()
            is_forked = bool((context or {}).get("forked")) or pmode in (
                "fork", "forked", "zygote"
            )
            if tok == "r2" and is_forked:
                return {"raw": route_name, "norm": norm, "ladder": "livestream",
                        "rung": tok, "weight": ROUTE_TIER_WEIGHTS[tok], "iface_weight": None}
            return {"raw": route_name, "norm": norm, "ladder": "ambiguous",
                    "rung": tok, "weight": ROUTE_TIER_WEIGHTS[tok],
                    "iface_weight": _IFACE_BARE_WEIGHT[tok]}
        if tok in ROUTE_TIER_WEIGHTS:
            w = ROUTE_TIER_WEIGHTS[tok]
            return {"raw": route_name, "norm": norm, "ladder": "both",
                    "rung": tok, "weight": w, "iface_weight": w}
        if tok.startswith("r") and tok[1:].isdigit():
            w = int(tok[1:])
            return {"raw": route_name, "norm": norm, "ladder": "interface",
                    "rung": tok, "weight": w, "iface_weight": w}
        return None

    spec = _from_token(norm)
    if spec is not None:
        return spec
    if head != norm:
        return _from_token(head)
    return None


def _route_tier_weight(route_name: Any) -> int | None:
    spec = _resolve_route_spec(route_name)
    return spec["weight"] if spec is not None else None


def _conflict_rung(spec_a: dict[str, Any], spec_b: dict[str, Any]) -> str:
    """Deterministic rung label for a conflict pair, independent of field order."""
    if spec_a["rung"] == spec_b["rung"]:
        return spec_a["rung"].upper()
    in_a = spec_a["rung"] in _CONFLICTING_RUNGS
    in_b = spec_b["rung"] in _CONFLICTING_RUNGS
    if in_a and not in_b:
        return spec_a["rung"].upper()
    if in_b and not in_a:
        return spec_b["rung"].upper()
    return min(spec_a["rung"], spec_b["rung"]).upper()


def _specs_conflict(spec_a: dict[str, Any], spec_b: dict[str, Any]) -> str | None:
    """Return a conflict description if two route specs disagree across conflicting rungs."""
    if spec_a["rung"] == spec_b["rung"] and spec_a["rung"] in _CONFLICTING_RUNGS:
        if spec_a["ladder"] != spec_b["ladder"]:
            first, second = sorted(
                (spec_a, spec_b),
                key=lambda sp: (
                    {"livestream": 0, "interface": 1, "ambiguous": 2}.get(sp["ladder"], 3),
                    sp["norm"],
                ),
            )
            return (
                f"conflicting definitions for rung {first['rung'].upper()}: "
                f"{first['raw']!r} ({first['ladder']}) vs "
                f"{second['raw']!r} ({second['ladder']})"
            )
    ladders = {spec_a["ladder"], spec_b["ladder"]}
    # Only interface tags on the conflicting subset (R2 drain, R3 cache, R4 rebuild, R1 recapture)
    # conflict across ladders with livestream R2/R3/R4 (`R5..R9` remain heavier in both).
    if ladders == {"livestream", "interface"}:
        ls = spec_a if spec_a["ladder"] == "livestream" else spec_b
        if_ = spec_b if ls is spec_a else spec_a
        if if_["rung"] in _CONFLICTING_RUNGS or if_["norm"] in ("r1 recapture", "recapture"):
            return (
                f"incompatible ladder rungs: {ls['raw']!r} (livestream) "
                f"vs {if_['raw']!r} (INTERFACE-v1)"
            )
    if "ambiguous" in ladders:
        amb = spec_a if spec_a["ladder"] == "ambiguous" else spec_b
        other = spec_b if amb is spec_a else spec_a
        if (other["ladder"] == "interface" and other["rung"] in _CONFLICTING_RUNGS) or (
            amb["rung"] in ("r2", "r3", "r4") and other["rung"] in ("r1", "t1")
        ):
            return (
                f"ambiguous rung {amb['raw']!r} conflicts across ladders with "
                f"{other['raw']!r}"
            )
    return None


def detect_ladder_conflicts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect conflicts across `records` where the same protocol rung (e.g. R2/R3/R4)
    or step fields mix incompatible meanings from the livestream vs INTERFACE-v1 ladders."""
    seen_specs: dict[tuple[str, str, str], dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    seen_msgs: set[str] = set()

    def _record_conflict(cid: str, spec_a: dict[str, Any], spec_b: dict[str, Any]) -> None:
        msg = _specs_conflict(spec_a, spec_b)
        if msg and msg not in seen_msgs:
            seen_msgs.add(msg)
            conflicts.append({
                "check": "ladder_conflict",
                "candidate_id": cid,
                "rung": _conflict_rung(spec_a, spec_b),
                "observed": msg,
            })

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        cid = str(rec.get("candidate_id") or f"record_{idx}")
        field_vals = [
            rec.get(k) for k in ("route", "actual_route", "tier", "mechanism", "predicted_route")
        ]
        brief_val = rec.get("brief")
        if isinstance(brief_val, PricedBrief):
            field_vals.append(brief_val.predicted_route)
        elif isinstance(brief_val, dict) and "predicted_route" in brief_val:
            field_vals.append(brief_val.get("predicted_route"))
        mechs = rec.get("mechanisms")
        if isinstance(mechs, (list, tuple)):
            field_vals.extend(mechs)
        specs = [
            s for v in field_vals if (s := _resolve_route_spec(v, rec)) is not None
        ]
        for i in range(len(specs)):
            for j in range(i + 1, len(specs)):
                _record_conflict(cid, specs[i], specs[j])
        for s in specs:
            for prev in seen_specs.values():
                _record_conflict(cid, prev, s)
        for s in specs:
            seen_specs.setdefault((s["ladder"], s["rung"], s["norm"]), s)
    return conflicts


def _role_group(label: str) -> str:
    """Group key for witness participant labels: ranks of one pool share a group so a silent
    rank (`rank1: {}` beside `rank0: {k: 8}`, or `workers/rank1: {}` beside `workers/rank0`)
    still fails, while an uninvolved role (`core: {}` beside `workers/rank0`) can abstain."""
    text = str(label)
    if "/" in text:
        return text.split("/", 1)[0]
    stripped = text.rstrip("0123456789")
    if stripped and stripped != text:
        return f"_{stripped}"
    return text


def _extract_witness_info(record: dict[str, Any]) -> tuple[int, bool]:
    """Return `(total_hits, is_proven)` from a step record."""
    explicit_proven = record.get("proven")
    witness_data = record.get("witness")
    abstained = set(record.get("abstained") or ())

    if isinstance(witness_data, dict) and "witness" in witness_data and "proven" in witness_data:
        if explicit_proven is None:
            explicit_proven = bool(witness_data.get("proven"))
        abstained.update(witness_data.get("abstained") or ())
        witness_data = witness_data.get("witness")

    total_hits = 0
    has_zero_target = False
    if witness_data is None:
        total_hits = 0
    elif isinstance(witness_data, (int, float)):
        total_hits = int(witness_data)
    elif isinstance(witness_data, dict):
        if not witness_data:
            total_hits = 0
        elif all(isinstance(val, dict) for val in witness_data.values()):
            # per rank: every involved rank must run every target (a sum hides a rank that ran
            # none); an uninvolved role on a multi-role engine (`core: {}` on a worker edit)
            # abstains when no sibling rank of its role group staged anything.
            active_groups = {_role_group(label) for label, rank_map in witness_data.items()
                             if rank_map and label not in abstained}
            for label, rank_map in witness_data.items():
                if label in abstained:
                    continue
                if not rank_map and active_groups and _role_group(label) not in active_groups:
                    continue
                counts = [int(hit_count) for hit_count in rank_map.values()]
                total_hits += sum(counts)
                has_zero_target = has_zero_target or not counts or min(counts) <= 0
        else:
            counts = [int(val) for val in witness_data.values() if isinstance(val, (int, float))]
            total_hits = sum(counts)
            has_zero_target = any(count <= 0 for count in counts)

    if explicit_proven is False:
        return total_hits, False
    return total_hits, (total_hits > 0 and not has_zero_target)


def _extract_unaccepted_findings(record: dict[str, Any]) -> list[str]:
    raw_findings = record.get("findings") or []
    accepted_raw = (
        record.get("accepted_findings")
        if record.get("accepted_findings") is not None
        else record.get("accept")
    )
    # as kernel.findings.decide: "all" accepts every overridable finding; a name or an accept
    # record accepts that check; a finding that is not overridable is never accepted
    accept_all = accepted_raw == "all"
    if isinstance(accepted_raw, str):
        accepted_raw = () if accept_all else (accepted_raw,)
    accepted_set = {str(item.get("check")) if isinstance(item, dict) else str(item)
                    for item in (accepted_raw or ())}

    unaccepted: list[str] = []
    for item in raw_findings:
        if isinstance(item, dict):
            if item.get("advisory") is True:
                continue
            check_name = str(item.get("check") or item.get("finding") or "")
            if item.get("overridable") is False:
                if check_name:
                    unaccepted.append(check_name)
                continue
            if item.get("accepted") is True:     # annotated by decide
                continue
        else:
            check_name = str(item)
        if check_name and not accept_all and check_name not in accepted_set:
            unaccepted.append(check_name)
    return unaccepted


def cross_check_step_record(
    record: dict[str, Any],
    brief: PricedBrief | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run independent reviewer cross-checks on a live-lab step record.

    Detects:
    1. `zero_witness_pass`: `verdict == "pass"` or `kept == True` when `witness` has 0
       total hits, a rank or target with 0 hits, or `proven is False`.
    2. `unaccepted_finding_ignored`: open `findings` present that are not accepted
       (`accepted_findings` or `accept`, as `kernel.findings.decide`: "all" or the check's
       name; a finding with `overridable: False` never is).
    3. `silent_tier_escalation`: actual execution route/tier is heavier than
       `brief.predicted_route` without an explicit escalation reason.
    4. `unpaired_fork_throughput_claim`: a throughput gain is claimed (`kept == True`)
       from a `forked` rank run without `fresh_process_confirmed == True`.
    """
    resolved_brief: PricedBrief | None = None
    if isinstance(brief, PricedBrief):
        resolved_brief = brief
    elif isinstance(brief, dict):
        resolved_brief = PricedBrief.from_dict(brief)
    elif isinstance(record.get("brief"), PricedBrief):
        resolved_brief = record["brief"]
    elif isinstance(record.get("brief"), dict):
        resolved_brief = PricedBrief.from_dict(record["brief"])

    candidate_id = record.get("candidate_id") or (
        resolved_brief.candidate_id if resolved_brief is not None else "unknown"
    )
    verdict = str(record.get("verdict") or "").lower()
    is_kept = bool(record.get("kept"))
    is_pass_or_kept = verdict == "pass" or is_kept

    issues: list[dict[str, Any]] = []

    # 1. zero_witness_pass
    total_hits, is_proven = _extract_witness_info(record)
    if is_pass_or_kept and (not is_proven or total_hits <= 0):
        issues.append(
            {
                "check": "zero_witness_pass",
                "candidate_id": candidate_id,
                "severity": "blocking",
                "observed": f"verdict={verdict or 'kept'} with total_hits={total_hits}, proven={is_proven}",
                "expected": "at least 1 witness hit and proven=True before marking pass or kept",
            }
        )

    # 2. unaccepted_finding_ignored
    unaccepted_findings = _extract_unaccepted_findings(record)
    if unaccepted_findings:
        issues.append(
            {
                "check": "unaccepted_finding_ignored",
                "candidate_id": candidate_id,
                "severity": "blocking",
                "unaccepted": unaccepted_findings,
                "observed": f"unaccepted findings present: {', '.join(unaccepted_findings)}",
                "expected": "all open findings resolved or explicitly listed in accepted_findings",
            }
        )

    # 3. silent_tier_escalation & ladder_conflict (reconciled livestream + INTERFACE-v1)
    combined_for_conflict = dict(record)
    if resolved_brief is not None:
        combined_for_conflict["predicted_route"] = resolved_brief.predicted_route
    for conflict in detect_ladder_conflicts([combined_for_conflict]):
        issues.append(
            {
                "check": "ladder_conflict",
                "candidate_id": candidate_id,
                "severity": "blocking",
                "rung": conflict["rung"],
                "observed": conflict["observed"],
                "expected": "consistent ladder semantics without conflicting R2/R3/R4 rungs",
            }
        )

    if resolved_brief is not None:
        actual_route = (
            record.get("route")
            or record.get("actual_route")
            or record.get("tier")
        )
        pred_spec = _resolve_route_spec(resolved_brief.predicted_route)
        act_spec = _resolve_route_spec(actual_route, record)
        use_iface = (
            pred_spec is not None and act_spec is not None
            and (pred_spec["ladder"] == "interface" or act_spec["ladder"] == "interface")
        )

        def _spec_weight(sp):
            if sp is None:
                return None
            if use_iface and sp.get("iface_weight") is not None:
                return sp["iface_weight"]
            return sp["weight"]

        predicted_weight = _spec_weight(pred_spec)
        actual_weight = _spec_weight(act_spec)
        escalation_reason = (
            record.get("escalation_reason")
            or record.get("tier_escalation_reason")
            or record.get("route_reason")
            or record.get("reason")
        )
        if (
            predicted_weight is not None
            and actual_weight is not None
            and actual_weight > predicted_weight
            and not (escalation_reason and str(escalation_reason).strip())
        ):
            issues.append(
                {
                    "check": "silent_tier_escalation",
                    "candidate_id": candidate_id,
                    "severity": "blocking",
                    "predicted_route": resolved_brief.predicted_route,
                    "actual_route": actual_route,
                    "observed": (
                        f"escalated from predicted {resolved_brief.predicted_route} "
                        f"to {actual_route} without an explicit reason"
                    ),
                    "expected": "matching route tier or non-empty escalation_reason",
                }
            )

    # 4. unpaired_fork_throughput_claim
    route_lower = str(record.get("route") or record.get("actual_route") or "").strip().lower()
    process_mode = str(record.get("process_mode") or record.get("launch_mode") or "").strip().lower()
    is_forked_run = (
        bool(record.get("forked"))
        or process_mode in ("fork", "forked", "zygote")
        or route_lower in (
            "fork", "forked", "zygote", "zygote_refresh", "fork_refresh", "worker_respawn"
        )
    )
    gain_metric = float(
        record.get("delta_pct")
        or record.get("speedup_pct")
        or record.get("throughput_gain_pct")
        or 0.0
    )
    claims_throughput_gain = is_kept or bool(record.get("throughput_claim")) or (
        verdict == "pass" and gain_metric > 0.0
    )
    if is_forked_run and claims_throughput_gain and record.get("fresh_process_confirmed") is not True:
        issues.append(
            {
                "check": "unpaired_fork_throughput_claim",
                "candidate_id": candidate_id,
                "severity": "blocking",
                "observed": "throughput claim from a forked rank run without fresh_process_confirmed=True",
                "expected": "fresh_process_confirmed=True before keeping a forked throughput claim",
            }
        )

    return issues


def build_dashboard_summary(
    records: list[dict[str, Any]],
    lock_holders: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a structured read-only run dashboard summary across step `records`."""
    active_locks = list(lock_holders or [])
    latest_record = records[-1] if records else None

    engine_status = {
        "state": "busy" if active_locks else ("active" if records else "idle"),
        "lock_count": len(active_locks),
        "active_locks": active_locks,
        "total_steps": len(records),
        "latest_candidate_id": (
            latest_record.get("candidate_id") if latest_record is not None else None
        ),
        "latest_route": (
            latest_record.get("route") or latest_record.get("actual_route")
            if latest_record is not None
            else None
        ),
    }

    route_stats: dict[str, dict[str, Any]] = {}
    total_overhead_seconds = 0.0
    ledger_entries: list[dict[str, Any]] = []
    kept_candidates: list[str] = []
    cumulative_delta_pct = 0.0
    total_replays = 0
    passed_replays = 0
    failed_replays = 0
    flagged_records: list[dict[str, Any]] = []

    for record in records:
        candidate_id = str(record.get("candidate_id") or "unknown")
        route_name = str(record.get("route") or record.get("actual_route") or "R0")
        overhead_sec = float(
            record.get("overhead_seconds")
            or record.get("elapsed_seconds")
            or record.get("wall_seconds")
            or 0.0
        )
        total_overhead_seconds += overhead_sec

        bucket = route_stats.setdefault(
            route_name,
            {"count": 0, "total_overhead_seconds": 0.0, "mean_overhead_seconds": 0.0},
        )
        bucket["count"] += 1
        bucket["total_overhead_seconds"] = round(
            bucket["total_overhead_seconds"] + overhead_sec, 6
        )
        bucket["mean_overhead_seconds"] = round(
            bucket["total_overhead_seconds"] / bucket["count"], 6
        )

        cross_checks = cross_check_step_record(record)
        if cross_checks:
            flagged_records.append(
                {"candidate_id": candidate_id, "issues": cross_checks}
            )

        verdict = str(record.get("verdict") or "").lower()
        is_kept = bool(record.get("kept"))
        delta_pct = float(
            record.get("delta_pct")
            or record.get("speedup_pct")
            or record.get("throughput_gain_pct")
            or 0.0
        )
        _hits, is_proven = _extract_witness_info(record)

        if is_kept:
            kept_candidates.append(candidate_id)
            cumulative_delta_pct = round(cumulative_delta_pct + delta_pct, 6)

        if bool(record.get("standing_replay")) or record.get("kind") == "replay":
            total_replays += 1
            if verdict == "pass" and not cross_checks:
                passed_replays += 1
            else:
                failed_replays += 1

        ledger_entries.append(
            {
                "candidate_id": candidate_id,
                "route": route_name,
                "verdict": verdict or ("pass" if is_kept else "unknown"),
                "kept": is_kept,
                "delta_pct": delta_pct,
                "overhead_seconds": overhead_sec,
                "proven": is_proven,
                "cross_check_issues": cross_checks,
            }
        )

    restart_cost_ladder = {
        "by_route": route_stats,
        "total_overhead_seconds": round(total_overhead_seconds, 6),
        "ladder_conflicts": detect_ladder_conflicts(records),
    }

    hill_climb_ledger = {
        "entries": ledger_entries,
        "attempted_count": len(records),
        "kept_count": len(kept_candidates),
        "kept_candidates": kept_candidates,
        "cumulative_delta_pct": cumulative_delta_pct,
    }

    standing_replay_summary = {
        "total_replays": total_replays,
        "passed_replays": passed_replays,
        "failed_replays": failed_replays,
        "pass_rate": round(passed_replays / total_replays, 4) if total_replays else 0.0,
        "flagged_records": flagged_records,
    }

    yield_and_spend = {
        "total_candidates": len(records),
        "kept_candidates": len(kept_candidates),
        "yield_rate": round(len(kept_candidates) / len(records), 4) if records else 0.0,
        "total_spend_seconds": round(total_overhead_seconds, 6),
        "mean_spend_seconds": (
            round(total_overhead_seconds / len(records), 6) if records else 0.0
        ),
        "flagged_count": len(flagged_records),
    }

    return {
        "engine_status": engine_status,
        "restart_cost_ladder": restart_cost_ladder,
        "hill_climb_ledger": hill_climb_ledger,
        "standing_replay_summary": standing_replay_summary,
        "yield_and_spend": yield_and_spend,
    }
