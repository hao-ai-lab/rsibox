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
    "live": 0,
    "live_swap": 0,
    "in_place": 0,
    "hot_swap": 0,
    "r1": 1,
    "graph_recapture": 1,
    "recompile": 1,
    "compile_reset": 1,
    "r2": 2,
    "worker_respawn": 2,
    "fork": 2,
    "forked": 2,
    "zygote": 2,
    "zygote_refresh": 2,
    "fork_refresh": 2,
    "r3": 3,
    "code_restart": 3,
    "warm_restart": 3,
    "engine_restart": 3,
    "r4": 4,
    "cold_restart": 4,
    "full_restart": 4,
    "container_restart": 4,
    # the kernel's own names (INTERFACE-v1.md mechanisms, swap.py routes)
    "r10": 0,          # hand repair cell: in process, ms per call
    "t1": 1,           # rebuild the objects in place (constructor edits, new fields)
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


def _route_tier_weight(route_name: Any) -> int | None:
    if route_name is None:
        return None
    normalized = str(route_name).strip().lower()
    if normalized not in ROUTE_TIER_WEIGHTS and normalized.split():
        normalized = normalized.split()[0]       # "R0 swap", "R10 repair": the protocol key
    if normalized in ROUTE_TIER_WEIGHTS:
        return ROUTE_TIER_WEIGHTS[normalized]
    if normalized.startswith("r") and normalized[1:].isdigit():
        return int(normalized[1:])
    return None


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

    # 3. silent_tier_escalation
    if resolved_brief is not None:
        actual_route = (
            record.get("route")
            or record.get("actual_route")
            or record.get("tier")
        )
        predicted_weight = _route_tier_weight(resolved_brief.predicted_route)
        actual_weight = _route_tier_weight(actual_route)
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
        or route_lower in ("fork", "forked", "zygote", "zygote_refresh", "fork_refresh")
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
