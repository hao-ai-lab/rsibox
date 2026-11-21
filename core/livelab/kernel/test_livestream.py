"""Unit tests for livestream cross-checking and run dashboard harness (`core/livelab/livestream.py`).

Run from the repo root:
    python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v
"""
import unittest

import livestream


class LivestreamCrossCheckTest(unittest.TestCase):
    """Verify all 4 independent reviewer checks in `cross_check_step_record`."""

    def test_clean_step_record_produces_no_issues(self):
        brief = livestream.PricedBrief(
            candidate_id="cand_clean",
            predicted_route="R0",
            expected_overhead_seconds=0.05,
            noise_floor_pct=0.5,
            cheapest_probe="step_latency",
        )
        record = {
            "candidate_id": "cand_clean",
            "route": "R0",
            "verdict": "pass",
            "kept": True,
            "witness": {"mod:fn:10": 4},
            "proven": True,
            "findings": [{"check": "jit_rebuild", "advisory": True}],
            "accepted_findings": [],
        }
        self.assertEqual(livestream.cross_check_step_record(record, brief), [])
        self.assertEqual(
            livestream.PricedBrief.from_dict(brief.to_dict()),
            brief,
        )

    def test_zero_witness_pass_detector(self):
        zero_hit_record = {
            "candidate_id": "cand_zero_hits",
            "verdict": "pass",
            "kept": True,
            "witness": {"mod:fn:10": 0},
            "proven": False,
        }
        issues = livestream.cross_check_step_record(zero_hit_record)
        self.assertEqual([item["check"] for item in issues], ["zero_witness_pass"])

        unproven_kept_record = {
            "candidate_id": "cand_unproven",
            "verdict": "fail",
            "kept": True,
            "witness": {"mod:fn:10": 3},
            "proven": False,
        }
        issues_unproven = livestream.cross_check_step_record(unproven_kept_record)
        self.assertEqual(
            [item["check"] for item in issues_unproven], ["zero_witness_pass"]
        )

    def test_unaccepted_finding_ignored_detector(self):
        record = {
            "candidate_id": "cand_unaccepted",
            "verdict": "pass",
            "kept": True,
            "witness": {"mod:fn:10": 5},
            "proven": True,
            "findings": [
                {"check": "stale_binding", "advisory": False},
                {"check": "outside_changes", "advisory": False},
            ],
            "accepted_findings": ["outside_changes"],
        }
        issues = livestream.cross_check_step_record(record)
        self.assertEqual(
            [item["check"] for item in issues], ["unaccepted_finding_ignored"]
        )
        self.assertEqual(issues[0]["unaccepted"], ["stale_binding"])

    def test_silent_tier_escalation_detector(self):
        brief = livestream.PricedBrief(
            candidate_id="cand_escalated",
            predicted_route="R0",
            expected_overhead_seconds=0.02,
            noise_floor_pct=0.4,
            cheapest_probe="live_swap_probe",
        )
        silent_record = {
            "candidate_id": "cand_escalated",
            "route": "R3",
            "verdict": "pass",
            "kept": True,
            "witness": {"mod:fn:10": 2},
            "proven": True,
        }
        issues = livestream.cross_check_step_record(silent_record, brief)
        self.assertEqual([item["check"] for item in issues], ["silent_tier_escalation"])

        explained_record = {
            **silent_record,
            "escalation_reason": "altered @triton.autotune key requires code restart",
        }
        self.assertEqual(
            livestream.cross_check_step_record(explained_record, brief), []
        )

    def test_unpaired_fork_throughput_claim_detector(self):
        unconfirmed_fork_record = {
            "candidate_id": "cand_fork",
            "route": "R2",
            "process_mode": "forked",
            "verdict": "pass",
            "kept": True,
            "delta_pct": 4.2,
            "witness": {"mod:fn:10": 8},
            "proven": True,
            "fresh_process_confirmed": False,
        }
        issues = livestream.cross_check_step_record(unconfirmed_fork_record)
        self.assertEqual(
            [item["check"] for item in issues], ["unpaired_fork_throughput_claim"]
        )

        confirmed_fork_record = {
            **unconfirmed_fork_record,
            "fresh_process_confirmed": True,
        }
        self.assertEqual(
            livestream.cross_check_step_record(confirmed_fork_record), []
        )


def _checks(record, brief=None):
    return [item["check"] for item in livestream.cross_check_step_record(record, brief)]


class CrossCheckBypassTest(unittest.TestCase):
    """Records that read as proven or accepted when the kernel would not say so."""

    def test_a_rank_or_target_with_zero_hits_is_not_a_pass(self):
        """Input: passing records whose witness is {rank0: {k: 8}, rank1: {k: 0}}, {rank0:
        {k: 8}, rank1: {}}, and {op_a: 5, op_b: 0} with proven=True. Expected: each is a
        zero_witness_pass. Pain point: hits were summed across ranks (8 + 0 > 0), an empty
        rank counted as nothing, and proven=True skipped the per-target check, so an edit one
        rank or one target never ran read as proven. Not: the kernel's own proof."""
        base = {"verdict": "pass", "kept": True}
        for witness, proven in (({"rank0": {"k": 8}, "rank1": {"k": 0}}, None),
                                ({"rank0": {"k": 8}, "rank1": {}}, None),
                                ({"op_a": 5, "op_b": 0}, True)):
            record = {**base, "witness": witness, **({"proven": proven} if proven else {})}
            self.assertEqual(_checks(record), ["zero_witness_pass"], witness)
        self.assertEqual(_checks({**base, "witness": {"rank0": {"k": 8}, "rank1": {"k": 2}}}), [])
        self.assertEqual(_checks({**base, "witness": {"workers/rank0": {"k": 8}, "core": {}}}), [])
        self.assertEqual(_checks({**base, "witness": {
            "workers/rank0": {"k": 8}, "workers/rank1": {}, "core": {}}}), ["zero_witness_pass"])

    def test_accept_all_and_not_overridable_findings(self):
        """Input: (a) accept="all" with an overridable constructor_edit finding; (b) a
        `tampered` finding (overridable False) listed in accepted_findings; (c) a finding
        `decide` already marked accepted. Expected: (a) and (c) no issue; (b)
        unaccepted_finding_ignored. Pain point: "all" was read as the letters a and l, so
        (a) was flagged; and a physical limit was accepted by listing its name. Not: accept
        records with a subject (matched by check name here)."""
        record = {"verdict": "pass", "witness": {"op": 4}, "proven": True}
        overridable = {"check": "constructor_edit", "overridable": True}
        self.assertEqual(_checks({**record, "accept": "all", "findings": [overridable]}), [])
        tampered = {"check": "tampered", "overridable": False}
        self.assertEqual(_checks({**record, "accepted_findings": ["tampered"],
                                  "findings": [tampered]}), ["unaccepted_finding_ignored"])
        self.assertEqual(_checks({**record, "findings": [{**overridable, "accepted": True}]}), [])

    def test_kernel_route_names_have_a_tier(self):
        """Input: briefs predicting "R0 swap" or "R9", and records whose route is "R9
        respawn", "R10 repair" or "T1". Expected: R0 swap -> R9 respawn is a
        silent_tier_escalation; R9 -> R10 repair is not (a repair cell runs in process, ms);
        R0 -> T1 (an in-place rebuild) is. Pain point: the kernel's own names ("R0 swap",
        "R10 repair", "T1") had no tier, so escalations were missed, and "R10" ranked above
        the warm restart R9. Not: the costs of R2-R8 (unchanged)."""
        record = {"verdict": "pass", "witness": {"op": 1}, "proven": True}

        def brief(route):
            return livestream.PricedBrief(candidate_id="c", predicted_route=route)
        self.assertEqual(_checks({**record, "route": "R9 respawn"}, brief("R0 swap")),
                         ["silent_tier_escalation"])
        self.assertEqual(_checks({**record, "route": "R10 repair"}, brief("R9")), [])
        self.assertEqual(_checks({**record, "route": "T1"}, brief("R0 swap")),
                         ["silent_tier_escalation"])


class DashboardSummaryTest(unittest.TestCase):
    """Verify `build_dashboard_summary` aggregates all 5 dashboard sections."""

    def test_build_dashboard_summary_aggregates_sections(self):
        records = [
            {
                "candidate_id": "cand_1",
                "route": "R0",
                "verdict": "pass",
                "kept": True,
                "delta_pct": 3.5,
                "overhead_seconds": 0.1,
                "witness": {"mod:fn:1": 10},
                "proven": True,
                "standing_replay": True,
            },
            {
                "candidate_id": "cand_2",
                "route": "R2",
                "process_mode": "forked",
                "verdict": "pass",
                "kept": True,
                "delta_pct": 2.0,
                "overhead_seconds": 1.5,
                "witness": {"mod:fn:2": 0},
                "proven": False,
                "fresh_process_confirmed": False,
                "standing_replay": True,
            },
        ]
        locks = [{"holder": "agent_worker_0", "resource": "gpu_0"}]
        dashboard = livestream.build_dashboard_summary(records, lock_holders=locks)

        self.assertEqual(
            set(dashboard.keys()),
            {
                "engine_status",
                "restart_cost_ladder",
                "hill_climb_ledger",
                "standing_replay_summary",
                "yield_and_spend",
            },
        )
        self.assertEqual(dashboard["engine_status"]["state"], "busy")
        self.assertEqual(dashboard["engine_status"]["lock_count"], 1)
        self.assertEqual(dashboard["engine_status"]["latest_candidate_id"], "cand_2")
        self.assertEqual(
            dashboard["restart_cost_ladder"]["total_overhead_seconds"], 1.6
        )
        self.assertEqual(
            dashboard["restart_cost_ladder"]["by_route"]["R0"]["count"], 1
        )
        self.assertEqual(dashboard["hill_climb_ledger"]["attempted_count"], 2)
        self.assertEqual(dashboard["hill_climb_ledger"]["kept_count"], 2)
        self.assertEqual(dashboard["hill_climb_ledger"]["cumulative_delta_pct"], 5.5)
        self.assertEqual(dashboard["standing_replay_summary"]["total_replays"], 2)
        self.assertEqual(dashboard["standing_replay_summary"]["passed_replays"], 1)
        self.assertEqual(dashboard["standing_replay_summary"]["failed_replays"], 1)
        self.assertEqual(dashboard["yield_and_spend"]["total_candidates"], 2)
        self.assertEqual(dashboard["yield_and_spend"]["kept_candidates"], 2)
        self.assertEqual(dashboard["yield_and_spend"]["flagged_count"], 1)

    def test_only_standing_replay_records_count_as_replays(self):
        """Input: two passing steps, one marked standing_replay. Expected: total_replays 1.
        Pain point: every step with a pass or fail verdict counted as a standing replay, so
        the replay pass rate mixed in the hill climb's own steps. Not: replay kinds other
        than `standing_replay` / `kind == "replay"`."""
        step = {"verdict": "pass", "witness": {"k": 1}, "proven": True}
        summary = livestream.build_dashboard_summary([step, {**step, "standing_replay": True}])
        self.assertEqual(summary["standing_replay_summary"]["total_replays"], 1)

    def test_interface_v1_route_names_and_ladder_conflict_detection(self):
        """Input: INTERFACE-v1 mechanism names (`drain`, `cache`, `snapshot`, `recapture`,
        `respawn`) and cross-ladder combinations between livestream (`r2`=worker_respawn,
        `r3`=code_restart, `r4`=cold_restart) and INTERFACE-v1 (`R2`=drain, `R3`=cache,
        `R4`=rebuild, `R1`=recapture). Expected: INTERFACE names resolve cleanly; `R2 drain`
        -> `R1 recapture` is detected as `silent_tier_escalation`; mixing conflicting rungs
        within a record, between brief and record, or across records in a dashboard run is
        flagged with `ladder_conflict`; compatible escalations (`worker_respawn` -> `R9 respawn`)
        do not raise false conflicts."""
        for name in ("R0 swap", "drain", "R2 drain", "cache", "R3 cache",
                     "recapture", "R1 recapture", "snapshot", "R8 snapshot",
                     "respawn", "R9 respawn", "R10 repair"):
            self.assertIsNotNone(livestream._route_tier_weight(name), name)

        clean_rec = {"verdict": "pass", "witness": {"op": 2}, "proven": True}
        brief = lambda r: livestream.PricedBrief(candidate_id="m5", predicted_route=r)

        # True INTERFACE cost escalations (drain 0.1ms -> cache 1-2ms -> recapture 4.5-7.8s)
        self.assertEqual(_checks({**clean_rec, "route": "R1 recapture"}, brief("R2 drain")),
                         ["silent_tier_escalation"])
        self.assertEqual(_checks({**clean_rec, "route": "R3 cache"}, brief("R2 drain")),
                         ["silent_tier_escalation"])
        self.assertEqual(_checks({**clean_rec, "route": "R1 recapture",
                                  "escalation_reason": "graph invalidation"}, brief("R2 drain")), [])

        # Compatible cross-ladder escalation (worker_respawn -> R9 respawn): no conflict
        self.assertEqual(_checks({**clean_rec, "route": "R9 respawn"}, brief("worker_respawn")),
                         ["silent_tier_escalation"])

        # Cross-ladder conflicts: brief vs record
        self.assertIn("ladder_conflict",
                      _checks({**clean_rec, "route": "R2"}, brief("R2 drain")))
        self.assertIn("ladder_conflict",
                      _checks({**clean_rec, "route": "R2 drain"}, brief("worker_respawn")))
        self.assertIn("ladder_conflict",
                      _checks({**clean_rec, "route": "worker_respawn"}, brief("R3 cache")))
        self.assertIn("ladder_conflict",
                      _checks({**clean_rec, "route": "R1 recapture"}, brief("R2")))

        # In-record conflicting fields (`R2` + `forked` vs `mechanism="drain"`, `R2` vs `R2 drain`)
        in_rec_1 = {**clean_rec, "route": "R2", "process_mode": "forked", "mechanism": "drain",
                    "fresh_process_confirmed": True}
        self.assertIn("ladder_conflict", _checks(in_rec_1))
        in_rec_2 = {**clean_rec, "route": "R2", "actual_route": "R2 drain"}
        self.assertIn("ladder_conflict", _checks(in_rec_2))

        # Cross-record run conflict detected by detect_ladder_conflicts & build_dashboard_summary
        run_records = [
            {**clean_rec, "candidate_id": "c1", "route": "R2", "process_mode": "forked",
             "fresh_process_confirmed": True},
            {**clean_rec, "candidate_id": "c2", "route": "R2 drain"},
        ]
        summary = livestream.build_dashboard_summary(run_records)
        self.assertEqual(len(summary["restart_cost_ladder"]["ladder_conflicts"]), 1)
        self.assertEqual(summary["restart_cost_ladder"]["ladder_conflicts"][0]["rung"], "R2")

    def test_ladder_conflict_handles_universal_aliases_brief_and_cross_record_mixing(self):
        """Universal R1 aliases (`graph_recapture`, `recompile`, `compile_reset`) resolve to
        canonical rung `r1` and conflict with bare R2/R3/R4 with order-independent rung labels;
        cross-record ladder mixing across different rungs and `rec["brief"]` are detected by
        `detect_ladder_conflicts` / `build_dashboard_summary`."""
        clean_rec = {"verdict": "pass", "witness": {"op": 2}, "proven": True}
        for alias in ("graph_recapture", "recompile", "compile_reset"):
            spec = livestream._resolve_route_spec(alias)
            self.assertEqual(spec["rung"], "r1", alias)
            c_fwd = livestream.detect_ladder_conflicts(
                [{**clean_rec, "route": alias, "predicted_route": "R2"}]
            )
            c_rev = livestream.detect_ladder_conflicts(
                [{**clean_rec, "route": "R2", "predicted_route": alias}]
            )
            self.assertEqual(len(c_fwd), 1, alias)
            self.assertEqual(len(c_rev), 1, alias)
            self.assertEqual(c_fwd[0]["rung"], "R2")
            self.assertEqual(c_rev[0]["rung"], "R2")

        # Cross-record mixing across different rungs (worker_respawn + R3 cache)
        cross_diff = livestream.build_dashboard_summary([
            {**clean_rec, "candidate_id": "c1", "route": "worker_respawn",
             "fresh_process_confirmed": True},
            {**clean_rec, "candidate_id": "c2", "route": "R3 cache"},
        ])
        self.assertEqual(len(cross_diff["restart_cost_ladder"]["ladder_conflicts"]), 1)

        # Cross-record ambiguous + interface mixing (bare R2 + R3 cache)
        cross_amb = livestream.detect_ladder_conflicts([
            {**clean_rec, "candidate_id": "c1", "route": "R2"},
            {**clean_rec, "candidate_id": "c2", "route": "R3 cache"},
        ])
        self.assertEqual(len(cross_amb), 1)

        # `brief.predicted_route` (dict and PricedBrief) inspected by detect_ladder_conflicts
        brief_dict_rec = {
            **clean_rec,
            "candidate_id": "c_dict",
            "route": "worker_respawn",
            "fresh_process_confirmed": True,
            "brief": {"candidate_id": "c_dict", "predicted_route": "R2 drain"},
        }
        brief_obj_rec = {
            **clean_rec,
            "candidate_id": "c_obj",
            "route": "worker_respawn",
            "fresh_process_confirmed": True,
            "brief": livestream.PricedBrief(candidate_id="c_obj", predicted_route="R3 cache"),
        }
        self.assertEqual(len(livestream.detect_ladder_conflicts([brief_dict_rec])), 1)
        self.assertEqual(
            len(livestream.build_dashboard_summary([brief_obj_rec])["restart_cost_ladder"]["ladder_conflicts"]),
            1,
        )


if __name__ == "__main__":
    unittest.main()
