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


if __name__ == "__main__":
    unittest.main()
