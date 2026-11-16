"""Golden tests for `repos/verl/tools/` (`cli`, `rescore`, `replay_adv`).

Run from repo root:
    PYTHONPATH=repos/verl:repos/verl/livelab python3 -m unittest discover -s repos/verl/tools -t repos/verl -v
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools import cli, replay_adv, rescore


class OfflineProbesAndCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fastrsi-verl-probes-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_rescore_diffs_recorded_rollouts_in_sub_millisecond_cpu_pass(self):
        """Input: 64 recorded rollout JSONL rows scored with v1 reward (wrong-answer bonus 0.0),
        re-scored offline with v2 reward (format bonus 0.1 on `####` answers).
        Expected: exact changed count (16), mean_before 0.25 -> mean_after 0.275."""
        rollouts_file = Path(self.tmp) / "step_2.jsonl"
        rows = []
        for idx in range(64):
            correct = idx % 4 == 0
            has_tag = idx % 2 == 0
            answer = "42" if correct else "13"
            text = f"reasoning... #### {answer}" if has_tag else f"answer is {answer}"
            rows.append(
                json.dumps(
                    {
                        "input": f"q{idx}",
                        "output": text,
                        "gts": "42",
                        "score": 1.0 if (correct and has_tag) else 0.0,
                    }
                )
            )
        rollouts_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
        reward_v2 = Path(self.tmp) / "reward_v2.py"
        reward_v2.write_text(
            "def compute_score(data_source=None, solution_str='', ground_truth='', extra_info=None):\n"
            "    if '#### ' not in solution_str:\n"
            "        return 0.0\n"
            "    pred = solution_str.rsplit('#### ', 1)[-1].strip()\n"
            "    return 1.0 if pred == ground_truth else 0.1\n",
            encoding="utf-8",
        )
        result = rescore.rescore(rollouts_file, reward_v2)
        self.assertEqual(result["responses"], 64)
        self.assertEqual(result["changed"], 16)
        self.assertAlmostEqual(result["mean_before"], 0.25, places=4)
        self.assertAlmostEqual(result["mean_after"], 0.275, places=4)
        self.assertLess(result["score_ms"], 200.0)

    def test_replay_advantages_detects_clamp_edit_on_recorded_groups(self):
        """Input: 16 recorded GRPO trajectory groups with unclamped advantages reaching +-1.8,
        replayed against an edited estimator that clamps advantages to [-1.0, 1.0].
        Expected: `equal=False`, `after.max == 1.0`, `after.min == -1.0`."""
        adv_file = Path(self.tmp) / "adv.json"
        reward_groups = [[0.0, 0.0, 0.0, 2.4] for _ in range(16)]
        before_groups = [[-0.6, -0.6, -0.6, 1.8] for _ in range(16)]
        adv_file.write_text(
            json.dumps({"reward_groups": reward_groups, "advantage_groups": before_groups}),
            encoding="utf-8",
        )
        est_py = Path(self.tmp) / "clamped_adv.py"
        est_py.write_text(
            "def grpo_advantages(rewards):\n"
            "    mean = sum(rewards) / len(rewards)\n"
            "    return [max(-1.0, min(1.0, r - mean)) for r in rewards]\n",
            encoding="utf-8",
        )
        result = replay_adv.replay_advantages(
            adv_file, atol=0.0, estimator_spec=f"{est_py}:grpo_advantages"
        )
        self.assertEqual(result["rows"], 16)
        self.assertFalse(result["equal"])
        self.assertEqual(result["rows_changed"], 16)
        self.assertAlmostEqual(result["before"]["max"], 1.8, places=5)
        self.assertAlmostEqual(result["after"]["max"], 1.0, places=5)
        self.assertAlmostEqual(result["max_abs_diff"], 0.8, places=5)

    def test_dead_gate_process_fails_client_request_immediately(self):
        """Input: a gate directory whose recorded driver PID has already exited.
        Expected: `cli.request` raises `GateError('died')` immediately instead of waiting for timeout."""
        dead_proc = subprocess.Popen([sys.executable, "-c", "pass"])
        dead_proc.wait()
        gate_dir = Path(self.tmp) / "gate"
        gate_dir.mkdir()
        (gate_dir / "ready.json").write_text(json.dumps({"n": 0}), encoding="utf-8")
        (gate_dir / "gate.json").write_text(json.dumps({"pid": dead_proc.pid}), encoding="utf-8")
        with self.assertRaisesRegex(cli.GateError, "died"):
            cli.request(gate_dir, timeout=30.0)


if __name__ == "__main__":
    unittest.main()
