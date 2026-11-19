"""Golden tests for the real-GPU conformance lab's harness (`conformance_lab.py`), without a GPU.

Run from repo root:
    PYTHONPATH=repos/verl/livelab:repos/verl python3 -m unittest discover -s repos/verl/livelab/adapter -t repos/verl/livelab -v
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from adapter import conformance_lab as CL


FAKE_CORE = '''def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, epsilon=1e-6, norm_adv_by_std_in_grpo=True, config=None):
    scores = token_level_rewards.sum(dim=-1)

    id2score = {}
    with torch.no_grad():
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    return scores, scores


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    return loss_mat


def other(
    loss_scale_factor: Optional[int] = None,
):
    return None


def kl_penalty(logprob, ref_logprob, kl_penalty):
    # Strip the optional '+' suffix so e.g. "k3+" dispatches to "k3".
    return logprob


def compute_other_outcome_advantage(token_level_rewards, response_mask, index):
    scores = token_level_rewards.sum(dim=-1)

    id2score = {}
    return scores, scores
'''


class ConformanceHarnessTest(unittest.TestCase):
    def test_patches_apply_to_the_shipped_reward_and_core_algos_anchors(self):
        """Input: the shipped `examples/grpo-gsm8k/reward_gsm8k.py` and a `core_algos` text with
        the three anchors the harness edits. Expected: every patch class is produced, each edit
        changes exactly its target, the chain's third step edits the reward file, and `multi`
        touches 3 functions in `core_algos` plus the reward (>= 3 functions in >= 2 modules)."""
        reward = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "examples", "grpo-gsm8k", "reward_gsm8k.py"))
        lab = CL.VerlConformanceLab.__new__(CL.VerlConformanceLab)
        lab.reward = reward
        lab._base = lambda key: FAKE_CORE if key == CL.CORE else Path(reward).read_text(encoding="utf-8")
        P = lab.patches()
        self.assertEqual(set(P), {"noop", "visible", "chain", "multi", "unreached", "refused"})
        self.assertIn("+ epsilon) * 1.0", P["noop"]["files"][CL.CORE])
        self.assertIn("scores[i] = = (", P["refused"]["syntax"]["files"][CL.CORE])
        self.assertIn("+ epsilon) * 2.0", P["visible"]["files"][CL.CORE])
        self.assertIn("+ epsilon) * 3.0", P["chain"][1]["files"][CL.CORE])
        self.assertIn("return SCORE * 2 if", P["chain"][2]["files"][reward])
        multi = P["multi"]["files"][CL.CORE]
        self.assertTrue(all(s in multi for s in ("* 2.0", "loss_agg_mode = str(loss_agg_mode)", "kl_penalty = str(kl_penalty)")))
        self.assertIn("return 0.5\n", P["unreached"]["files"][reward])
        self.assertEqual(set(P["refused"]), {"syntax", "closure", "module_state", "empty"})
        self.assertIn("SCORE = 2.0\n", P["refused"]["module_state"]["files"][reward])

    def test_output_projection_ignores_uids_and_row_order(self):
        """Input: two rollout dumps with the same (prompt, response, score, gt) rows in different
        order and different random uids. Expected: identical digests, and a changed score changes
        the digest; the step output carries the four metrics by their short names."""
        tmp = Path(tempfile.mkdtemp(prefix="fastrsi-rollouts-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        rows = [{"input": "q1", "output": "#### 4", "score": 1.0, "gts": "4", "uid": "a1"}, {"input": "q2", "output": "no", "score": 0.0, "gts": "7", "uid": "b2"}]
        (tmp / "1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        (tmp / "2.jsonl").write_text("\n".join(json.dumps({**r, "uid": "zz"}) for r in reversed(rows)) + "\n", encoding="utf-8")
        (tmp / "3.jsonl").write_text(json.dumps({**rows[0], "score": 2.0}) + "\n" + json.dumps(rows[1]) + "\n", encoding="utf-8")
        d1, d2, d3 = (CL._digest_rollouts(str(tmp / f"{i}.jsonl")) for i in (1, 2, 3))
        self.assertEqual(d1, d2)
        self.assertNotEqual(d1, d3)
        lab = CL.VerlConformanceLab.__new__(CL.VerlConformanceLab)
        out = lab._output({"steps_run": [{"metrics": {"critic/score/mean": 0.5, "actor/pg_loss": 0.008408406749367714}, "rollouts": str(tmp / "1.jsonl")}]})
        self.assertEqual(out, {"steps": [{"score_mean": 0.5, "adv_max": None, "adv_min": None, "pg_loss": 0.008408, "rollouts": d1}]})
        # the two pg_loss values a fresh and a live process gave for the same step on 4x GB200 compare equal
        live = lab._output({"steps_run": [{"metrics": {"critic/score/mean": 0.5, "actor/pg_loss": 0.008408404886722565}, "rollouts": str(tmp / "1.jsonl")}]})
        self.assertEqual(live, out)

    def test_launch_environment_maps_all_flags_and_strips_leading_separator(self):
        tmp = Path(tempfile.mkdtemp(prefix="fastrsi-verl-cli-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        gate_dir = tmp / "gate"
        reward_py = tmp / "reward.py"
        reward_py.write_text("def compute_score(**kw): return 1.0\n", encoding="utf-8")
        launch_cmd = "python3 -m tools.cli train --gate $GATE -- actor.x=1 reward.path=$REWARD"

        env = CL.VerlConformanceLab.launch_environment([
            "--",
            "--gate", str(gate_dir),
            "--reward", str(reward_py),
            "--launch", launch_cmd,
            "--check-timeout", "420",
            "--boot-timeout", "2400",
        ])
        self.assertEqual(env["FASTRSI_VERL_GATE"], str(gate_dir.resolve()))
        self.assertEqual(env["FASTRSI_VERL_REWARD"], os.path.realpath(str(reward_py)))
        self.assertEqual(env["FASTRSI_VERL_LAUNCH"], launch_cmd)
        self.assertEqual(env["FASTRSI_VERL_CHECK_TIMEOUT"], "420.0")
        self.assertEqual(env["FASTRSI_VERL_BOOT_TIMEOUT"], "2400.0")

        minimal = CL.VerlConformanceLab.launch_environment(["--gate", str(gate_dir), "--reward", str(reward_py)])
        self.assertEqual(set(minimal), {"FASTRSI_VERL_GATE", "FASTRSI_VERL_REWARD"})

    def test_gpus_busy_scopes_nvidia_smi_to_cuda_visible_devices(self):
        lab = CL.VerlConformanceLab.__new__(CL.VerlConformanceLab)
        seen_cmds = []

        def fake_run(cmd, **_kw):
            seen_cmds.append(cmd)
            return types.SimpleNamespace(stdout="1234\n")

        with mock.patch.object(CL.subprocess, "run", side_effect=fake_run):
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,2"}, clear=False):
                self.assertEqual(lab._gpus_busy(), ["1234"])
            self.assertEqual(
                seen_cmds[-1],
                ["nvidia-smi", "-i", "0,2", "--query-compute-apps=pid", "--format=csv,noheader"],
            )

            seen_cmds.clear()
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
                self.assertEqual(lab._gpus_busy(), [])
            self.assertEqual(seen_cmds, [])

            clean_env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
            with mock.patch.dict(os.environ, clean_env, clear=True):
                self.assertEqual(lab._gpus_busy(), ["1234"])
            self.assertEqual(
                seen_cmds[-1],
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            )

    def test_overlay_handles_package_init_and_reward_and_reap_clears_jobs(self):
        tmp = Path(tempfile.mkdtemp(prefix="fastrsi-verl-ov-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        pkg_root = tmp / "site"
        pkg = pkg_root / "verlovpkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("TAG = 'base'\n", encoding="utf-8")
        (pkg / "leaf.py").write_text("VAL = 1\n", encoding="utf-8")
        reward = tmp / "reward.py"
        reward.write_text("SCORE = 1.0\n", encoding="utf-8")
        other = tmp / "other.py"
        other.write_text("X = 1\n", encoding="utf-8")

        sys.path.insert(0, str(pkg_root))
        self.addCleanup(sys.path.remove, str(pkg_root))
        self.addCleanup(lambda: [sys.modules.pop(k, None) for k in list(sys.modules) if k == "verlovpkg" or k.startswith("verlovpkg.")])

        lab = CL.VerlConformanceLab.__new__(CL.VerlConformanceLab)
        lab.reward = os.path.realpath(str(reward))
        ov_tmp = tmp / "out"
        ov_tmp.mkdir()
        overlay, rew_out = lab._overlay(
            str(ov_tmp),
            {
                "verlovpkg": "TAG = 'patched'\n",
                "verlovpkg.leaf": "VAL = 2\n",
                str(reward): "SCORE = 2.0\n",
            },
        )
        self.assertEqual(Path(overlay, "verlovpkg", "__init__.py").read_text(encoding="utf-8"), "TAG = 'patched'\n")
        self.assertEqual(Path(overlay, "verlovpkg", "leaf.py").read_text(encoding="utf-8"), "VAL = 2\n")
        self.assertEqual(Path(rew_out).read_text(encoding="utf-8"), "SCORE = 2.0\n")
        self.assertEqual((pkg / "__init__.py").read_text(encoding="utf-8"), "TAG = 'base'\n")
        with self.assertRaises(ValueError):
            lab._overlay(str(ov_tmp), {str(other): "X = 2\n"})

        # _reap_job waits and pops finished Popen entries from _JOBS cleanly (no ResourceWarning).
        gate = tmp / "gate"
        gate.mkdir()
        CL._JOBS[str(gate)] = subprocess.Popen([sys.executable, "-c", "pass"])
        lab._reap_job(gate)
        self.assertNotIn(str(gate), CL._JOBS)

    def test_files_rejects_mixed_files_and_diff_and_supports_lazy_diffs(self):
        lab = CL.VerlConformanceLab.__new__(CL.VerlConformanceLab)
        lab._live_stack = [{"pkg.m": "X = 1\n"}]
        lab._disk_source = lambda m: "X = 0\n" if m == "pkg.m" else None
        self.assertEqual(lab.files({"files": {"pkg.m": "X = 2\n"}}), {"pkg.m": "X = 2\n"})
        with self.assertRaisesRegex(ValueError, "both"):
            lab.files({"files": {"pkg.m": "X = 2\n"}, "diff": "--- a/pkg/m.py\n+++ b/pkg/m.py\n"})

        fake_diffs = types.SimpleNamespace(
            apply=lambda d, cur: {"pkg.m": cur("pkg.m") + "# " + d}
        )
        import fastrsi
        with mock.patch.dict(sys.modules, {"fastrsi.diffs": fake_diffs}), \
             mock.patch.object(fastrsi, "diffs", fake_diffs, create=True):
            self.assertEqual(lab.files({"diff": "live"}, against="live"), {"pkg.m": "X = 1\n# live"})
            self.assertEqual(lab.files({"diff": "disk"}, against="disk"), {"pkg.m": "X = 0\n# disk"})


if __name__ == "__main__":
    unittest.main()
