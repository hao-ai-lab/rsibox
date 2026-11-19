"""Golden tests for the real-GPU conformance lab's harness (`conformance_lab.py`), without a GPU.

Run from repo root:
    PYTHONPATH=repos/verl/livelab:repos/verl python3 -m unittest discover -s repos/verl/livelab/adapter -t repos/verl/livelab -v
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

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



if __name__ == "__main__":
    unittest.main()
