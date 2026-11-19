"""Golden tests for the gate (`gate.py`) and the per-sample seeds (`determinism.py`), without a GPU or Ray.

Run from repo root:
    PYTHONPATH=repos/verl/livelab:repos/verl python3 -m unittest discover -s repos/verl/livelab/adapter -t repos/verl/livelab -v
"""
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

from adapter import determinism, gate


class _Cfg(dict):
    """A DictConfig stand-in: attribute access plus `.get`."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k) from None
        return _Cfg(v) if isinstance(v, dict) and not isinstance(v, _Cfg) else v

    def get(self, k, default=None):
        v = super().get(k, default)
        return _Cfg(v) if isinstance(v, dict) and not isinstance(v, _Cfg) else v


class ActorDiscoveryTest(unittest.TestCase):
    def test_actor_handles_include_agent_loop_workers_once_fit_sets_the_manager(self):
        """Input: a trainer with 2 reward handles, 2 trainer ranks and, after `fit()`, an
        `agent_loop_manager` holding 3 `AgentLoopWorkerTQ` handles. Expected: before fit, 4
        participants (reward0-1, trainer0-1); after fit, the 3 rollout workers are discovered as
        rollout0-2 between them. This is the regression for the missing-rollout bug: a reward or
        agent-loop edit committed without them would never reach the actors that run it."""
        t = types.SimpleNamespace(
            reward_loop_manager=types.SimpleNamespace(reward_loop_worker_handles=["r0", "r1"]),
            actor_rollout_wg=types.SimpleNamespace(_workers=["w0", "w1"]),
        )
        before = gate.GateMixin._actor_handles(t)
        self.assertEqual([n for n, _ in before], ["reward0", "reward1", "trainer0", "trainer1"])
        t.agent_loop_manager = types.SimpleNamespace(agent_loop_workers=["a0", "a1", "a2"])
        after = gate.GateMixin._actor_handles(t)
        self.assertEqual([n for n, _ in after], ["reward0", "reward1", "rollout0", "rollout1", "rollout2", "trainer0", "trainer1"])
        self.assertEqual(dict(after)["rollout2"], "a2")


class InjectIntoActorTest(unittest.TestCase):
    def test_inject_reaches_a_remote_participant_through_ray_call(self):
        """Input: a trainer whose `reward0` handle is a Ray actor; `__ray_call__.remote(fn, *args)`
        calls `fn(actor_instance, *args)`. Action: inject `lost` into reward0 and `fail_apply`
        into the driver. Expected: both report ok and the fault lands (`faults.STATE['lost']`);
        on 4x GB200 the first shape of this call raised TypeError inside the RewardLoopWorker."""
        from adapter import faults
        calls = []

        def ray_call(handles, fn, *args):
            calls.append(handles)
            return [fn(object(), *args)]          # the actor instance comes first
        t = types.SimpleNamespace(
            _actor_handles=lambda: [("reward0", "handle-r0")], _ray_call=ray_call, _coordinator=None)
        self.addCleanup(faults.STATE.clear)
        r = gate.GateMixin._inject(t, {"fault": "lost", "participant": "reward0"})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["participant"], "reward0")
        self.assertEqual(calls, [["handle-r0"]])
        self.assertTrue(faults.STATE.get("lost"))
        faults.STATE.clear()
        d = gate.GateMixin._inject(t, {"fault": "hang", "participant": "driver", "kwargs": {"hang_s": 1.5}})
        self.assertTrue(d["ok"] and faults.STATE.get("hang_s") == 1.5, d)
        bad = gate.GateMixin._inject(t, {"fault": "lost", "participant": "trainer9"})
        self.assertFalse(bad["ok"])


class FsdpBatchSizesTest(unittest.TestCase):
    def test_batch_sizes_on_fsdp_config_without_megatron_section(self):
        """Input: an FSDP config (no `actor.megatron` key), 4 trainer ranks, mini batch 64,
        train batch 64, n=10. Expected: per-GPU train batch 160 and infer batch 160, not an
        AttributeError. Before the fix every install request crashed on FSDP jobs."""
        cfg = _Cfg({"actor_rollout_ref": {"actor": {"strategy": "fsdp", "ppo_mini_batch_size": 64}, "rollout": {"n": 10}}, "data": {"train_batch_size": 64}})
        t = types.SimpleNamespace(config=cfg, actor_rollout_wg=types.SimpleNamespace(_workers=[0, 1, 2, 3]))
        self.assertEqual(gate.GateMixin._batch_sizes(t), {"train": 160, "infer": 160})
        cfg["actor_rollout_ref"]["actor"].update(strategy="megatron", megatron={"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 1})
        self.assertEqual(gate.GateMixin._batch_sizes(t), {"train": 320, "infer": 320})


class CoupledS0DataloaderTest(unittest.TestCase):
    def test_restore_s0_rewinds_the_dataloader_to_its_start_state(self):
        """Input: a trainer whose `train_dataloader` is a stateful loader at position 0 at
        `on_init_end`, advanced to position 3 by steps. Action: `_restore_s0`. Expected: the loader
        is loaded back to position 0, the iterator is dropped, and `reset` lists `dataloader`.
        Without it the check after a reset trains on the *next* prompt batch, so no two checks
        ever compare the same inputs."""
        class Loader:
            def __init__(self):
                self.pos = 0
            def state_dict(self):
                return {"pos": self.pos}
            def load_state_dict(self, sd):
                self.pos = sd["pos"]
        loader = Loader()
        t = types.SimpleNamespace(train_dataloader=loader, train_dataloader_it=object(), actor_rollout_wg=types.SimpleNamespace(_workers=[]), checkpoint_manager=None)
        t._s0_dataloader = gate.GateMixin._dataloader_state(t)
        loader.pos = 3
        t._ray_workers = lambda fn, *a: [{"ok": True, "digest": "d"}]
        r = gate.GateMixin._restore_s0(t)
        self.assertTrue(r["ok"], r)
        self.assertEqual(loader.pos, 0)
        self.assertIsNone(t.train_dataloader_it)
        self.assertIn("dataloader", r["reset"])
        t2 = types.SimpleNamespace(train_dataloader=None)
        self.assertIsNone(gate.GateMixin._dataloader_state(t2))


class InlineSourcesTest(unittest.TestCase):
    def test_edits_resolve_module_names_and_paths_and_refuse_unknown_keys(self):
        """Input: a request with one on-disk file and inline sources keyed by a module name
        (`json`) and by a file path. Expected: all three resolve to real paths with the right
        text; an unknown key raises `ValueError` (the gate reports `patch_invalid`)."""
        tmp = tempfile.mkdtemp(prefix="fastrsi-gate-edits-")
        self.addCleanup(shutil.rmtree, tmp, True)
        f1, f2 = Path(tmp) / "a.py", Path(tmp) / "b.py"
        f1.write_text("A = 1\n", encoding="utf-8")
        f2.write_text("B = 1\n", encoding="utf-8")
        edits = gate.GateMixin._edits(None, {"files": [str(f1)], "sources": {str(f2): "B = 2\n", "json": "# json\n"}})
        self.assertEqual(edits[os.path.realpath(str(f1))], "A = 1\n")
        self.assertEqual(edits[os.path.realpath(str(f2))], "B = 2\n")
        self.assertEqual(edits[gate.path_for_key("json")], "# json\n")
        self.assertTrue(gate.path_for_key("json").endswith("json/__init__.py"))
        with self.assertRaises(ValueError):
            gate.GateMixin._edits(None, {"sources": {"no.such.module.here": "x"}})

    def test_tracked_modules_default_adds_the_reward_file(self):
        cfg = _Cfg({"reward": {"custom_reward_function": {"path": __file__}}})
        self.assertEqual(gate.tracked_modules(cfg, env={}), ["verl.trainer.ppo.core_algos", os.path.realpath(__file__)])
        self.assertEqual(gate.tracked_modules(cfg, env={"FASTRSI_RL_TRACK": os.pathsep.join(["a.b", "c"])}), ["a.b", "c"])


class CrossHostGateClientTest(unittest.TestCase):
    def test_dead_pid_on_another_host_is_not_a_dead_job(self):
        """Input: gate.json records a PID that is dead here and `host` = another machine (the
        trainer runs on a compute node, the client on a login node sharing the gate dir).
        Expected: `_job_died` is False; with `host` = this machine it is True. Before the fix
        every cross-host request failed at once with "job process died"."""
        from tools import cli
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        tmp = Path(tempfile.mkdtemp(prefix="fastrsi-gate-host-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "gate.json").write_text(json.dumps({"pid": dead.pid, "host": "some-other-node"}), encoding="utf-8")
        self.assertFalse(cli._job_died(tmp))
        (tmp / "gate.json").write_text(json.dumps({"pid": dead.pid, "host": socket.gethostname()}), encoding="utf-8")
        self.assertTrue(cli._job_died(tmp))


class PerSampleSeedTest(unittest.TestCase):
    def test_seed_agent_loops_gives_each_session_its_own_reproducible_seed(self):
        """Input: an agent-loop worker class whose `_run_agent_loop` records its sampling params,
        sessions 0..2 of sample 7 and session 0 of sample 8. Expected: seeds 42+7000+{0,1,2} and
        42+8000, identical on a second pass (so a check after S0 reset resamples the same text),
        the original params untouched, and a second call is a no-op."""
        seen = []

        class Worker:
            async def _run_agent_loop(self, sampling_params, trajectory, *, session_id=0, **kw):
                seen.append(sampling_params.get("seed"))
                return sampling_params

        host = Worker()
        r = determinism.seed_agent_loops(host, 42)
        self.assertTrue(r["ok"] and not r.get("already"))
        params = {"temperature": 1.0}
        for sid in range(3):
            asyncio.run(host._run_agent_loop(params, {"sample_index": 7}, session_id=sid))
        asyncio.run(host._run_agent_loop(params, {"sample_index": 8}, session_id=0))
        self.assertEqual(seen, [7042, 7043, 7044, 8042])
        self.assertEqual(params, {"temperature": 1.0})
        seen.clear()
        asyncio.run(host._run_agent_loop(params, {"sample_index": 7}, session_id=2))
        self.assertEqual(seen, [7044])
        self.assertTrue(determinism.seed_agent_loops(host, 42).get("already"))


if __name__ == "__main__":
    unittest.main()


class WitnessUninvolvedRolesTest(unittest.TestCase):
    def test_flush_leaves_an_empty_entry_for_roles_that_never_ran_the_edit(self):
        """Input: a proof whose witness names the driver with 1 call of the GRPO advantage
        function and eight rollout actors plus four trainers with 0 calls (they import
        `core_algos`, so the swap reached them, but only the driver computes advantages).
        Action: `_flush`. Expected: the response keeps `{}` for every silent role and the
        driver's count, forwards the kernel's `abstained` list, and stays proven. The
        conformance c2 case reads an empty entry as an uninvolved role; a `{key: 0}` entry
        made the real 4-GPU run fail c2 with 'witness > 0 on every participant'."""
        key = "verl.trainer.ppo.core_algos:compute_grpo_outcome_advantage:267"
        witness = {"driver": {key: 1}, **{f"rollout{i}": {key: 0} for i in range(8)}, **{f"trainer{i}": {key: 0} for i in range(4)}}
        findings = [{"check": "zero_witness", "subject": key, "participant": f"rollout{i}"} for i in range(8)]
        proof = {"receipt": {"witness": witness, "findings": findings, "abstained": ["reward0"]}}
        tmp = Path(tempfile.mkdtemp())
        g = types.SimpleNamespace(_pending={"n": 7, "steps_run": 1, "t_seen": 0.0}, _gate=tmp,
                                  _session=types.SimpleNamespace(stack=["txn-1"]),
                                  _coordinator=types.SimpleNamespace(prove=lambda tid: proof))
        gate.GateMixin._flush(g)
        resp = json.loads((tmp / "resp-7.json").read_text())
        self.assertEqual(resp["witness"]["driver"], {key: 1})
        self.assertEqual(resp["witness"]["rollout3"], {})
        self.assertEqual(resp["witness"]["trainer0"], {})
        self.assertEqual(resp["abstained"], ["reward0"])
        self.assertTrue(resp["proven"], resp)
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "core" / "livelab"))
        from conformance import cases
        involved = cases._involved_witnesses(resp["witness"], resp.get("abstained"))
        self.assertEqual(set(involved), {"driver"})
        # a silent sibling rank of an involved role is still a failure
        resp["witness"]["rollout0"] = {key: 2}
        self.assertIn("rollout3", cases._involved_witnesses(resp["witness"]))
