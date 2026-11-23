"""MilesLab: `LabUnderTest` implementation over `rsibox.lab.BaseLab` and `MilesStepGate`."""
import json
import os

import adapter  # noqa: F401
from adapter import gate as G
from rsibox.lab import BaseLab


class MilesLab(BaseLab):
    """Live lab host for `miles` (`radixark/miles` / `THUDM/slime`). Never raises from a verb."""

    name = "miles"

    def __init__(self, receipts=None, mode="sync", rpc_timeout=30.0, check_timeout=60.0, **kwargs):
        self.mode = mode
        self.options = dict(kwargs)
        self.runtime = {}
        self.data_buffer = None
        self.coupled_reset = None
        self.gate = None
        self.coordinator = None
        super().__init__(receipts=receipts, check_timeout=check_timeout, rpc_timeout=rpc_timeout)

    def make_runtime(self):
        if callable(launcher := self.options.get("runtime_factory")):
            return launcher()
        raise RuntimeError("MilesLab requires runtime_factory or ToyMilesLab for CPU conformance")

    def run_step(self, workload=None):
        return self.runtime["run_step"](workload)

    def _start_engine(self):
        rt = self.runtime = self.make_runtime()
        self.base_text = dict(rt["base_text"])
        self.data_buffer = rt.get("data_buffer")
        self.coupled_reset = rt.get("coupled_reset")
        self.gate = G.MilesStepGate(
            rt["rollout_actors"],
            rt["train_actors"],
            extra_actors=rt.get("extra_actors"),
            get=rt.get("get"),
            timeout_s=self.rpc_timeout,
            mode=self.mode,
            rollout_controller=rt.get("rollout_controller"),
            data_buffer=self.data_buffer,
            at_gap_fn=rt.get("at_gap"),
            writer=self.writer,
            engine_meta={
                "name": "miles",
                "version": "0.2.0",
                "commit": os.environ.get("RSIBOX_ENGINE_COMMIT"),
            },
        )
        self.coordinator = self.gate.coordinator
        return self.coordinator, list(rt["modules"])

    def _stop_engine(self):
        if callable(shutdown := getattr(self, "runtime", {}).get("shutdown")):
            shutdown()

    def _build_plan(self, patch, live_map):
        return self.gate.build_plan(patch, live_map)

    def _after_commit(self, patch, receipt):
        if self.data_buffer is not None:
            self.data_buffer.advance_epoch()

    def _after_uninstall(self, undone):
        if self.data_buffer is not None:
            self.data_buffer.advance_epoch()

    def _run_workload(self, workload):
        def step_job():
            if self.coupled_reset is not None:
                self.coupled_reset.restore_and_sync()
            return self.run_step(workload)

        return json.loads(json.dumps(self.gate.at_gap(step_job, timeout=self.check_timeout)))

    def _clean_witness(self, raw_witness, proof_ok):
        return G.clean_witness(raw_witness, proof_ok)

    def _reset_state(self):
        return self.gate.at_gap(
            lambda: self.coupled_reset.restore_and_sync() if self.coupled_reset else {"reset": ["state"]},
            timeout=self.check_timeout,
        )
