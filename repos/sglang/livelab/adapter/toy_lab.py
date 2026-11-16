"""ToySglangLab: the SGLang live-lab host (`lab.py`) on the CPU `toysgl` engine (`fixture/toysgl`)."""
from adapter.fixture.harness import toy_config, toy_patches
from adapter.lab import SglangLab

FIXTURE = "adapter.fixture.toysgl.srt"
SAMPLER, LOGITS = f"{FIXTURE}.layers.sampler", f"{FIXTURE}.layers.logits_processor"
LAYERNORM, ACTIVATION, RUNNER = f"{FIXTURE}.layers.layernorm", f"{FIXTURE}.layers.activation", f"{FIXTURE}.model_executor.model_runner"
WORKLOAD = {"input_ids": [[5, 9, 2, 7, 1], [5, 9, 2, 7, 3], [5, 9, 2, 7, 8, 4], [11, 3]], "max_new_tokens": 6, "temperature": 0.0}

class ToySglangLab(SglangLab):
    name, package, fresh_mode = "sglang-toy", "adapter", "toy"
    scheduler_module, io_module = f"{FIXTURE}.managers.scheduler", f"{FIXTURE}.managers.io_struct"
    def __init__(self, tp_size=2, receipts=None, **kwargs):
        cfg = toy_config(tp_size, kwargs)
        kwargs.setdefault("rpc_timeout", 5.0)
        kwargs.setdefault("check_timeout", 1.5)
        super().__init__(cfg, (SAMPLER, LOGITS, LAYERNORM, ACTIVATION, RUNNER), receipts=receipts, **kwargs)
    def engine_class(self):
        from adapter.fixture.toysgl.srt.entrypoints.engine import Engine
        return Engine
    def default_workload(self):
        return dict(WORKLOAD)
    def run_generate(self, workload):
        merged = {**WORKLOAD, **(workload or {})}
        params = {"temperature": float(merged["temperature"]), "max_new_tokens": int(merged["max_new_tokens"])}
        out = [x["output_ids"] for x in self.engine.generate(merged["input_ids"], sampling_params=params, timeout=self.check_timeout + 1.0)]
        return {f"rank{r}": out for r in range(self.ranks)}
    def inject(self, fault, **kwargs):
        if fault == "hang": kwargs.setdefault("hang_s", self.check_timeout + 1.0)
        if fault == "tamper":
            kwargs.update({"module": kwargs.get("module", SAMPLER), "function": kwargs.get("function", "argmax")})
        return super().inject(fault, **kwargs)
    def patches(self):
        return toy_patches(self._base, SAMPLER, LOGITS, LAYERNORM, RUNNER)
