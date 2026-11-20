"""ToyFastVideoLab: the FastVideo lab host (lab.py) on the FastVideo-shaped toy (toyfv/).

Everything but the engine is the real adapter: spawned workers, `agent.rpc` reached through
each worker's pipe with a deadline, one kernel participant per rank, the session, receipts,
fault injection in the workers, and `fresh_check` from an overlay of the package. CI runs the
conformance cases c0-c10 on it:

    cd core/livelab && PYTHONPATH=<repo>/repos/fastvideo/livelab \\
        python3 -m conformance --lab adapter.toy_lab:ToyFastVideoLab
"""
from fastrsi import harness
from adapter.lab import FastVideoLab

OPS = "adapter.toyfv.ops"
STAGES = "adapter.toyfv.stages"

_edit = harness.edit_once


class ToyFastVideoLab(FastVideoLab):
    name = "fastvideo-toy"
    package = "adapter"
    fresh_module = "adapter.toyfv.fresh"

    def __init__(self, ranks=2, receipts=None, **kw):
        kw.setdefault("rpc_timeout", 30.0)
        kw.setdefault("check_timeout", 2.0)
        super().__init__({"ranks": ranks}, receipts=receipts, modules=[OPS, STAGES], **kw)

    def make_generator(self):
        from adapter.toyfv.engine import Generator
        return Generator(self.config["ranks"])

    def request(self, workload=None):
        return dict(workload or {"seed": 1, "frames": 4})

    def output(self, result):
        return result, None, {}

    def inject(self, fault, **kw):
        if fault == "hang":
            kw.setdefault("hang_s", self.check_timeout + 1.5)    # not the base +30 s
        if fault == "tamper":
            kw.setdefault("module", OPS)
            kw.setdefault("function", "layout")
        return super().inject(fault, **kw)

    def patches(self):
        ops, st = self._base(OPS), self._base(STAGES)
        v1 = _edit(st, "(v * SCALE + 1) % 251", "(v * SCALE + 2) % 251")
        v2 = _edit(v1, "(v * SCALE + 2) % 251", "(v * SCALE + 3) % 251")
        v3 = _edit(v2, "return [v % 7 for v in", "return [v % 5 for v in")

        def p(text, module=OPS, **extra):
            return {"files": {module: text}, **extra}

        def s(text):
            return p(text, STAGES)
        return {
            "noop": s(_edit(st, "        n = layout(batch)\n", "        n = layout(batch)\n        n = n + 0\n")),
            "visible": s(v1),
            "chain": [s(v1), s(v2), s(v3)],
            "multi": {"files": {STAGES: v1, OPS: _edit(
                _edit(ops, "    return batch[\"frames\"]\n", "    return int(batch[\"frames\"])\n"),
                "    return batch[\"frames\"] + 2\n", "    return batch[\"frames\"] + 3\n")}},
            "unreached": p(_edit(ops, "return [3, 2, 1]", "return [4, 2, 1]")),
            "refused": {
                "syntax": s(_edit(st, "(v * SCALE + 1) % 251", "(v * SCALE + 1 % 251")),
                "closure": p(_edit(ops, "lambda value: isinstance(value, int)]",
                                   "lambda value: isinstance(value, int) and checks]")),
                "module_state": p(_edit(ops, "SCALE = 2", "SCALE = 3")),
                "empty": p(_edit(ops, "def layout(batch):", "def layout(batch):  # faster")),
            },
        }
