"""The toy's executor: N spawned workers, each in FastVideo's worker busy loop.

Mirrors the pin's `MultiprocExecutor` protocol: the driver sends `{"method", "args", "kwargs"}`
on each worker's pipe; `execute_forward` runs the pipeline; a callable runs as
`method(worker, *args, **kwargs)`; an exception comes back as `{RPC_ERROR: True, "error"}` and the
worker keeps serving; `shutdown` ends the loop.
"""
import multiprocessing as mp
import types

RPC_ERROR = "__fastvideo_rpc_error__"


class Worker:
    def __init__(self, rank):
        from adapter.toyfv import ops
        self.rank = rank
        self.pipeline = ops.Pipeline()


def busy_loop(pipe, rank):
    w = Worker(rank)
    pipe.send("ready")
    while True:
        try:
            call = pipe.recv()
        except EOFError:
            break
        m = call.get("method")
        try:
            if m == "shutdown":
                pipe.send({"ok": True})
                break
            if m == "execute_forward":
                pipe.send(w.pipeline.forward(dict(call["kwargs"]["batch"])))
            else:
                pipe.send(m(w, *call.get("args", ()), **call.get("kwargs", {})))
        except Exception as e:
            pipe.send({RPC_ERROR: True, "error": f"{type(e).__name__}: {e}"})


class Generator:
    """`VideoGenerator`-shaped: `.executor.workers[i].pipe`, `generate`, `shutdown`."""

    def __init__(self, ranks=2):
        ctx = mp.get_context("spawn")
        workers = []
        for r in range(ranks):
            a, b = ctx.Pipe()
            p = ctx.Process(target=busy_loop, args=(b, r), daemon=True)
            p.start()
            workers.append(types.SimpleNamespace(proc=p, pipe=a, rank=r))
        for w in workers:
            assert w.pipe.recv() == "ready"
        self.executor = types.SimpleNamespace(workers=workers)

    def generate(self, batch):
        for w in self.executor.workers:
            w.pipe.send({"method": "execute_forward", "args": (), "kwargs": {"batch": batch}})
        out = [w.pipe.recv() for w in self.executor.workers]
        errs = [o["error"] for o in out if isinstance(o, dict) and o.get(RPC_ERROR)]
        if errs:
            raise RuntimeError("; ".join(errs))
        return out[0]

    def shutdown(self):
        for w in self.executor.workers:
            try:
                w.pipe.send({"method": "shutdown"})
            except Exception:
                pass
        for w in self.executor.workers:
            w.proc.join(2)
            if w.proc.is_alive():
                w.proc.kill()
                w.proc.join(2)
