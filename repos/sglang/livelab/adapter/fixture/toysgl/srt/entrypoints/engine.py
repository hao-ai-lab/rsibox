"""Engine (sglang/srt/entrypoints/engine.py): runs in the driver process and spawns one
scheduler process per TP rank. `run_scheduler_process_func` is a class attribute so a
subclass can launch the schedulers through its own entry (SGLang keeps it for private forks).
"""
import itertools
import multiprocessing as mp
import time

from adapter.fixture.toysgl.srt.managers import scheduler as scheduler_module
from adapter.fixture.toysgl.srt.managers.io_struct import (BatchTokenIDOutput, FlushCacheReqInput,
                                                   FlushCacheReqOutput, RpcReqInput,
                                                   TokenizedGenerateReqInput)

DEFAULTS = {"tp_size": 2, "vocab_size": 64, "random_seed": 0, "max_running_requests": 8,
            "cuda_graph_max_bs": 4, "disable_cuda_graph": False,
            "disable_overlap_schedule": False, "disable_radix_cache": False}
_rids = itertools.count()


class Engine:
    run_scheduler_process_func = staticmethod(scheduler_module.run_scheduler_process)

    def __init__(self, **server_args):
        self.server_args = {**DEFAULTS, **server_args}
        tp_size = self.server_args["tp_size"]
        context = mp.get_context("spawn")
        self.send_to_scheduler, tokenizer_end = context.Pipe()
        self.recv_from_detokenizer, detokenizer_end = context.Pipe()
        self.send_to_rpc, rpc_end = context.Pipe()
        hub = {peer: context.Pipe() for peer in range(1, tp_size)}
        self.procs, readers = [], []
        for tp_rank in range(tp_size):
            reader, writer = context.Pipe(duplex=False)
            if tp_rank == 0:
                conns = {"tp": {peer: pair[0] for peer, pair in hub.items()},
                         "tokenizer": tokenizer_end, "rpc": rpc_end,
                         "detokenizer": detokenizer_end}
            else:
                conns = {"tp": {0: hub[tp_rank][1]}}
            proc = context.Process(target=type(self).run_scheduler_process_func,
                                   args=(self.server_args, tp_rank, conns, writer),
                                   daemon=True, name=f"toysgl-scheduler-tp{tp_rank}")
            proc.start()
            self.procs.append(proc)
            readers.append(reader)
        self.scheduler_info = []
        for reader in readers:
            if not reader.poll(self.server_args.get("start_timeout_s", 60)):
                self.shutdown()
                raise RuntimeError("a scheduler did not start")
            self.scheduler_info.append(reader.recv())

    def alive(self):
        return all(proc.is_alive() for proc in self.procs)

    def generate(self, input_ids, sampling_params=None, timeout=None):
        """input_ids: list of prompts (token id lists). Returns [{"output_ids": [...]}]."""
        params = dict(sampling_params or {})
        rids = []
        for prompt in input_ids:
            rid = f"r{next(_rids)}"
            rids.append(rid)
            self.send_to_scheduler.send(TokenizedGenerateReqInput(
                rid, list(prompt), params.get("max_new_tokens", 6),
                params.get("temperature", 0.0)))
        done, deadline = {}, None if timeout is None else time.monotonic() + timeout
        while len(done) < len(rids):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"no output in {timeout} s ({len(done)}/{len(rids)} done)")
            if not self.alive():
                raise RuntimeError("a scheduler process died")
            if self.recv_from_detokenizer.poll(0.01):
                out = self.recv_from_detokenizer.recv()
                if isinstance(out, BatchTokenIDOutput):
                    done[out.rid] = out.output_ids
        return [{"output_ids": done[rid]} for rid in rids]

    def flush_cache(self):
        self.send_to_scheduler.send(FlushCacheReqInput())
        while True:
            out = self.recv_from_detokenizer.recv()
            if isinstance(out, FlushCacheReqOutput):
                return out.success

    def collective_rpc(self, method, **kwargs):
        self.send_to_rpc.send(RpcReqInput(method, kwargs))
        out = self.send_to_rpc.recv()
        assert out.success, out.message

    def shutdown(self):
        for proc in self.procs:
            if proc.is_alive():
                proc.kill()
        for proc in self.procs:
            proc.join(5)
