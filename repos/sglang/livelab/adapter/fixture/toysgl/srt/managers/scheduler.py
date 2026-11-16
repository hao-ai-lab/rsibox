"""The scheduler of one TP rank (sglang/srt/managers/scheduler.py), same loop structure.

    while True:
        recv_reqs = self.recv_requests()        # rank 0 reads, every rank gets the same list
        self.process_input_requests(recv_reqs)  # control messages run here, between batches
        batch = self.get_next_batch_to_run()
        result = self.run_batch(batch)          # forward + all-gather + sample, on every rank
        self.process_batch_result(batch, result)

`event_loop_overlap` processes each batch's result one iteration later (`result_queue`).
"""
import sys
import time
from collections import deque

from adapter.fixture.toysgl.srt.managers.io_struct import (BatchTokenIDOutput,
                                                   ContinueGenerationReqInput,
                                                   FlushCacheReqInput, FlushCacheReqOutput,
                                                   PauseGenerationReqInput, RpcReqInput,
                                                   RpcReqOutput, TokenizedGenerateReqInput)
from adapter.fixture.toysgl.srt.managers.tp_worker import TpModelWorker
from adapter.fixture.toysgl.srt.mem_cache.memory_pool import RadixCache


class Req:
    def __init__(self, recv_req):
        self.rid = recv_req.rid
        self.origin_input_ids = list(recv_req.input_ids)
        self.max_new_tokens = recv_req.max_new_tokens
        self.temperature = recv_req.temperature
        self.output_ids = []
        self.kv, self.states, self.kv_sum = [], [], 0
        self.req_pool_idx = self.mamba_slot = None
        self.finished_sent = False

    def finished(self):
        return len(self.output_ids) >= self.max_new_tokens


class ScheduleBatch:
    def __init__(self, reqs, forward_mode):
        self.reqs, self.forward_mode = reqs, forward_mode

    def is_extend(self):
        return self.forward_mode == "extend"

    def copy(self):
        return ScheduleBatch(list(self.reqs), self.forward_mode)


class Scheduler:
    def __init__(self, server_args, tp_rank, tp_group, recv_from_tokenizer=None,
                 recv_from_rpc=None, send_to_detokenizer=None):
        self.server_args = server_args
        self.tp_rank, self.tp_group, self.tp_size = tp_rank, tp_group, tp_group.size
        self.enable_overlap = not server_args.get("disable_overlap_schedule", False)
        self.recv_from_tokenizer = recv_from_tokenizer
        self.recv_from_rpc = recv_from_rpc
        self.send_to_detokenizer = send_to_detokenizer
        self.idle_sleep_s = server_args.get("idle_sleep_s", 0.0005)
        self.tp_worker = TpModelWorker(server_args, tp_rank, tp_group)
        self.init_memory_pool_and_cache()
        self.waiting_queue, self.running_reqs = [], []
        self.cur_batch = self.last_batch = None
        self.result_queue = deque()
        self._engine_paused = False
        self.forward_ct = 0
        self.init_request_dispatcher()

    def init_memory_pool_and_cache(self):
        runner = self.tp_worker.model_runner
        self.req_to_token_pool = runner.req_to_token_pool
        self.token_to_kv_pool_allocator = runner.token_to_kv_pool_allocator
        self.mamba_pool = runner.mamba_pool
        self.tree_cache = RadixCache(disable=self.server_args.get("disable_radix_cache", False))

    def init_request_dispatcher(self):
        self._request_dispatcher = {
            TokenizedGenerateReqInput: self.handle_generate_request,
            FlushCacheReqInput: self.flush_cache_wrapped,
            PauseGenerationReqInput: self.pause_generation,
            ContinueGenerationReqInput: self.continue_generation,
            RpcReqInput: self.handle_rpc_request,
        }

    # -- loops ---------------------------------------------------------------------------
    def event_loop_normal(self):
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.on_idle()
            self.last_batch = batch

    def event_loop_overlap(self):
        self.result_queue = deque()

        def pop_and_process():
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            if self.last_batch:
                pop_and_process()
            elif batch is None:
                self.on_idle()
            self.last_batch = batch

    def recv_requests(self):
        """Receive at tp_rank 0 and broadcast to every other TP rank."""
        if self.tp_rank == 0:
            recv_reqs = []
            for conn in (self.recv_from_tokenizer, self.recv_from_rpc):
                while conn is not None and conn.poll():
                    recv_reqs.append(conn.recv())
        else:
            recv_reqs = None
        return self.tp_group.broadcast_pyobj(recv_reqs)

    def process_input_requests(self, recv_reqs):
        for recv_req in recv_reqs:
            output = self._request_dispatcher[type(recv_req)](recv_req)
            if output is not None and self.tp_rank == 0:
                if isinstance(output, RpcReqOutput):
                    self.recv_from_rpc.send(output)
                else:
                    self.send_to_detokenizer.send(output)

    def on_idle(self):
        time.sleep(self.idle_sleep_s)

    # -- requests and batches ------------------------------------------------------------
    def handle_generate_request(self, recv_req):
        self.waiting_queue.append(Req(recv_req))

    def get_next_batch_to_run(self):
        if self.last_batch is not None and self.last_batch.is_extend():
            self.running_reqs += [req for req in self.last_batch.reqs if req not in self.running_reqs]
        self.running_reqs = [req for req in self.running_reqs if not req.finished()]
        admitted = []
        while self.waiting_queue and self.req_to_token_pool.available_size():
            req = self.waiting_queue.pop(0)
            req.req_pool_idx = self.req_to_token_pool.alloc()
            req.mamba_slot = self.mamba_pool.alloc()
            req.kv, req.states = self.tree_cache.match_prefix(req.origin_input_ids)
            req.kv_sum = sum(req.kv)
            self.token_to_kv_pool_allocator.alloc(len(req.origin_input_ids) + req.max_new_tokens)
            admitted.append(req)
        if admitted:
            return ScheduleBatch(admitted, "extend")
        if self.running_reqs:
            return ScheduleBatch(list(self.running_reqs), "decode")
        return None

    def run_batch(self, batch):
        self.forward_ct += 1
        next_token_ids = self.tp_worker.forward_batch_generation(batch)
        for req, token in zip(batch.reqs, next_token_ids):
            req.output_ids.append(token)
        return next_token_ids

    def process_batch_result(self, batch, result):
        for req in batch.reqs:
            if req.finished() and not req.finished_sent:
                self.finish_req(req)

    def finish_req(self, req):
        req.finished_sent = True
        tokens = req.origin_input_ids + req.output_ids[:-1]
        self.tree_cache.insert(tokens, req.kv, req.states)
        self.req_to_token_pool.free(req.req_pool_idx)
        self.mamba_pool.free(req.mamba_slot)
        self.token_to_kv_pool_allocator.free(len(req.origin_input_ids) + req.max_new_tokens)
        if self.tp_rank == 0:
            self.send_to_detokenizer.send(BatchTokenIDOutput(req.rid, list(req.output_ids)))

    # -- control -------------------------------------------------------------------------
    def is_fully_idle(self):
        return not self.waiting_queue and not self.running_reqs and not self.result_queue

    def flush_cache(self):
        if self.is_fully_idle():
            self.cur_batch = self.last_batch = None
            self.tree_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()
            self.mamba_pool.clear()
            return True
        return False

    def flush_cache_wrapped(self, recv_req):
        return FlushCacheReqOutput(self.flush_cache())

    def handle_rpc_request(self, recv_req):
        success, error = True, None
        try:
            func = getattr(self, recv_req.method)
            func(**(recv_req.parameters or {}))
        except Exception as e:
            success, error = False, e
            print(f"rpc {recv_req.method} failed: {e}", file=sys.stderr)
        self.tp_group.barrier()
        return RpcReqOutput(success, "" if error is None else str(error))

    def pause_generation(self, recv_req):
        self._engine_paused = True
        if recv_req.mode == "in_place":
            return
        if self.enable_overlap and self.last_batch:
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)
        if self.last_batch and self.last_batch.is_extend():
            self.running_reqs += [req for req in self.last_batch.reqs
                                  if req not in self.running_reqs and not req.finished()]
        self.last_batch = None
        self.cur_batch = None

    def continue_generation(self, recv_req):
        self._engine_paused = False


def run_scheduler_process(server_args, tp_rank, conns, ready):
    """The scheduler process: build the scheduler, report ready, run the event loop."""
    from adapter.fixture.toysgl.srt.distributed import TpGroup
    try:
        group = TpGroup(tp_rank, server_args.get("tp_size", 1), conns["tp"])
        scheduler = Scheduler(server_args, tp_rank, group, conns.get("tokenizer"),
                              conns.get("rpc"), conns.get("detokenizer"))
        ready.send({"status": "ready", "tp_rank": tp_rank})
        if scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    except BaseException as e:
        print(f"scheduler tp_rank {tp_rank} died: {type(e).__name__}: {e}", file=sys.stderr)
        sys.stderr.flush()
    import os
    os._exit(1)
