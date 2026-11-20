"""`code_epoch`-tagged `DataBuffer` drain and coupled trainer-to-rollout `S0` weight reset."""
from collections import deque
import copy, threading


class EpochDataBuffer:
    """`code_epoch`-aware trajectory buffer wrapping or standing in for `miles` `DataBuffer`."""
    def __init__(self, capacity=64, unused_handler=None):
        self.capacity, self.unused_handler = max(1, int(capacity)), unused_handler
        self.code_epoch, self.paused, self.stale_evicted = 0, False, 0
        self._entries, self._retry_groups, self._lock = deque(), deque(), threading.RLock()

    def admits(self, *, in_flight=0):
        with self._lock: return (not self.paused) and (len(self._entries) + int(in_flight) < self.capacity)
    def pause(self):
        with self._lock: self.paused = True
    def resume(self):
        with self._lock: self.paused = False
    def advance_epoch(self):
        with self._lock: self.code_epoch += 1; return self.code_epoch

    def put_group(self, group, prompt_group=None, epoch=None):
        with self._lock:
            stamped = self.code_epoch if epoch is None else int(epoch)
            self._entries.append({"epoch": stamped, "prompt_group": copy.deepcopy(prompt_group), "group": copy.deepcopy(group)})

    def get_groups(self, num_groups):
        with self._lock:
            self.drain_stale(clear_all=False)
            if len(self._entries) < num_groups:
                raise RuntimeError(f"buffer holds {len(self._entries)} group(s), need {num_groups}")
            return [self._entries.popleft()["group"] for _ in range(num_groups)]

    def drain_stale(self, clear_all=False):
        with self._lock:
            kept, evicted = deque(), 0
            while self._entries:
                entry = self._entries.popleft()
                if clear_all or entry["epoch"] < self.code_epoch:
                    evicted += 1
                    self.stale_evicted += 0 if clear_all else 1
                    if self.unused_handler is not None:
                        self.unused_handler(entry["group"], "stale_code_epoch")
                else:
                    kept.append(entry)
            self._entries = kept
            if clear_all:
                evicted += len(self._retry_groups); self._retry_groups.clear()
            return {"evicted": evicted, "code_epoch": self.code_epoch, "remaining": len(self._entries)}


def sync_weights_backuper(actor, tag="actor"):
    """Refresh `actor.weights_backuper` after an S0 restore so subsequent steps do not revert to stale weights."""
    wb = getattr(actor, "weights_backuper", None)
    tags = getattr(wb, "backup_tags", None) or ()
    if wb is not None and tag in tags:
        wb.backup(tag)
        return True
    return False


class CoupledWeightReset:
    """Snapshot trainer `S0` state once and restore trainer + rollout weights together."""
    def __init__(self, get_trainer_states, set_trainer_states, sync_rollout_weights,
                 flush_rollout_cache=None, data_buffer=None, post_restore=None):
        self._get_trainer, self._set_trainer = get_trainer_states, set_trainer_states
        self._sync_rollout, self._flush_cache = sync_rollout_weights, flush_rollout_cache
        self.data_buffer, self._post_restore, self.initial_states = data_buffer, post_restore, None

    def capture_s0(self):
        self.initial_states = copy.deepcopy(self._get_trainer())
        return self.initial_states

    def restore_and_sync(self, broadcast_rollout=True):
        if self.initial_states is None:
            self.capture_s0()
        reset_items, evicted = ["trainer_s0"], 0
        if self.data_buffer is not None:
            evicted = self.data_buffer.drain_stale(clear_all=True)["evicted"]
            reset_items.insert(0, "data_buffer")
        self._set_trainer(copy.deepcopy(self.initial_states))
        if self._post_restore is not None:
            self._post_restore()
            reset_items.append("weights_backuper")
        if broadcast_rollout:
            self._sync_rollout(); reset_items.append("rollout_weights")
        if self._flush_cache is not None:
            self._flush_cache(); reset_items.append("kv_cache")
        return {"reset": reset_items, "evicted": evicted}
