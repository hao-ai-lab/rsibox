"""Live-edit replays of two SGLang bugs (#50148 Mamba deterministic dt_proj & #50147 DP abort race).

Runnable both via `unittest` in CPU CI and as a standalone step-by-step walkthrough:

    PYTHONPATH=repos/sglang/livelab python3 -m adapter.test_bug_replays
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from kernel import coordinator as C, locate as L, session as SS

# -- Bug 1: SGLang PR #50148 (batch_invariant_ops.matmul_persistent a.is_contiguous() dispatch) ----
BUG_50148_BASE = '''"""Stand-in of sglang/srt/batch_invariant_ops/batch_invariant_ops.py::matmul_persistent (#50148)."""
import struct


def _bf16(val):
    """Exact IEEE-754 bfloat16 truncation (upper 16 bits of binary32)."""
    return struct.unpack(">f", struct.pack(">f", float(val))[:2] + b"\\x00\\x00")[0]


class StridedView:
    """2D bf16 view: as in PyTorch, shape (1, K) reports is_contiguous() == True even when
    stride[0] > K (sliced from Mamba-1 x_dbl[:, :dt_rank]), while (B, K) for B > 1 is False."""

    def __init__(self, rows, row_stride):
        self.rows = [[_bf16(v) for v in row] for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))
        self.row_stride = row_stride

    def is_contiguous(self):
        return self.shape[0] <= 1 or self.row_stride == self.shape[1]

    def contiguous(self):
        return StridedView(self.rows, self.shape[1])


def _matmul_persistent_deepgemm(a, weight):
    # DeepGEMM TMA warp-group pairwise reduction in bf16.
    out = []
    for row in a.rows:
        cols = []
        for col_idx in range(len(weight[0])):
            vals = [_bf16(row[k] * _bf16(weight[k][col_idx])) for k in range(len(row))]
            while len(vals) > 1:
                vals = [_bf16(vals[i] + vals[i + 1]) for i in range(0, len(vals), 2)]
            cols.append(vals[0])
        out.append(cols)
    return out


def _matmul_persistent_triton(a, weight):
    # Triton sequential K-tile reduction in bf16 (non-associative vs pairwise tree).
    out = []
    for row in a.rows:
        cols = []
        for col_idx in range(len(weight[0])):
            acc = 0.0
            for k in range(len(row)):
                acc = _bf16(acc + _bf16(row[k] * _bf16(weight[k][col_idx])))
            cols.append(acc)
        out.append(cols)
    return out


def matmul_persistent(a, weight):
    if a.is_contiguous():
        return _matmul_persistent_deepgemm(a, weight)
    return _matmul_persistent_triton(a, weight)
'''

BUG_50148_FIX = BUG_50148_BASE.replace(
    "    if a.is_contiguous():\n        return _matmul_persistent_deepgemm(a, weight)",
    "    if not a.is_contiguous():\n        a = a.contiguous()\n    return _matmul_persistent_deepgemm(a, weight)",
)


# -- Bug 2: SGLang PR #50147 (DP-attention AbortReq arriving on control broadcast before work_req) -
BUG_50147_BASE = '''"""Stand-in of sglang/srt/managers/scheduler.py DP-attention recv & abort (#50147)."""


class DPSchedulerRank:
    def __init__(self, rank):
        self.rank = rank
        self.work_socket = []
        self.waiting_queue = []
        self.aborted = []
        self.executed = []
        self.max_recv_per_poll = 1

    def poll_work(self):
        out = []
        while self.work_socket and len(out) < self.max_recv_per_poll:
            out.append(self.work_socket.pop(0))
        return out

    def handle_generate_request(self, req):
        self.waiting_queue.append(req)

    def abort_request(self, abort_rid):
        kept = []
        for req in self.waiting_queue:
            if req["rid"] == abort_rid:
                self.aborted.append(abort_rid)
            else:
                kept.append(req)
        self.waiting_queue = kept

    def step_once(self, broadcast_aborts):
        work_reqs = self.poll_work()
        for req in work_reqs:
            self.handle_generate_request(req)
        for abort_rid in broadcast_aborts:
            self.abort_request(abort_rid)
        while self.waiting_queue:
            req = self.waiting_queue.pop(0)
            self.executed.append(req["rid"])
'''

BUG_50147_FIX = BUG_50147_BASE.replace(
    "    def handle_generate_request(self, req):\n"
    "        self.waiting_queue.append(req)\n\n"
    "    def abort_request(self, abort_rid):\n"
    "        kept = []\n"
    "        for req in self.waiting_queue:\n"
    "            if req[\"rid\"] == abort_rid:\n"
    "                self.aborted.append(abort_rid)\n"
    "            else:\n"
    "                kept.append(req)\n"
    "        self.waiting_queue = kept",
    "    def handle_generate_request(self, req):\n"
    "        early = getattr(self, \"_early_aborts\", set())\n"
    "        if req[\"rid\"] in early:\n"
    "            early.discard(req[\"rid\"])\n"
    "            self.aborted.append(req[\"rid\"])\n"
    "            return\n"
    "        self.waiting_queue.append(req)\n\n"
    "    def abort_request(self, abort_rid):\n"
    "        kept, matched = [], False\n"
    "        for req in self.waiting_queue:\n"
    "            if req[\"rid\"] == abort_rid:\n"
    "                self.aborted.append(abort_rid)\n"
    "                matched = True\n"
    "            else:\n"
    "                kept.append(req)\n"
    "        if not matched:\n"
    "            self.__dict__.setdefault(\"_early_aborts\", set()).add(abort_rid)\n"
    "        self.waiting_queue = kept",
)


def _load_temp_module(tmp_dir, mod_name, source_text):
    file_path = Path(tmp_dir) / f"{mod_name}.py"
    file_path.write_text(source_text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def replay_50148():
    """Return before/after/undo mismatch counts out of 15 decode steps for SGLang #50148."""
    with tempfile.TemporaryDirectory(prefix="sgl-50148-") as tmp_dir:
        mod_name = "sgl_repro_50148"
        loaded = _load_temp_module(tmp_dir, mod_name, BUG_50148_BASE)
        try:
            local_p = C.LocalParticipant("rank0")
            coord = C.StepGapCoordinator([local_p], gap=lambda fn: fn())
            sess = SS.Session(coord, [mod_name])
            weight = [[0.137 * ((k * 7 + j * 3) % 19 + 1) for j in range(8)] for k in range(16)]

            def count_divergent_steps():
                diverged = 0
                for step in range(15):
                    req0 = [0.11 * ((step + k * 5) % 17 + 1) for k in range(16)]
                    others = [[0.07 * ((step + r + k) % 13 + 1) for k in range(16)] for r in range(3)]
                    # Slice of x_dbl[:, :16] from (M, 48): row_stride=48 > K=16!
                    solo_view = loaded.StridedView([req0], row_stride=48)
                    batch_view = loaded.StridedView([req0] + others, row_stride=48)
                    solo_out = loaded.matmul_persistent(solo_view, weight)[0]
                    batch_out = loaded.matmul_persistent(batch_view, weight)[0]
                    if solo_out != batch_out:
                        diverged += 1
                return diverged

            before = count_divergent_steps()
            plan = {"modules": {mod_name: {"source": BUG_50148_FIX,
                                           "base_hash": L.text_hash(BUG_50148_BASE)}},
                    "recipes": ["R0 swap"]}
            installed = sess.install(plan)
            after = count_divergent_steps()
            proof = coord.prove(installed["txn"])
            undone = sess.uninstall()
            restored = count_divergent_steps()
            return {
                "before_diverged": before,
                "committed": installed["outcome"] == "committed",
                "after_diverged": after,
                "proven": proof["ok"],
                "witness": proof["receipt"]["witness"]["rank0"],
                "undone": undone["ok"] and coord.at_base()["ok"],
                "restored_diverged": restored,
            }
        finally:
            sys.modules.pop(mod_name, None)


def replay_50147():
    """Return lost-abort counts across 16 DP-attention trials before/after/undo for SGLang #50147."""
    with tempfile.TemporaryDirectory(prefix="sgl-50147-") as tmp_dir:
        r0_mod, r1_mod = "sgl_repro_50147_r0", "sgl_repro_50147_r1"
        plan_mod = "sgl_repro_50147"
        loaded_0 = _load_temp_module(tmp_dir, r0_mod, BUG_50147_BASE)
        loaded_1 = _load_temp_module(tmp_dir, r1_mod, BUG_50147_BASE)
        try:
            p0 = C.LocalParticipant("rank0", alias={plan_mod: r0_mod})
            p1 = C.LocalParticipant("rank1", alias={plan_mod: r1_mod})
            coord = C.StepGapCoordinator([p0, p1], gap=lambda fn: fn())
            sess = SS.Session(coord, [plan_mod])

            def count_lost_aborts():
                lost = 0
                for trial in range(16):
                    sched0 = loaded_0.DPSchedulerRank(0)
                    sched1 = loaded_1.DPSchedulerRank(1)
                    target_rid = f"req-{trial}"
                    # Rank 1's ZMQ work socket has a filler ahead of target_rid (max_recv_per_poll=1),
                    # while AbortReq(target_rid) arrives on Rank 0's control socket and broadcasts in Poll 1.
                    sched0.work_socket.append({"rid": f"r0-work-{trial}"})
                    sched1.work_socket.extend([{"rid": f"filler-{trial}"}, {"rid": target_rid}])
                    sched0.step_once([target_rid])
                    sched1.step_once([target_rid])
                    sched0.step_once([])
                    sched1.step_once([])
                    if target_rid in sched1.executed:
                        lost += 1
                return lost

            before = count_lost_aborts()
            plan = {"modules": {plan_mod: {"source": BUG_50147_FIX,
                                           "base_hash": L.text_hash(BUG_50147_BASE)}},
                    "recipes": ["R0 swap"]}
            installed = sess.install(plan)
            after = count_lost_aborts()
            proof = coord.prove(installed["txn"])
            undone = sess.uninstall()
            restored = count_lost_aborts()
            return {
                "before_lost": before,
                "committed": installed["outcome"] == "committed",
                "after_lost": after,
                "proven": proof["ok"],
                "witness": proof["receipt"]["witness"],
                "undone": undone["ok"] and coord.at_base()["ok"],
                "restored_lost": restored,
            }
        finally:
            sys.modules.pop(r0_mod, None)
            sys.modules.pop(r1_mod, None)


class SglangBugReplayTests(unittest.TestCase):
    def test_sglang_50148_mamba_deterministic_dt_proj_live_fix(self):
        result = replay_50148()
        self.assertEqual(result["before_diverged"], 15)
        self.assertTrue(result["committed"])
        self.assertEqual(result["after_diverged"], 0)
        self.assertTrue(result["proven"])
        self.assertTrue(result["undone"])
        self.assertEqual(result["restored_diverged"], 15)

    def test_sglang_50147_dp_attention_abort_race_live_fix(self):
        result = replay_50147()
        self.assertEqual(result["before_lost"], 16)
        self.assertTrue(result["committed"])
        self.assertEqual(result["after_lost"], 0)
        self.assertTrue(result["proven"])
        self.assertEqual(sorted(result["witness"]), ["rank0", "rank1"])
        self.assertTrue(result["undone"])
        self.assertEqual(result["restored_lost"], 16)


if __name__ == "__main__":
    r50148 = replay_50148()
    print(f"[#50148 Mamba deterministic dt_proj] diverged steps: "
          f"before={r50148['before_diverged']}/15 -> live-fix={r50148['after_diverged']}/15 "
          f"(proven={r50148['proven']}) -> undo={r50148['restored_diverged']}/15 (at_base={r50148['undone']})")
    r50147 = replay_50147()
    print(f"[#50147 DP-attention abort race]     lost aborts:    "
          f"before={r50147['before_lost']}/16 -> live-fix={r50147['after_lost']}/16 "
          f"(proven={r50147['proven']}) -> undo={r50147['restored_lost']}/16 (at_base={r50147['undone']})")
