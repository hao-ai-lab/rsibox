"""CPU unit tests for KVStateSnapshotter and SpeculativeAcceptanceMocker.

Runs in standard-library + pytest environments (GitHub Actions CPU runner) as
well as environments with PyTorch installed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import snapshot_replay  # noqa: E402
from snapshot_replay import (  # noqa: E402
    KVStateSnapshotter,
    PLACEHOLDER_TOKEN_ID,
    SimpleCachedRequest,
    SimpleSamplerOutput,
    SpeculativeAcceptanceMocker,
    ThrashingRegimeSpec,
)


# ---- Pure-Python buffer doubles for vLLM V1 / V2 GPUModelRunner ----

class FakeBuffer1D:
    """1D integer buffer with slice, .tolist(), and snapshot hooks."""

    def __init__(self, size: int, fill_val: int = 0) -> None:
        self.data = [int(fill_val)] * size

    @property
    def shape(self) -> tuple[int]:
        return (len(self.data),)

    def to_list(self) -> list[int]:
        return list(self.data)

    def write_1d(self, values: Any) -> None:
        for idx, val in enumerate(values):
            self.data[idx] = int(val)

    def clone_slice(self, rows: int, _cols: int | None = None) -> list[int]:
        return list(self.data[:rows])

    def restore_slice(self, src: list[int], rows: int, _cols: int | None = None) -> None:
        self.data[:rows] = list(src[:rows])

    def clone(self) -> FakeBuffer1D:
        out = FakeBuffer1D(0)
        out.data = list(self.data)
        return out

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, slice):
            return FakeSliceView1D(self.data[item])
        return FakeScalarView(self.data[item])

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FakeBuffer1D):
            return self.data == other.data
        if isinstance(other, FakeSliceView1D):
            return self.data == other.values
        return False


class FakeSliceView1D:
    def __init__(self, values: list[int]) -> None:
        self.values = list(values)

    def tolist(self) -> list[int]:
        return list(self.values)

    def clone(self) -> FakeSliceView1D:
        return FakeSliceView1D(self.values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FakeSliceView1D):
            return self.values == other.values
        return False


class FakeScalarView:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def item(self) -> int:
        return self.value


class FakeBuffer2D:
    """Sparse-friendly 2D buffer with .shape and row read/write hooks."""

    def __init__(self, rows: int, cols: int, fill_val: Any = 0) -> None:
        self._rows = rows
        self._cols = cols
        self._fill = fill_val
        self._row_store: dict[int, dict[int, Any]] = {}

    @property
    def shape(self) -> tuple[int, int]:
        return (self._rows, self._cols)

    def read_row(self, row_idx: int, cols: int) -> list[Any]:
        row_map = self._row_store.get(row_idx, {})
        return [row_map.get(col, self._fill) for col in range(cols)]

    def write_row(self, row_idx: int, start_col: int, values: Any) -> None:
        row_map = self._row_store.setdefault(row_idx, {})
        for offset, val in enumerate(values):
            row_map[start_col + offset] = val

    def clone_slice(self, rows: int, cols: int | None = None) -> dict[int, dict[int, Any]]:
        col_limit = self._cols if cols is None else cols
        snap: dict[int, dict[int, Any]] = {}
        for row_idx in range(min(rows, self._rows)):
            row_map = self._row_store.get(row_idx)
            if row_map:
                snap[row_idx] = {col: val for col, val in row_map.items() if col < col_limit}
        return snap

    def restore_slice(self, snap: dict[int, dict[int, Any]], rows: int, cols: int | None = None) -> None:
        col_limit = self._cols if cols is None else cols
        for row_idx in range(min(rows, self._rows)):
            row_map = self._row_store.get(row_idx, {})
            for col in [key for key in row_map if key < col_limit]:
                del row_map[col]
            if row_idx in snap:
                row_map.update(snap[row_idx])


class FakeKVCacheTensor:
    """ Mimics a layer KV cache tensor shaped [2, num_phys_blocks, block_size, heads, dim]."""

    def __init__(self, num_phys_blocks: int, block_size: int = 16) -> None:
        self.shape = (2, num_phys_blocks, block_size, 2, 8)
        self.blocks = [0.0] * num_phys_blocks
        self.bytes_per_block = 2 * block_size * 2 * 8 * 2

    def fill_block(self, block_id: int, value: float) -> None:
        self.blocks[block_id] = float(value)

    def fill_blocks(self, _block_axis: int, block_ids: list[int], value: float) -> None:
        for blk_id in block_ids:
            self.blocks[blk_id] = float(value)

    def backup_blocks(self, _block_axis: int, block_ids: list[int]) -> tuple[list[float], int]:
        pages = [self.blocks[blk_id] for blk_id in block_ids]
        return pages, len(block_ids) * self.bytes_per_block

    def restore_blocks(self, _block_axis: int, block_ids: list[int], pages: list[float]) -> None:
        for pos, blk_id in enumerate(block_ids):
            self.blocks[blk_id] = float(pages[pos])

    def clone(self) -> FakeKVCacheTensor:
        out = FakeKVCacheTensor(len(self.blocks), self.shape[2])
        out.blocks = list(self.blocks)
        return out

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeKVCacheTensor) and self.blocks == other.blocks


class FakeCpuGpuPair1D:
    def __init__(self, size: int) -> None:
        self.cpu = FakeBuffer1D(size)
        self.gpu = FakeBuffer1D(size)


class FakeCpuGpuPair2D:
    def __init__(self, rows: int, cols: int) -> None:
        self.cpu = FakeBuffer2D(rows, cols, 0)
        self.gpu = FakeBuffer2D(rows, cols, 0)


class FakeBlockTable:
    def __init__(self, max_reqs: int, max_blocks: int, max_tokens: int, block_size: int = 16, is_recurrent: bool = False) -> None:
        self.block_size = block_size
        self.is_recurrent = is_recurrent
        self.block_table = FakeCpuGpuPair2D(max_reqs, max_blocks)
        self.num_blocks_per_row = FakeBuffer1D(max_reqs)
        self.slot_mapping = FakeCpuGpuPair1D(max_tokens)


class FakeMultiGroupBlockTable:
    def __init__(self, block_tables: list[FakeBlockTable]) -> None:
        self.block_tables = block_tables

    def __getitem__(self, idx: int) -> FakeBlockTable:
        return self.block_tables[idx]


class FakeInputBatch:
    def __init__(
        self,
        max_num_reqs: int = 8,
        max_model_len: int = 135_168,
        max_num_batched_tokens: int = 16_384,
        block_sizes: tuple[int, ...] = (16,),
        recurrent_groups: tuple[bool, ...] | None = None,
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self._req_ids: list[str | None] = [None] * max_num_reqs
        self.req_id_to_index: dict[str, int] = {}
        self.token_ids_cpu = FakeBuffer2D(max_num_reqs, max_model_len, 0)
        self.is_token_ids = FakeBuffer2D(max_num_reqs, max_model_len, False)
        self.num_tokens = FakeBuffer1D(max_num_reqs, 0)
        self.num_tokens_no_spec = FakeBuffer1D(max_num_reqs, 0)
        self.num_prompt_tokens = FakeBuffer1D(max_num_reqs, 0)
        self.num_computed_tokens_cpu = FakeBuffer1D(max_num_reqs, 0)
        self.num_accepted_tokens_cpu = FakeBuffer1D(max_num_reqs, 1)
        self.spec_token_ids: list[list[int]] = [[] for _ in range(max_num_reqs)]
        self.req_output_token_ids: list[list[int] | None] = [None for _ in range(max_num_reqs)]
        self.prev_req_id_to_index: dict[str, int] | None = None

        rec_flags = recurrent_groups or tuple(False for _ in block_sizes)
        tables = [
            FakeBlockTable(max_num_reqs, (max_model_len // blk_sz) + 32, max_num_batched_tokens, blk_sz, is_rec)
            for blk_sz, is_rec in zip(block_sizes, rec_flags)
        ]
        self.block_table = FakeMultiGroupBlockTable(tables)

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)


class FakeModelRunner:
    def __init__(
        self,
        max_num_reqs: int = 8,
        max_model_len: int = 135_168,
        max_num_batched_tokens: int = 16_384,
        num_phys_blocks: int = 512,
        block_sizes: tuple[int, ...] = (16,),
        recurrent_groups: tuple[bool, ...] | None = None,
        num_layers: int = 2,
    ) -> None:
        self.max_model_len = max_model_len
        self.requests: dict[str, SimpleCachedRequest] = {}
        self.input_batch = FakeInputBatch(max_num_reqs, max_model_len, max_num_batched_tokens, block_sizes, recurrent_groups)
        self.seq_lens = FakeCpuGpuPair1D(max_num_reqs)
        self.optimistic_seq_lens_cpu = FakeBuffer1D(max_num_reqs)
        self.query_start_loc = FakeCpuGpuPair1D(max_num_reqs + 1)
        self.positions = FakeCpuGpuPair1D(max_num_batched_tokens)
        self.num_computed_tokens = FakeCpuGpuPair1D(max_num_reqs)
        self.num_accepted_tokens = FakeCpuGpuPair1D(max_num_reqs)
        self.execute_model_state: Any = None
        self.kv_caches = [FakeKVCacheTensor(num_phys_blocks, block_sizes[0]) for _ in range(num_layers)]
        self.rejection_sampler: Any = None
        # vLLM's layer -> KV-cache group map (runner.kv_cache_config) and the layers the
        # tensors are bound to (static forward context): layer i is in group i % num_groups.
        num_groups = len(block_sizes)
        names = [f"model.layers.{i}.attn" for i in range(num_layers)]
        self.kv_cache_config = SimpleNamespace(kv_cache_groups=[
            SimpleNamespace(layer_names=[n for i, n in enumerate(names) if i % num_groups == g])
            for g in range(num_groups)])
        self.compilation_config = SimpleNamespace(static_forward_context={
            n: SimpleNamespace(kv_cache=[t]) for n, t in zip(names, self.kv_caches)})


class FakeJournal:
    def __init__(self) -> None:
        self._callbacks: list[Any] = []

    def record_call(self, callback: Any) -> None:
        self._callbacks.append(callback)

    def undo_all(self) -> None:
        while self._callbacks:
            callback = self._callbacks.pop()
            callback()


@dataclass
class FakeSpecDecodeMetadata:
    draft_token_ids: FakeBuffer1D
    num_draft_tokens: list[int]
    bonus_token_ids: FakeBuffer1D
    max_spec_len: int


class FakeLogits:
    def __init__(self, target_tokens: list[int], vocab_size: int = 64) -> None:
        self._targets = list(target_tokens)
        self.vocab_size = vocab_size

    def argmax_list(self) -> list[int]:
        return list(self._targets)


class FakeV1RejectionSampler:
    def __init__(self) -> None:
        self.observed_draft_tokens: list[list[int]] = []

    def forward(self, metadata: FakeSpecDecodeMetadata, _draft_probs: Any, logits: FakeLogits, _sampling_meta: Any = None) -> SimpleSamplerOutput:
        self.observed_draft_tokens.append(metadata.draft_token_ids.to_list())
        targets = logits.argmax_list()
        drafts = metadata.draft_token_ids.to_list()
        bonuses = metadata.bonus_token_ids.to_list()
        cols = metadata.max_spec_len + 1
        rows: list[list[int]] = []
        draft_off, logit_off = 0, 0
        for req_idx, num_d in enumerate(metadata.num_draft_tokens):
            row = [PLACEHOLDER_TOKEN_ID] * cols
            for pos in range(num_d):
                if drafts[draft_off + pos] == targets[logit_off + pos]:
                    row[pos] = drafts[draft_off + pos]
                else:
                    row[pos] = targets[logit_off + pos]
                    break
            else:
                row[num_d] = bonuses[req_idx]
            rows.append(row)
            draft_off += num_d
            logit_off += num_d + 1
        return SimpleSamplerOutput(sampled_token_ids=rows)


class FakeV2RejectionSampler:
    def __init__(self, max_num_reqs: int = 8) -> None:
        self.num_sampled = FakeBuffer1D(max_num_reqs)
        self.num_rejected = FakeBuffer1D(max_num_reqs)
        self.call_count = 0

    def __call__(self, _logits: Any, _input_batch: Any, *_args: Any, **_kwargs: Any) -> Any:
        self.call_count += 1
        return []


# ---- Golden test cases ----

def test_synthesize_128k_decode_plus_8k_prefill_thrashing_and_tail_snapshot() -> None:
    """Input: 4 decode requests at 128K context (131,072 tokens) + 1 prefill request at 8K chunk.
    Expect: query_start_loc = [0, 4, 8, 12, 16, 8208], seq_lens = [131076, ..., 8192]; tail mode
    backs up only dirty blocks (<512); two 15-step replays produce identical trajectories in <100ms
    and leave read-only prefix block 1 untouched."""
    runner = FakeModelRunner(
        max_num_reqs=8,
        max_model_len=135_168,
        max_num_batched_tokens=16_384,
        num_phys_blocks=512,
        block_sizes=(16,),
        num_layers=2,
    )
    spec = ThrashingRegimeSpec(
        num_decode_reqs=4,
        decode_context_len=131_072,
        num_prefill_reqs=1,
        prefill_chunk_len=8_192,
        prefill_prior_context_len=0,
        num_spec_tokens=3,
        horizon_steps=15,
        block_sizes=(16,),
    )
    snapshotter = KVStateSnapshotter(kv_mode="tail", horizon_steps=15, max_step_tokens=4, pin_memory=False)
    findings = snapshotter.synthesize_thrashing_state(runner, spec, fill_tail_kv=True)
    assert any(item.code == "snapshot_prefix_blocks_aliased_to_fit_pool" for item in findings)
    assert runner.input_batch.num_reqs == 5
    assert runner.query_start_loc.cpu[:6].tolist() == [0, 4, 8, 12, 16, 8_208]
    assert runner.seq_lens.cpu[:5].tolist() == [131_076, 131_076, 131_076, 131_076, 8_192]

    for kv_tensor in runner.kv_caches:
        kv_tensor.fill_block(1, 77.0)

    snap = snapshotter.capture(runner)
    summary = snap.summary()
    assert summary["num_reqs"] == 5
    assert summary["total_scheduled_tokens"] == 8_208
    assert 0 < summary["dirty_blocks_per_group"][0] < 512

    run_first = snapshotter.replay_steps(runner, num_steps=15, snapshot=snap)
    run_second = snapshotter.replay_steps(runner, num_steps=15, snapshot=snap)
    assert run_first["steps"] == run_second["steps"]
    assert run_first["restore_after_ms"] < 100.0
    assert runner.seq_lens.cpu[:5].tolist() == [131_076, 131_076, 131_076, 131_076, 8_192]
    assert runner.input_batch.num_computed_tokens_cpu[:5].tolist() == [131_072, 131_072, 131_072, 131_072, 0]
    for kv_tensor in runner.kv_caches:
        assert kv_tensor.blocks[1] == 77.0


def test_kv_modes_metadata_only_tail_and_full_plus_recurrent_group() -> None:
    """Input: hybrid model runner with standard attention (group 0) and recurrent Mamba (group 1).
    Expect: metadata_only copies 0 bytes; tail mode backs up only tail blocks for group 0 and all
    blocks for recurrent group 1; full mode backs up all allocated blocks for both groups."""
    runner = FakeModelRunner(
        max_num_reqs=4,
        max_model_len=4_096,
        max_num_batched_tokens=1_024,
        num_phys_blocks=256,
        block_sizes=(16, 16),
        recurrent_groups=(False, True),
        num_layers=2,
    )
    spec = ThrashingRegimeSpec(
        num_decode_reqs=2,
        decode_context_len=1_024,
        num_prefill_reqs=0,
        prefill_chunk_len=0,
        horizon_steps=5,
        block_sizes=(16, 16),
    )
    snapshotter = KVStateSnapshotter(kv_mode="tail", horizon_steps=5, max_step_tokens=1)
    snapshotter.synthesize_thrashing_state(runner, spec, fill_tail_kv=True)

    snap_meta = snapshotter.capture(runner, kv_mode="metadata_only")
    assert snap_meta.total_backed_up_bytes == 0
    assert snap_meta.kv_backups == []

    snap_tail = snapshotter.capture(runner, kv_mode="tail")
    grp0_dirty, grp1_dirty = snap_tail.dirty_block_ids_per_group
    assert len(grp0_dirty) <= 10
    assert len(grp1_dirty) > 100

    snap_full = snapshotter.capture(runner, kv_mode="full")
    assert len(snap_full.dirty_block_ids_per_group[0]) == len(grp1_dirty)
    assert snap_full.total_backed_up_bytes > snap_tail.total_backed_up_bytes


def test_frozen_context_manager_and_journal_undo() -> None:
    """Input: steady-state runner mutated across 4 steps inside frozen() and via journal.
    Expect: both frozen() exit and journal.undo_all() restore seq_lens, slot_mapping, and KV caches."""
    runner = FakeModelRunner(max_num_reqs=4, max_model_len=2_048, max_num_batched_tokens=512, num_phys_blocks=128)
    spec = ThrashingRegimeSpec(num_decode_reqs=2, decode_context_len=512, num_prefill_reqs=1, prefill_chunk_len=128, horizon_steps=5)
    snapshotter = KVStateSnapshotter(kv_mode="tail", horizon_steps=5, max_step_tokens=2)
    snapshotter.synthesize_thrashing_state(runner, spec, fill_tail_kv=True)

    orig_seq_lens = runner.seq_lens.cpu[:3].clone()
    orig_slots = runner.input_batch.block_table[0].slot_mapping.cpu[:130].clone()
    orig_kv0 = runner.kv_caches[0].clone()

    with snapshotter.frozen(runner):
        for _ in range(4):
            snapshotter.advance_steady_state_step(runner, accepted_tokens_per_req=2)
        assert runner.seq_lens.cpu[:3] != orig_seq_lens

    assert runner.seq_lens.cpu[:3] == orig_seq_lens
    assert runner.input_batch.block_table[0].slot_mapping.cpu[:130] == orig_slots
    assert runner.kv_caches[0] == orig_kv0

    journal = FakeJournal()
    snapshotter.capture(runner, journal=journal)
    snapshotter.advance_steady_state_step(runner, accepted_tokens_per_req=1)
    assert runner.seq_lens.cpu[:3] != orig_seq_lens
    journal.undo_all()
    assert runner.seq_lens.cpu[:3] == orig_seq_lens


def test_snapshot_validation_and_findings() -> None:
    """Input: regime exceeding max_model_len or exhausting physical KV blocks.
    Expect: ValueError with clear diagnostic message."""
    runner = FakeModelRunner(max_num_reqs=4, max_model_len=1_024, max_num_batched_tokens=512, num_phys_blocks=8)
    snapshotter = KVStateSnapshotter()

    with pytest.raises(ValueError, match="exceeds runner max_model_len"):
        snapshotter.synthesize_thrashing_state(runner, ThrashingRegimeSpec(decode_context_len=2_048, num_prefill_reqs=0))

    with pytest.raises(ValueError, match="Physical KV pool has only"):
        snapshotter.synthesize_thrashing_state(
            runner,
            ThrashingRegimeSpec(num_decode_reqs=4, decode_context_len=256, num_prefill_reqs=0, horizon_steps=20),
        )


def _make_v1_spec_inputs(num_draft_per_req: list[int]) -> tuple[FakeSpecDecodeMetadata, FakeLogits]:
    total_draft = sum(num_draft_per_req)
    total_logits = sum(cnt + 1 for cnt in num_draft_per_req)
    draft_buf = FakeBuffer1D(total_draft)
    draft_buf.write_1d(list(range(10, 10 + total_draft)))
    bonus_buf = FakeBuffer1D(len(num_draft_per_req))
    bonus_buf.write_1d([50 + idx for idx in range(len(num_draft_per_req))])
    metadata = FakeSpecDecodeMetadata(
        draft_token_ids=draft_buf,
        num_draft_tokens=list(num_draft_per_req),
        bonus_token_ids=bonus_buf,
        max_spec_len=max(num_draft_per_req, default=0),
    )
    rng = random.Random(123)
    logits = FakeLogits([rng.randint(1, 60) for _ in range(total_logits)], vocab_size=64)
    return metadata, logits


def test_speculative_acceptance_mocker_fixed_k_and_input_alignment() -> None:
    """Input: V1 RejectionSampler with [4, 3, 1] draft tokens and fixed k=2 acceptance.
    Expect: [3, 5] output with k_i draft tokens + bonus token 999 + -1 padding, and pre-aligned
    draft tokens seen by the underlying kernel."""
    sampler = FakeV1RejectionSampler()
    runner = FakeModelRunner()
    runner.rejection_sampler = sampler
    mocker = SpeculativeAcceptanceMocker(accepted_tokens_per_step=2, align_inputs=True, call_underlying=True, bonus_token_id=999)
    mocker.install(runner)

    metadata, logits = _make_v1_spec_inputs([4, 3, 1])
    orig_draft = metadata.draft_token_ids.to_list()
    out = runner.rejection_sampler.forward(metadata, None, logits, None)
    sampled = out.sampled_token_ids

    assert sampled.shape == (3, 5)
    assert sampled[0, :2].tolist() == orig_draft[:2]
    assert sampled[0, 2].item() == 999
    assert sampled[0, 3:].tolist() == [PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]
    assert sampled[1, :2].tolist() == orig_draft[4:6]
    assert sampled[1, 2].item() == 999
    assert sampled[2, 0].item() == orig_draft[7]
    assert sampled[2, 1].item() == 999
    assert sampled[2, 2:].tolist() == [PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]

    assert metadata.draft_token_ids.to_list() == orig_draft
    observed = sampler.observed_draft_tokens[0]
    targets = logits.argmax_list()
    assert observed[:2] == targets[:2]
    assert observed[2] != targets[2]

    stats = mocker.stats()
    assert stats["accepted_per_step"] == [[2, 2, 1]]
    assert stats["total_accepted_draft_tokens"] == 5
    mocker.uninstall()


def test_speculative_acceptance_mocker_cyclic_and_bresenham_alpha() -> None:
    """Input: cyclic schedule [4, 2, 1, 0], request schedule [3, 1, 0], and alpha=0.75 / 0.50.
    Expect: exact deterministic token counts and realized_alpha matching target alpha."""
    mocker_cycle = SpeculativeAcceptanceMocker(accepted_tokens_per_step=[4, 2, 1, 0])
    assert [mocker_cycle.resolve_step_acceptance([4, 4])[0] for _ in range(6)] == [4, 2, 1, 0, 4, 2]

    mocker_req = SpeculativeAcceptanceMocker(accepted_tokens_per_step=[3, 1, 0], schedule_axis="request")
    assert mocker_req.resolve_step_acceptance([4, 4, 4]) == [3, 1, 0]

    mocker_alpha = SpeculativeAcceptanceMocker(accepted_tokens_per_step=0.75)
    for _ in range(20):
        assert mocker_alpha.resolve_step_acceptance([4, 4]) == [3, 3]
    assert mocker_alpha.realized_alpha == pytest.approx(0.75, abs=1e-6)

    mocker_half = SpeculativeAcceptanceMocker(accepted_tokens_per_step=0.5)
    assert [mocker_half.resolve_step_acceptance([3])[0] for _ in range(6)] == [1, 2, 1, 2, 1, 2]
    assert mocker_half.realized_alpha == pytest.approx(0.5, abs=1e-6)


def test_speculative_acceptance_mocker_v2_sampler_and_journal_undo() -> None:
    """Input: V2 RejectionSampler with [[3, 1]] acceptance matrix.
    Expect: num_sampled=[4, 2], num_rejected=[1, 3], and journal undo restores sampler."""
    v2_sampler = FakeV2RejectionSampler(max_num_reqs=4)
    runner = FakeModelRunner(max_num_reqs=4)
    runner.rejection_sampler = v2_sampler
    runner.input_batch.req_id_to_index = {"req0": 0, "req1": 1}
    runner.input_batch.spec_token_ids[0] = [11, 12, 13, 14]
    runner.input_batch.spec_token_ids[1] = [21, 22, 23, 24]
    runner.input_batch.num_speculative_steps = 4

    journal = FakeJournal()
    mocker = SpeculativeAcceptanceMocker(accepted_tokens_per_step=[[3, 1]], bonus_token_id=777)
    mocker.install(runner, journal=journal)

    out = runner.rejection_sampler(None, runner.input_batch)
    assert out.shape == (2, 5)
    assert v2_sampler.num_sampled[:2].tolist() == [4, 2]
    assert v2_sampler.num_rejected[:2].tolist() == [1, 3]
    assert out[0, 3].item() == 777
    assert out[0, 4].item() == PLACEHOLDER_TOKEN_ID
    assert out[1, 1].item() == 777

    journal.undo_all()
    assert runner.rejection_sampler is v2_sampler


def test_combined_snapshot_and_speculative_acceptance_replay() -> None:
    """Input: 128K decode + 8K prefill snapshot replayed for 10 steps with [4, 2, 3, 1, 0] spec acceptance.
    Expect: bit-identical seq_lens trajectories across repeated 10-step runs in <500ms."""
    runner = FakeModelRunner(max_num_reqs=6, max_model_len=135_168, max_num_batched_tokens=16_384, num_phys_blocks=512)
    runner.rejection_sampler = FakeV1RejectionSampler()
    spec = ThrashingRegimeSpec(num_decode_reqs=2, decode_context_len=131_072, num_prefill_reqs=1, prefill_chunk_len=8_192, num_spec_tokens=4, horizon_steps=10)
    snapshotter = KVStateSnapshotter(kv_mode="tail", horizon_steps=10, max_step_tokens=5)
    snapshotter.synthesize_thrashing_state(runner, spec, fill_tail_kv=True)
    snap = snapshotter.capture(runner)
    mocker = SpeculativeAcceptanceMocker(accepted_tokens_per_step=[4, 2, 3, 1, 0])

    def run_ten_steps() -> list[list[int]]:
        mocker.reset_stats()
        with mocker.applied(runner):
            def step_callback(cur_runner: Any, _step_idx: int) -> list[int]:
                meta, logits = _make_v1_spec_inputs([4, 4])
                out = cur_runner.rejection_sampler.forward(meta, None, logits, None)
                emitted = out.sampled_token_ids.count_non_placeholder()
                snapshotter.advance_steady_state_step(cur_runner, accepted_tokens_per_req=[emitted[0], emitted[1], 1])
                return list(cur_runner.seq_lens.cpu[:3].tolist())

            result = snapshotter.replay_steps(runner, num_steps=10, step_fn=step_callback, snapshot=snap)
            assert result["total_ms"] < 500.0, result["total_ms"]
            return result["steps"]

    traj_first = run_ten_steps()
    traj_second = run_ten_steps()
    assert traj_first == traj_second
    assert mocker.stats()["accepted_per_step"][:5] == [[4, 4], [2, 2], [3, 3], [1, 1], [0, 0]]


# ---- Audit row 5: group mapping, RNG, synchronized timing, mocker guard ----

def _two_group_runner_out_of_order():
    """A hybrid runner (two KV-cache groups) whose layer order is not its group order, as in
    vLLM (runner.kv_caches is ordered by layer index): layer 0 is in group 1, layer 1 in
    group 0, layer 2 in group 1. Each group's requests hold different physical blocks."""
    runner = FakeModelRunner(max_num_reqs=4, max_model_len=4_096, max_num_batched_tokens=1_024,
                             num_phys_blocks=256, block_sizes=(16, 16), num_layers=3)
    names = [f"model.layers.{i}.attn" for i in range(3)]
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[
        SimpleNamespace(layer_names=[names[1]]), SimpleNamespace(layer_names=[names[0], names[2]])])
    spec = ThrashingRegimeSpec(num_decode_reqs=2, decode_context_len=1_024, num_prefill_reqs=0,
                               prefill_chunk_len=0, horizon_steps=5, block_sizes=(16, 16))
    snapshotter = KVStateSnapshotter(kv_mode="tail", horizon_steps=5, max_step_tokens=1)
    snapshotter.synthesize_thrashing_state(runner, spec)
    # Give group 1 its own physical blocks (offset by 100) so the groups' block sets differ.
    table1 = runner.input_batch.block_table.block_tables[1]
    for req_idx in range(2):
        n = table1.num_blocks_per_row.to_list()[req_idx]
        row = table1.block_table.cpu.read_row(req_idx, n)
        table1.block_table.cpu.write_row(req_idx, 0, [b + 100 for b in row])
    return runner, snapshotter, names


def test_kv_blocks_are_saved_per_kv_cache_group_not_per_layer_index() -> None:
    """Input: _two_group_runner_out_of_order(); every block of every layer holds a distinct
    value; capture (tail); then every block the horizon touches in each layer's OWN group is
    overwritten. Expect: restore() brings every one of them back, and every backup entry holds
    exactly its layer's group's dirty blocks. Before: layer i was given dirty_per_group[i]
    (layer 0 got group 0's blocks), so layer 0's real blocks stayed corrupted."""
    runner, snapshotter, names = _two_group_runner_out_of_order()
    for li, kv in enumerate(runner.kv_caches):
        kv.blocks = [float(li * 1000 + b) for b in range(len(kv.blocks))]
    before = [list(kv.blocks) for kv in runner.kv_caches]
    snap = snapshotter.capture(runner)
    dirty = snap.dirty_block_ids_per_group
    assert set(dirty[0]).isdisjoint(dirty[1]), dirty
    group_of_layer = [1, 0, 1]
    for li, kv in enumerate(runner.kv_caches):
        for b in dirty[group_of_layer[li]]:
            kv.blocks[b] = -1.0
    snapshotter.restore(runner, snap)
    assert [list(kv.blocks) for kv in runner.kv_caches] == before
    by_key = {entry["key"]: entry for entry in snap.kv_backups}
    for li, name in enumerate(names):
        assert by_key[name]["block_ids"] == dirty[group_of_layer[li]]
        assert by_key[name]["group"] == group_of_layer[li]


def test_multi_group_runner_without_a_group_map_is_an_error_not_a_guess() -> None:
    runner, snapshotter, _ = _two_group_runner_out_of_order()
    del runner.kv_cache_config
    with pytest.raises(snapshot_replay.SnapshotError, match="refusing to guess"):
        snapshotter.capture(runner)


def test_rng_state_covers_numpy_and_per_request_generators() -> None:
    """Input: capture, draw from Python, numpy and torch (CPU and a per-request generator),
    restore, draw again. Expect: the same draws. Before: numpy, CUDA and the per-request
    generators were not saved (only Python and torch CPU). CUDA is covered by the GPU check."""
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    runner, snapshotter, _ = _two_group_runner_out_of_order()
    runner.input_batch.generators = {0: torch.Generator().manual_seed(7)}
    snap = snapshotter.capture(runner, kv_mode="metadata_only")
    assert {"py_random", "numpy", "torch_cpu", "generators"} <= set(snap.rng_state)

    def draws():
        return (random.random(), float(np.random.rand()), float(torch.rand(1)),
                float(torch.rand(1, generator=runner.input_batch.generators[0])))
    first = draws()
    snapshotter.restore(runner, snap)
    assert draws() == first


def test_capture_and_restore_times_are_taken_after_a_device_synchronize(monkeypatch) -> None:
    """Every clock reading of capture/restore goes through _now_ms(), which synchronizes CUDA
    first; the per-phase times are in snapshot.timing. Before: perf_counter with no
    synchronize (GPU copies under-timed) and one total only."""
    calls = []
    monkeypatch.setattr(snapshot_replay, "_sync", lambda: calls.append(1))
    runner, snapshotter, _ = _two_group_runner_out_of_order()
    snap = snapshotter.capture(runner)
    n_capture = len(calls)
    snapshotter.restore(runner, snap)
    assert n_capture >= 4 and len(calls) - n_capture >= 4
    assert {"capture_metadata_ms", "capture_kv_d2h_ms", "capture_rng_ms", "capture_total_ms",
            "restore_metadata_ms", "restore_kv_h2d_ms", "restore_rng_ms",
            "restore_total_ms"} <= set(snap.timing)


def test_restore_refuses_when_a_snapshot_request_left_the_runner() -> None:
    """Before: restore() skipped a request that had left the batch and reported success."""
    runner, snapshotter, _ = _two_group_runner_out_of_order()
    snap = snapshotter.capture(runner)
    del runner.requests[next(iter(runner.requests))]
    with pytest.raises(snapshot_replay.SnapshotError, match="left the runner"):
        snapshotter.restore(runner, snap)


def test_speculative_acceptance_mocker_cannot_touch_a_scored_run(monkeypatch) -> None:
    """The mocker fakes acceptance, so (1) it cannot be installed inside scored_run() or with
    FASTRSI_SCORED=1, and (2) a scored run cannot start while one is installed."""
    target = SimpleNamespace(rejection_sampler=SimpleNamespace(forward=lambda *a, **k: None))
    with snapshot_replay.scored_run():
        with pytest.raises(snapshot_replay.MockedInScoredRunError):
            SpeculativeAcceptanceMocker(1).install(target)
    monkeypatch.setenv("FASTRSI_SCORED", "1")
    with pytest.raises(snapshot_replay.MockedInScoredRunError):
        SpeculativeAcceptanceMocker(1).install(target)
    monkeypatch.delenv("FASTRSI_SCORED")
    mocker = SpeculativeAcceptanceMocker(1).install(target)
    try:
        with pytest.raises(snapshot_replay.MockedInScoredRunError):
            with snapshot_replay.scored_run():
                pass
    finally:
        mocker.uninstall()
    with snapshot_replay.scored_run():
        snapshot_replay.assert_unmocked()


def _one_decode_runner() -> tuple[FakeModelRunner, KVStateSnapshotter]:
    runner = FakeModelRunner(max_num_reqs=4, max_model_len=1_024, max_num_batched_tokens=256, num_phys_blocks=64)
    spec = ThrashingRegimeSpec(num_decode_reqs=1, decode_context_len=64, num_prefill_reqs=0, horizon_steps=4)
    snapshotter = KVStateSnapshotter(kv_mode="tail", horizon_steps=4, max_step_tokens=2, pin_memory=False)
    snapshotter.synthesize_thrashing_state(runner, spec, fill_tail_kv=True)
    return runner, snapshotter


def _add_request_in_row_1(runner: FakeModelRunner) -> None:
    runner.requests["new-req"] = SimpleCachedRequest("new-req", [1, 2], 2, [9], ([10],), 0)
    runner.input_batch._req_ids[1] = "new-req"
    runner.input_batch.req_id_to_index["new-req"] = 1
    runner.input_batch.spec_token_ids[1] = [77, 88]
    runner.input_batch.req_output_token_ids[1] = [99]


def test_restore_clears_the_rows_of_requests_added_after_the_snapshot() -> None:
    """Input: a snapshot of a 1-request batch; then a request joins in row 1 with draft tokens
    and output tokens. Action: restore. Expect: row 1's spec_token_ids is [] and its
    req_output_token_ids None, as for a row no request holds. Before: restore wrote rows below
    num_reqs only and left row 1's tokens for the next request in that row."""
    runner, snapshotter = _one_decode_runner()
    snap = snapshotter.capture(runner)
    _add_request_in_row_1(runner)
    snapshotter.restore(runner, snap)
    assert "new-req" not in runner.requests and runner.input_batch.num_reqs == 1
    assert runner.input_batch.spec_token_ids[1] == []
    assert runner.input_batch.req_output_token_ids[1] is None


def test_restore_checks_the_seeded_rows_before_it_changes_anything() -> None:
    """Input: (1) a snapshot with a seeded request in row 0, then a seeded request joins in
    row 1; (2) a snapshot with a seeded row 0 whose generator is gone at restore. Action:
    restore. Expect: (1) succeeds and drops row 1's generator with its request; (2) raises
    SnapshotError with the runner untouched (the post-snapshot request still there). Before:
    (1) raised SnapshotError after it had already deleted the new request and rewritten the
    batch and the KV blocks."""
    torch = pytest.importorskip("torch")
    runner, snapshotter = _one_decode_runner()
    runner.input_batch.generators = {0: torch.Generator().manual_seed(42)}
    snap = snapshotter.capture(runner)
    _add_request_in_row_1(runner)
    runner.input_batch.generators[1] = torch.Generator().manual_seed(99)
    snapshotter.restore(runner, snap)
    assert set(runner.input_batch.generators) == {0} and "new-req" not in runner.requests
    _add_request_in_row_1(runner)
    del runner.input_batch.generators[0]
    with pytest.raises(snapshot_replay.SnapshotError, match="seeded"):
        snapshotter.restore(runner, snap)
    assert "new-req" in runner.requests and runner.input_batch.req_id_to_index["new-req"] == 1


def test_a_prefill_chunk_advances_by_its_scheduled_tokens() -> None:
    """Input: one decode request and one 8,192-token prefill chunk. Action: one
    advance_steady_state_step. Expect: the decode advances by 1 (accepted tokens) and the
    prefill by 8,192 (its whole chunk). Before: both advanced by min(scheduled, accepted) = 1,
    so a replay never moved the prefill through its context."""
    runner = FakeModelRunner(max_num_reqs=4, max_model_len=32_768, max_num_batched_tokens=16_384, num_phys_blocks=512)
    spec = ThrashingRegimeSpec(num_decode_reqs=1, decode_context_len=64, num_prefill_reqs=1, prefill_chunk_len=8_192, horizon_steps=4)
    snapshotter = KVStateSnapshotter(kv_mode="tail", horizon_steps=4, max_step_tokens=2, pin_memory=False)
    snapshotter.synthesize_thrashing_state(runner, spec, fill_tail_kv=True)
    before = runner.input_batch.num_computed_tokens_cpu[:2].tolist()
    snapshotter.advance_steady_state_step(runner)
    after = runner.input_batch.num_computed_tokens_cpu[:2].tolist()
    assert [after[0] - before[0], after[1] - before[1]] == [1, 8_192]


# ---- Pure-Python V2 GPUModelRunner doubles & tests (#148) ----

class _FakeMatrix:
    """Pure-Python 2D numeric array supporting both `m[r, c]` and `m[r][c]`."""

    def __init__(self, rows: int, cols: int, fill_val: Any = 0) -> None:
        self.rows, self.cols = rows, cols
        self.data = [[fill_val] * cols for _ in range(rows)]
        self.shape = (rows, cols)

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, tuple):
            return self.data[idx[0]][idx[1]]
        return self.data[idx]

    def __setitem__(self, idx: Any, val: Any) -> None:
        if isinstance(idx, tuple):
            self.data[idx[0]][idx[1]] = val
        else:
            self.data[idx] = val


class FakeUvaBackedTensor:
    """Mimics vLLM V2 `UvaBackedTensor`: `.np` is source of truth; `.gpu` syncs only on `copy_to_uva()`."""

    def __init__(self, shape: int | tuple[int, int], fill_val: Any = 0) -> None:
        if isinstance(shape, tuple):
            self.np: Any = _FakeMatrix(shape[0], shape[1], fill_val)
            self.gpu: Any = _FakeMatrix(shape[0], shape[1], fill_val)
        else:
            self.np = [fill_val] * shape
            self.gpu = [fill_val] * shape
        self.uva_sync_count = 0

    def copy_to_uva(self) -> Any:
        self.uva_sync_count += 1
        self.gpu = copy.deepcopy(self.np)
        return self.gpu


class FakeStagedWriteTensor:
    """Mimics vLLM V2 `StagedWriteTensor` (`.gpu` + `clear_staged_writes()`)."""

    def __init__(self, shape: int | tuple[int, int], fill_val: Any = 0) -> None:
        self.gpu: Any = (
            FakeBuffer2D(shape[0], shape[1], fill_val)
            if isinstance(shape, tuple)
            else [fill_val] * shape
        )
        self.staged_pending = False
        self.clear_count = 0

    def clear_staged_writes(self) -> None:
        self.staged_pending = False
        self.clear_count += 1


class FakeV2RequestState:
    def __init__(self, max_num_reqs: int = 6, max_model_len: int = 256, num_speculative_steps: int = 2) -> None:
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices: list[int] = list(range(max_num_reqs))
        self.all_token_ids = FakeStagedWriteTensor((max_num_reqs, max_model_len), fill_val=0)
        self.prompt_len = FakeUvaBackedTensor(max_num_reqs, fill_val=0)
        self.prefill_len = FakeUvaBackedTensor(max_num_reqs, fill_val=0)
        self.total_len = FakeStagedWriteTensor(max_num_reqs, fill_val=0)
        self.num_computed_prefill_tokens = [0] * max_num_reqs
        self.num_computed_tokens = FakeStagedWriteTensor(max_num_reqs, fill_val=0)
        self.num_computed_tokens_np = [0] * max_num_reqs
        self.last_sampled_tokens = [[0] for _ in range(max_num_reqs)]
        self.max_seq_len = [0] * max_num_reqs
        self.draft_tokens = [[0] * num_speculative_steps for _ in range(max_num_reqs)]
        self.next_prefill_tokens = _FakeMatrix(1, max_num_reqs, fill_val=0)

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)


class FakeV2BlockTables:
    def __init__(
        self,
        max_num_reqs: int = 6,
        block_sizes: tuple[int, ...] = (16,),
        max_num_blocks: int = 16,
        slot_mapping_enabled: tuple[bool, ...] | None = None,
    ) -> None:
        self.block_sizes = list(block_sizes)
        self.num_kv_cache_groups = len(self.block_sizes)
        self._slot_mapping_enabled = (
            list(slot_mapping_enabled) if slot_mapping_enabled is not None else [True] * self.num_kv_cache_groups
        )
        self.block_tables = [
            FakeStagedWriteTensor((max_num_reqs, max_num_blocks), fill_val=-1)
            for _ in self.block_sizes
        ]
        self.num_blocks = FakeUvaBackedTensor((self.num_kv_cache_groups, max_num_reqs), fill_val=0)
        self._num_dirty_rows = 0


class FakeV2Sampler:
    def __init__(self, max_num_reqs: int = 6, vocab_size: int = 32, thinking_enabled: bool = True) -> None:
        self.sampling_states = SimpleNamespace(
            temperature=FakeUvaBackedTensor(max_num_reqs, fill_val=1.0),
            top_k=FakeUvaBackedTensor(max_num_reqs, fill_val=vocab_size),
            top_p=FakeUvaBackedTensor(max_num_reqs, fill_val=1.0),
            min_p=FakeUvaBackedTensor(max_num_reqs, fill_val=0.0),
            seeds=FakeUvaBackedTensor(max_num_reqs, fill_val=0),
            seeds_set=[False] * max_num_reqs,
            num_logprobs=[-1] * max_num_reqs,
        )
        self.penalties_state = SimpleNamespace(
            use_penalty=[False] * max_num_reqs,
            repetition_penalty=FakeUvaBackedTensor(max_num_reqs, fill_val=1.0),
            frequency_penalty=FakeUvaBackedTensor(max_num_reqs, fill_val=0.0),
            presence_penalty=FakeUvaBackedTensor(max_num_reqs, fill_val=0.0),
            output_bin_counts=[[0] * vocab_size for _ in range(max_num_reqs)],
        )
        self.thinking_budget_state = SimpleNamespace(
            enabled=thinking_enabled,
            use_thinking_budget=[False] * max_num_reqs,
            thinking_token_budget=FakeUvaBackedTensor(max_num_reqs, fill_val=-1),
            cached_last_start=[-1] * max_num_reqs,
            cached_last_end=[-1] * max_num_reqs,
            cached_scan_pos=[0] * max_num_reqs,
        )


class FakeV2ModelRunner:
    """Stand-in for vLLM 0.33.0 V2 `GPUModelRunner` (no `input_batch`)."""

    def __init__(
        self,
        *,
        max_num_reqs: int = 6,
        block_sizes: tuple[int, ...] = (16,),
        slot_mapping_enabled: tuple[bool, ...] | None = None,
        num_phys_blocks: int = 64,
        with_config: bool = True,
    ) -> None:
        self.req_states = FakeV2RequestState(max_num_reqs=max_num_reqs)
        self.block_tables = FakeV2BlockTables(
            max_num_reqs=max_num_reqs,
            block_sizes=block_sizes,
            slot_mapping_enabled=slot_mapping_enabled,
        )
        self.sampler = FakeV2Sampler(max_num_reqs=max_num_reqs)
        self.model_state = SimpleNamespace(
            num_accepted_tokens_gpu=[1] * max_num_reqs,
            _mamba_state_idx_gpu=[0] * max_num_reqs,
        )
        self.execute_model_state: Any = {"cached": True}
        self.kv_caches = [FakeKVCacheTensor(num_phys_blocks, block_size=bsize) for bsize in block_sizes]
        if with_config:
            self.kv_cache_config = SimpleNamespace(
                kv_cache_groups=[SimpleNamespace(layer_names=[f"layer_{g}"]) for g in range(len(block_sizes))]
            )
            self.compilation_config = SimpleNamespace(
                static_forward_context={f"layer_{g}": SimpleNamespace(kv_cache=[self.kv_caches[g]]) for g in range(len(block_sizes))}
            )
        else:
            self.kv_cache_config = None
            self.compilation_config = None


def _populate_sparse_v2_runner(
    *,
    block_sizes: tuple[int, ...] = (16,),
    slot_mapping_enabled: tuple[bool, ...] | None = None,
    with_config: bool = True,
) -> FakeV2ModelRunner:
    runner = FakeV2ModelRunner(
        block_sizes=block_sizes,
        slot_mapping_enabled=slot_mapping_enabled,
        with_config=with_config,
    )
    rs, bt = runner.req_states, runner.block_tables
    # Sparse slots [2, 5]; slots [0, 1, 3, 4] free (slot 0 next popped).
    rs.req_id_to_index = {"r2": 2, "r5": 5}
    rs.index_to_req_id = {2: "r2", 5: "r5"}
    rs.free_indices = [4, 3, 1, 0]

    for slot, prompt_len, comp_tokens, seed, blocks_grp0 in (
        (2, 30, 38, 101, [1, 2, 3]),     # 38 tokens -> tail overlaps block index 2 (phys 3); prefix phys 1, 2
        (5, 18, 22, 202, [4, 5]),        # 22 tokens -> tail overlaps block index 1 (phys 5); prefix phys 4
    ):
        rs.prompt_len.np[slot] = prompt_len
        rs.prefill_len.np[slot] = prompt_len
        rs.max_seq_len[slot] = 128
        rs.num_computed_tokens_np[slot] = comp_tokens
        rs.num_computed_prefill_tokens[slot] = prompt_len
        rs.total_len.gpu[slot] = comp_tokens + 1
        rs.num_computed_tokens.gpu[slot] = comp_tokens
        rs.last_sampled_tokens[slot] = [500 + slot]
        rs.draft_tokens[slot] = [600 + slot, 700 + slot]
        rs.next_prefill_tokens[0, slot] = 800 + slot
        rs.all_token_ids.gpu.write_row(slot, 0, [10 + i + slot for i in range(comp_tokens + 1)])

        bt.block_tables[0].gpu.write_row(slot, 0, blocks_grp0)
        bt.num_blocks.np[0, slot] = len(blocks_grp0)
        if len(bt.block_tables) > 1:
            grp1_blocks = [20 + slot * 2, 21 + slot * 2]
            bt.block_tables[1].gpu.write_row(slot, 0, grp1_blocks)
            bt.num_blocks.np[1, slot] = len(grp1_blocks)

        runner.sampler.sampling_states.seeds.np[slot] = seed
        runner.sampler.sampling_states.seeds_set[slot] = True
        runner.sampler.sampling_states.num_logprobs[slot] = 1

    rs.prompt_len.copy_to_uva()
    rs.prefill_len.copy_to_uva()
    bt.num_blocks.copy_to_uva()
    runner.sampler.sampling_states.seeds.copy_to_uva()

    # Enable penalties only on slot 2; thinking budget on slot 5.
    runner.sampler.penalties_state.use_penalty[2] = True
    runner.sampler.penalties_state.repetition_penalty.np[2] = 1.15
    runner.sampler.penalties_state.output_bin_counts[2][7] = 4
    runner.sampler.thinking_budget_state.use_thinking_budget[5] = True
    runner.sampler.thinking_budget_state.thinking_token_budget.np[5] = 64
    runner.sampler.thinking_budget_state.cached_last_start[5] = 9
    runner.sampler.thinking_budget_state.cached_scan_pos[5] = 18
    runner.model_state.num_accepted_tokens_gpu[2] = 2
    runner.model_state._mamba_state_idx_gpu[5] = 3
    bt._num_dirty_rows = 2

    for g_idx, kv_t in enumerate(runner.kv_caches):
        for bid in range(1, 32):
            kv_t.fill_block(bid, float(g_idx * 100 + bid))
    return runner


def test_v2_kv_modes_and_recurrent_group() -> None:
    """V2 `GPUModelRunner`: `metadata_only`, `tail`, `full`, recurrent group (`_slot_mapping_enabled=False`),
    and invalid `kv_mode` / missing runner state diagnostics."""
    runner = _populate_sparse_v2_runner(block_sizes=(16, 16), slot_mapping_enabled=(True, False))
    snapper = KVStateSnapshotter(kv_mode="tail", horizon_steps=2, max_step_tokens=2, pin_memory=False)

    snap_meta = snapper.capture(runner, kv_mode="metadata_only")
    assert snap_meta.input_slices["runner_kind"] == "v2"
    assert snap_meta.dirty_block_ids_per_group == [] and snap_meta.kv_backups == []

    # Under tail mode: slot 2 (comp=38 -> blk 2 = phys 3, end=ceil((39+4)/16)+1=4 -> [3]);
    # slot 5 (comp=22 -> blk 1 = phys 5). Group 1 has slot_mapping_enabled=False -> all blocks [24, 25, 30, 31].
    snap_tail = snapper.capture(runner, kv_mode="tail")
    assert snap_tail.dirty_block_ids_per_group == [[3, 5], [24, 25, 30, 31]]

    snap_full = snapper.capture(runner, kv_mode="full")
    assert snap_full.dirty_block_ids_per_group == [[1, 2, 3, 4, 5], [24, 25, 30, 31]]

    with pytest.raises(ValueError, match="Unsupported kv_mode"):
        snapper.capture(runner, kv_mode="invalid")

    bare_runner = SimpleNamespace(input_batch=None, req_states=None, block_tables=None)
    with pytest.raises(ValueError, match="input_batch"):
        snapper.capture(bare_runner)


def test_v2_capture_restore_sparse_slots_uva_and_post_snapshot_cleanup() -> None:
    """V2 `capture()` + post-snapshot mutation + `restore()`: restores sparse active slots,
    GPU/UVA buffers (`copy_to_uva()`), staged write clearing, KV blocks, and drops post-snapshot requests."""
    runner = _populate_sparse_v2_runner(block_sizes=(16,), slot_mapping_enabled=(True,))
    snapper = KVStateSnapshotter(kv_mode="tail", horizon_steps=4, max_step_tokens=1, pin_memory=False)
    ref_kv = runner.kv_caches[0].clone()
    snap = snapper.capture(runner)
    assert snap.dirty_block_ids_per_group == [[3, 5]]

    rs, bt = runner.req_states, runner.block_tables
    uva_prompt_before = rs.prompt_len.uva_sync_count
    uva_nb_before = bt.num_blocks.uva_sync_count
    uva_seed_before = runner.sampler.sampling_states.seeds.uva_sync_count

    # Mutate active slots [2, 5], stage pending writes, add post-snapshot req in slot 0, corrupt tail KV.
    rs.req_id_to_index["r99"] = 0
    rs.index_to_req_id[0] = "r99"
    rs.free_indices.pop()
    rs.total_len.gpu[0] = 42
    rs.num_computed_tokens.gpu[0] = 41
    rs.prompt_len.np[5] = 999
    rs.prefill_len.np[5] = 999
    rs.num_computed_tokens_np[2] = 77
    rs.num_computed_tokens.gpu[2] = 77
    rs.total_len.gpu[2] = 78
    rs.last_sampled_tokens[2] = [9999]
    rs.draft_tokens[2] = [8888, 8889]
    rs.next_prefill_tokens[0, 5] = 7777
    rs.all_token_ids.gpu.write_row(2, 0, [999] * 40)
    rs.all_token_ids.staged_pending = True

    bt.block_tables[0].gpu.write_row(2, 0, [1, 2, 14, 15])
    bt.block_tables[0].staged_pending = True
    bt.num_blocks.np[0, 2] = 4
    bt._num_dirty_rows = 5

    runner.sampler.sampling_states.seeds.np[2] = -999
    runner.sampler.sampling_states.seeds_set[0] = True
    runner.sampler.sampling_states.num_logprobs[0] = 5
    runner.sampler.penalties_state.use_penalty[0] = True
    runner.sampler.penalties_state.output_bin_counts[2][7] = 99
    runner.sampler.thinking_budget_state.cached_last_start[5] = 99
    runner.model_state.num_accepted_tokens_gpu[2] = 9
    runner.model_state._mamba_state_idx_gpu[5] = 9
    runner.execute_model_state = {"stale": 1}
    runner.kv_caches[0].fill_blocks(1, [3, 5], -123.0)

    snapper.restore(runner, snap)

    assert rs.req_id_to_index == {"r2": 2, "r5": 5}
    assert rs.index_to_req_id == {2: "r2", 5: "r5"}
    assert rs.free_indices == [4, 3, 1, 0]
    assert rs.total_len.gpu[0] == 0 and rs.num_computed_tokens.gpu[0] == 0
    assert rs.prompt_len.np[5] == 18 and rs.prefill_len.np[5] == 18
    assert rs.num_computed_tokens_np[2] == 38 and rs.num_computed_tokens.gpu[2] == 38
    assert rs.total_len.gpu[2] == 39
    assert rs.last_sampled_tokens[2] == [502] and rs.draft_tokens[2] == [602, 702]
    assert rs.next_prefill_tokens[0, 5] == 805
    assert rs.all_token_ids.gpu.read_row(2, 39) == [10 + i + 2 for i in range(39)]
    assert rs.all_token_ids.staged_pending is False
    assert bt.block_tables[0].gpu.read_row(2, 3) == [1, 2, 3]
    assert bt.block_tables[0].staged_pending is False
    assert bt.num_blocks.np[0, 2] == 3 and bt._num_dirty_rows == 0
    assert rs.prompt_len.uva_sync_count > uva_prompt_before
    assert bt.num_blocks.uva_sync_count > uva_nb_before
    assert runner.sampler.sampling_states.seeds.uva_sync_count > uva_seed_before
    assert runner.sampler.sampling_states.seeds.np[2] == 101
    assert runner.sampler.sampling_states.seeds_set[0] is False
    assert runner.sampler.sampling_states.num_logprobs[0] == -1
    assert runner.sampler.penalties_state.use_penalty[0] is False
    assert runner.sampler.penalties_state.output_bin_counts[2][7] == 4
    assert runner.sampler.thinking_budget_state.cached_last_start[5] == 9
    assert runner.model_state.num_accepted_tokens_gpu[2] == 2
    assert runner.model_state._mamba_state_idx_gpu[5] == 3
    assert runner.execute_model_state is None
    assert runner.kv_caches[0] == ref_kv


def test_v2_restore_fails_closed_when_request_leaves_or_kv_config_missing() -> None:
    """V2 fail-closed checks: multi-group runner without `kv_cache_config` raises `SnapshotError`;
    removing a snapshot request causes `restore()` to raise `SnapshotError` before mutating."""
    bad_runner = _populate_sparse_v2_runner(block_sizes=(16, 16), with_config=False)
    snapper = KVStateSnapshotter(kv_mode="tail", horizon_steps=2, max_step_tokens=1, pin_memory=False)
    with pytest.raises(snapshot_replay.SnapshotError, match="KV-cache groups"):
        snapper.capture(bad_runner)

    runner = _populate_sparse_v2_runner(block_sizes=(16,))
    snap = snapper.capture(runner)
    del runner.req_states.req_id_to_index["r5"]
    runner.req_states.num_computed_tokens_np[2] = 777
    with pytest.raises(snapshot_replay.SnapshotError, match="r5"):
        snapper.restore(runner, snap)
    assert runner.req_states.num_computed_tokens_np[2] == 777
