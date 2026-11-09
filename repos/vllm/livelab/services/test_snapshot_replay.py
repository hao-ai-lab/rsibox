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
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    Expect: bit-identical seq_lens trajectories across repeated 10-step runs in <100ms."""
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
            assert result["total_ms"] < 100.0
            return result["steps"]

    traj_first = run_ten_steps()
    traj_second = run_ten_steps()
    assert traj_first == traj_second
    assert mocker.stats()["accepted_per_step"][:5] == [[4, 4], [2, 2], [3, 3], [1, 1], [0, 0]]
