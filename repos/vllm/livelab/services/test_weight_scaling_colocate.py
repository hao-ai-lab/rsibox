"""
Unit and concurrency tests for:
1. Case 2A: Intra-node multi-worker weight colocation (`SharedWeightHolder`,
   `ColocatedWeightLease`, `GpuTimeSliceLock`).
2. Case 2B: Inter-node fast spin-up across 8 nodes x 4 GPUs = 32 GPUs via
   `DistributedWeightSeeder` NCCL slab broadcast.
3. Case 3: Active + Warm-Standby worker pool fault-tolerance contract
   (`RecoverablePatchError` <5ms rollback vs `StickyCudaError` 0ms failover).
"""

from pathlib import Path
import sys
import threading
import time
import unittest
from typing import Dict, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from weight_scaling_colocate import (  # noqa: E402
    ActiveStandbyWorkerPool,
    DistributedWeightSeeder,
    FaultClass,
    HbmBudgetExceededError,
    RecoverablePatchError,
    SharedWeightHolder,
    StandbySyncMode,
    StickyCudaError,
    WeightCorruptionError,
    torch,
)


GIB = 1024**3


def _build_mock_model_shard(local_rank: int) -> Dict[str, torch.Tensor]:
    """Create a deterministic set of model weight tensors for a TP shard."""
    torch.manual_seed(1000 + local_rank)
    return {
        "layers.0.attn.qkv.weight": torch.randn(64, 64, dtype=torch.float32),
        "layers.0.attn.o_proj.weight": torch.randn(64, 64, dtype=torch.float32),
        "layers.0.moe.gate.weight": torch.randn(16, 64, dtype=torch.float32),
        "layers.0.moe.experts.w13": torch.randn(8, 32, 64, dtype=torch.float32),
        "lm_head.weight": torch.randn(128, 64, dtype=torch.float32),
    }


class TestCase2AIntraNodeColocation(unittest.TestCase):
    """Verify Case 2A: K=4 workers colocating on 1 WeightHolder on a 189.5 GiB GB200 GPU."""

    def test_four_colocated_workers_zero_extra_weight_memory(self) -> None:
        state_dict = _build_mock_model_shard(local_rank=0)
        raw_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
        # Scale simulated weight size to 140 GiB (73.9% of 189.5 GiB GB200 HBM)
        target_weight_bytes = 140 * GIB
        scale = target_weight_bytes // raw_bytes

        holder = SharedWeightHolder(
            node_id=0,
            local_rank=0,
            total_hbm_bytes=int(189.5 * GIB),
            reserved_headroom_bytes=4 * GIB,
            simulated_bytes_scale=scale,
        )
        holder.load_weights(state_dict)

        # Without colocation, even 2 copies of 140 GiB weights (280 GiB) would OOM 189.5 GiB.
        # With ColocatedWeightLease, K=4 active workers + 1 warm standby attach to the SAME
        # physical weight pointers (0 extra weight bytes), each taking 6 GiB KV + 2 GiB scratch.
        leases = []
        for worker_index in range(4):
            lease = holder.attach_worker(
                worker_id=f"agent-{worker_index}",
                kv_cache_bytes=6 * GIB,
                scratch_bytes=2 * GIB,
                role="active",
            )
            leases.append(lease)
            self.assertEqual(lease.extra_weight_bytes, 0)

        # Also attach 1 warm-standby replica with deferred KV allocation (2 GiB scratch only)
        standby_lease = holder.attach_worker(
            worker_id="standby-0",
            kv_cache_bytes=6 * GIB,
            scratch_bytes=2 * GIB,
            role="standby",
            defer_kv_allocation=True,
        )
        self.assertEqual(standby_lease.total_dedicated_bytes, 2 * GIB)

        # Verify all 5 workers point to the exact same physical storage address (`data_ptr()`)
        for name in state_dict:
            base_ptr = leases[0].weights[name].data_ptr()
            for lease in leases[1:] + [standby_lease]:
                self.assertEqual(lease.weights[name].data_ptr(), base_ptr)

        # Total used HBM = ~140 GiB weights + 4 GiB headroom + 4*(8 GiB) + 2 GiB = ~178 GiB < 189.5 GiB
        self.assertLess(holder.used_hbm_bytes, holder.total_hbm_bytes)
        self.assertTrue(holder.verify_integrity())

        # Verify HBM budget enforcement rejects an oversized 6th worker requesting 20 GiB
        with self.assertRaises(HbmBudgetExceededError):
            holder.attach_worker(
                worker_id="agent-oom",
                kv_cache_bytes=16 * GIB,
                scratch_bytes=4 * GIB,
            )

    def test_parallel_cpu_compile_and_serialized_gpu_time_slice(self) -> None:
        holder = SharedWeightHolder(node_id=0, local_rank=0, total_hbm_bytes=189 * GIB)
        holder.load_weights(_build_mock_model_shard(0))

        for worker_index in range(4):
            holder.attach_worker(f"agent-{worker_index}", kv_cache_bytes=4 * GIB, scratch_bytes=1 * GIB)

        cpu_compile_starts = []
        barrier = threading.Barrier(4)

        def _worker_step(worker_id: str) -> None:
            # Phase 1: Parallel CPU compile / AST diff (`prepare`)
            barrier.wait()
            cpu_compile_starts.append(time.monotonic())
            time.sleep(0.015)
            # Phase 2: Time-sliced GPU micro-benchmark (`check()`)
            with holder.gpu_lock.acquire_gpu_window(worker_id):
                time.sleep(0.005)

        threads = [
            threading.Thread(target=_worker_step, args=(f"agent-{worker_index}",)) for worker_index in range(4)
        ]
        for worker_thread in threads:
            worker_thread.start()
        for worker_thread in threads:
            worker_thread.join()

        # All 4 workers started CPU compilation within <5ms of each other (parallel CPU work)
        self.assertLess(max(cpu_compile_starts) - min(cpu_compile_starts), 0.010)
        # GPU execution windows were strictly non-overlapping (zero SM/HBM contention)
        self.assertEqual(holder.gpu_lock.overlap_violations, 0)
        history = holder.gpu_lock.history
        self.assertEqual(len(history), 4)
        for entry_index in range(1, len(history)):
            prev_end = history[entry_index - 1][2]
            curr_start = history[entry_index][1]
            self.assertGreaterEqual(curr_start, prev_end)


class TestCase2BInterNodeNcclBroadcast(unittest.TestCase):
    """Verify Case 2B: 8 Nodes x 4 GPUs = 32 GPUs seeded from Node 0 via NCCL broadcast."""

    def test_eight_node_cluster_seeding_single_storage_read(self) -> None:
        num_nodes = 8
        tp_size = 4
        holders: Dict[Tuple[int, int], SharedWeightHolder] = {}
        for node_id in range(num_nodes):
            for local_rank in range(tp_size):
                holders[(node_id, local_rank)] = SharedWeightHolder(
                    node_id=node_id,
                    local_rank=local_rank,
                    total_hbm_bytes=int(189.5 * GIB),
                )

        storage_calls_by_shard: Dict[int, int] = {shard_rank: 0 for shard_rank in range(tp_size)}

        def _mock_nfs_loader(local_rank: int) -> Dict[str, torch.Tensor]:
            storage_calls_by_shard[local_rank] += 1
            return _build_mock_model_shard(local_rank)

        # Force multi-slab coalescing by setting slab_max_bytes to 32 KB
        seeder = DistributedWeightSeeder(
            num_nodes=num_nodes,
            tp_size=tp_size,
            slab_max_bytes=32 * 1024,
        )
        metrics = seeder.seed_cluster_holders(holders, _mock_nfs_loader)

        # 1. Storage read happened EXACTLY ONCE per TP shard (on Node 0), not 8 times!
        for shard_rank in range(tp_size):
            self.assertEqual(storage_calls_by_shard[shard_rank], 1)
            self.assertEqual(seeder.storage_read_count_by_rank.get((0, shard_rank)), 1)
            for node_id in range(1, num_nodes):
                self.assertEqual(seeder.storage_read_count_by_rank.get((node_id, shard_rank), 0), 0)

        # 2. Multi-slab coalescing was exercised and inter-node PG was destroyed post-seed
        self.assertGreaterEqual(metrics.num_slabs_per_rail, 2)
        self.assertTrue(metrics.inter_node_pg_destroyed)
        self.assertFalse(seeder.inter_node_pg_active)

        # 3. Every node (Nodes 1..7) has bit-exact weights matching Node 0 in its local holder
        for local_rank in range(tp_size):
            ref_lease = holders[(0, local_rank)].attach_worker("local-0", 4 * GIB, 1 * GIB)
            for node_id in range(1, num_nodes):
                node_holder = holders[(node_id, local_rank)]
                self.assertTrue(node_holder.verify_integrity())
                node_lease = node_holder.attach_worker(f"local-{node_id}", 4 * GIB, 1 * GIB)
                for name, ref_tensor in ref_lease.weights.items():
                    self.assertTrue(torch.equal(ref_tensor, node_lease.weights[name]))
                    # Different nodes hold independent node-local buffers (zero cross-node fetch)
                    self.assertNotEqual(
                        ref_tensor.data_ptr(), node_lease.weights[name].data_ptr()
                    )


class TestCase3FaultToleranceActiveStandby(unittest.TestCase):
    """Verify Case 3: Class A (<5ms rollback) vs Class B (0ms warm-standby failover)."""

    def test_recoverable_vs_sticky_cuda_fault_recovery(self) -> None:
        holder = SharedWeightHolder(node_id=0, local_rank=0, total_hbm_bytes=189 * GIB)
        holder.load_weights(_build_mock_model_shard(0))

        pool = ActiveStandbyWorkerPool(
            holder=holder,
            kv_cache_bytes=8 * GIB,
            scratch_bytes=2 * GIB,
            initial_pid_base=20000,
        )
        initial_active_pid = pool.active.pid
        initial_standby_pid = pool.standby.pid
        self.assertEqual(initial_active_pid, 20000)
        self.assertEqual(initial_standby_pid, 20001)

        # 1. Valid patch succeeds on active worker
        res_ok = pool.evaluate_patch("patch-valid-triton", lambda worker, weights: None)
        self.assertTrue(res_ok.success)
        self.assertEqual(res_ok.fault_class, FaultClass.NONE)
        self.assertEqual(res_ok.active_pid_after, initial_active_pid)
        self.assertFalse(res_ok.failover_triggered)

        # 2. Class A fault (RecoverablePatchError: e.g. shape mismatch / gate failure)
        # Rolls back in-place in <5ms; active worker PID does NOT change!
        def _class_a_patch(worker, weights) -> None:
            raise RecoverablePatchError("Shape mismatch in fused QKV projection")

        res_a = pool.evaluate_patch("patch-shape-bug", _class_a_patch)
        self.assertFalse(res_a.success)
        self.assertEqual(res_a.fault_class, FaultClass.CLASS_A_RECOVERABLE)
        self.assertFalse(res_a.failover_triggered)
        self.assertEqual(res_a.active_pid_after, initial_active_pid)
        self.assertLess(res_a.recovery_latency_ms, 5.0)
        self.assertEqual(pool.inplace_rollback_count, 1)

        # 3. Class B fault (StickyCudaError: e.g. cudaErrorIllegalAddress / CUDA 719)
        # Poisons active worker -> instant <1ms failover to Warm-Standby (pid 20001)
        # and background respawn of new Warm-Standby (pid 20002).
        def _class_b_patch(worker, weights) -> None:
            raise StickyCudaError("CUDA error 719: cudaErrorIllegalAddress in custom kernel")

        res_b = pool.evaluate_patch("patch-oob-cuda-kernel", _class_b_patch)
        self.assertFalse(res_b.success)
        self.assertEqual(res_b.fault_class, FaultClass.CLASS_B_STICKY_CUDA)
        self.assertTrue(res_b.failover_triggered)
        self.assertEqual(res_b.active_pid_before, initial_active_pid)
        self.assertEqual(res_b.active_pid_after, initial_standby_pid)
        self.assertEqual(res_b.new_standby_pid, 20002)
        self.assertLess(res_b.recovery_latency_ms, 5.0)
        self.assertEqual(pool.failover_count, 1)

        # 4. Subsequent patch runs immediately on promoted standby (20001) with zero delay
        res_next = pool.evaluate_patch("patch-post-failover", lambda worker, weights: None)
        self.assertTrue(res_next.success)
        self.assertEqual(res_next.active_pid_after, initial_standby_pid)

    def test_accept_gated_shadow_standby_preserves_accepted_patches_across_failover(self) -> None:
        """
        Verify Option 3 (`OPTION_3_ACCEPT_GATED_SHADOW`):
        - Active accepts [P1, P2, P3]; Standby is shadow-updated only on `accept()`.
        - Unaccepted candidate P_bad_classA fails with RecoverablePatchError -> rolled
          back to [P1, P2, P3] without touching Standby.
        - Unaccepted candidate P4_crash fails with StickyCudaError -> Active crashes,
          Standby is NEVER exposed to P4_crash, promotes in 0ms ALREADY holding
          [P1, P2, P3] (`failover_replayed_patches == 0`), and newly respawned Standby
          automatically replays [P1, P2, P3] from `CommittedPatchLedger`!
        """
        holder = SharedWeightHolder(node_id=0, local_rank=0, total_hbm_bytes=189 * GIB)
        holder.load_weights(_build_mock_model_shard(0))

        pool = ActiveStandbyWorkerPool(
            holder=holder,
            kv_cache_bytes=8 * GIB,
            scratch_bytes=2 * GIB,
            initial_pid_base=30000,
            sync_mode=StandbySyncMode.OPTION_3_ACCEPT_GATED_SHADOW,
        )

        # Evaluate P1 WITHOUT accepting yet -> Active has staged change, Standby still at []
        first_result = pool.evaluate_patch(
            "P1_fused_rmsnorm",
            lambda worker, _: worker.worker_state.__setitem__("rmsnorm", "v2"),
            accept_on_pass=False,
        )
        self.assertTrue(first_result.success)
        self.assertFalse(first_result.accepted)
        self.assertEqual(pool.standby.committed_patches, [])
        self.assertNotIn("rmsnorm", pool.standby.worker_state)

        # Now explicitly call accept_patch("P1_fused_rmsnorm") -> Standby gets P1!
        pool.accept_patch("P1_fused_rmsnorm")
        self.assertEqual(pool.active.committed_patches, ["P1_fused_rmsnorm"])
        self.assertEqual(pool.standby.committed_patches, ["P1_fused_rmsnorm"])
        self.assertEqual(pool.standby.worker_state.get("rmsnorm"), "v2")

        # Accept P2 and P3
        pool.evaluate_patch(
            "P2_triton_qkv",
            lambda worker, _: worker.worker_state.__setitem__("qkv", "triton_v1"),
            accept_on_pass=True,
        )
        pool.evaluate_patch(
            "P3_cudagraph_pad",
            lambda worker, _: worker.worker_state.__setitem__("cg_pad", True),
            accept_on_pass=True,
        )
        expected_accepted = ["P1_fused_rmsnorm", "P2_triton_qkv", "P3_cudagraph_pad"]
        self.assertEqual(pool.ledger.patch_ids, expected_accepted)
        self.assertEqual(pool.active.committed_patches, expected_accepted)
        self.assertEqual(pool.standby.committed_patches, expected_accepted)

        # Test a Class A failing candidate P_bad_shape that mutates worker_state before raising:
        # Must roll back worker_state to [P1, P2, P3]!
        def _bad_class_a(worker, _) -> None:
            worker.worker_state["qkv"] = "broken_qkv"
            raise RecoverablePatchError("Shape mismatch in broken_qkv")

        res_a = pool.evaluate_patch("P_bad_shape", _bad_class_a)
        self.assertFalse(res_a.success)
        self.assertEqual(pool.active.worker_state["qkv"], "triton_v1")
        self.assertEqual(pool.active.committed_patches, expected_accepted)

        # Now test Class B crashing candidate P4_sticky_oob:
        # Crashes Active (30000); Standby (30001) was NEVER exposed to P4_sticky_oob,
        # promotes immediately with [P1, P2, P3] intact and 0 failover replay needed!
        def _bad_class_b(worker, _) -> None:
            worker.worker_state["corrupted_by_p4"] = True
            raise StickyCudaError("CUDA error 700: illegal memory access in P4")

        res_b = pool.evaluate_patch("P4_sticky_oob", _bad_class_b)
        self.assertFalse(res_b.success)
        self.assertTrue(res_b.failover_triggered)
        self.assertEqual(res_b.active_pid_before, 30000)
        self.assertEqual(res_b.active_pid_after, 30001)
        self.assertEqual(res_b.failover_replayed_patches, 0)
        self.assertEqual(res_b.active_committed_patches, expected_accepted)
        self.assertNotIn("corrupted_by_p4", pool.active.worker_state)
        self.assertEqual(pool.active.worker_state["rmsnorm"], "v2")
        self.assertEqual(pool.active.worker_state["qkv"], "triton_v1")
        self.assertEqual(pool.active.worker_state["cg_pad"], True)

        # And verify the newly respawned background Standby (30002) also replayed [P1, P2, P3]!
        self.assertEqual(res_b.new_standby_pid, 30002)
        self.assertEqual(pool.standby.committed_patches, expected_accepted)
        self.assertEqual(pool.standby.worker_state["qkv"], "triton_v1")

    def test_option1_zygote_only_and_option2_clean_base_replay_modes(self) -> None:
        """Compare Option 2 (clean P0 standby + replay on failover) and Option 1 (Zygote-only)."""
        holder = SharedWeightHolder(node_id=0, local_rank=0, total_hbm_bytes=189 * GIB)
        holder.load_weights(_build_mock_model_shard(0))

        # --- Option 2: Standby sits at clean base P0 until failover ---
        pool_opt2 = ActiveStandbyWorkerPool(
            holder=holder,
            kv_cache_bytes=6 * GIB,
            scratch_bytes=2 * GIB,
            initial_pid_base=40000,
            sync_mode=StandbySyncMode.OPTION_2_CLEAN_BASE_REPLAY_ON_FAILOVER,
        )
        pool_opt2.evaluate_patch(
            "P1", lambda worker, _: worker.worker_state.__setitem__("k1", 1), accept_on_pass=True
        )
        pool_opt2.evaluate_patch(
            "P2", lambda worker, _: worker.worker_state.__setitem__("k2", 2), accept_on_pass=True
        )
        # Before failover, Option 2 standby is still at clean base P0 (`[]`)
        self.assertEqual(pool_opt2.standby.committed_patches, [])

        # On Class B crash, Option 2 standby replays [P1, P2] (`failover_replayed_patches == 2`)
        res_opt2 = pool_opt2.evaluate_patch(
            "P3_crash",
            lambda worker, _: (_ for _ in ()).throw(StickyCudaError("illegal address")),
        )
        self.assertTrue(res_opt2.failover_triggered)
        self.assertEqual(res_opt2.failover_replayed_patches, 2)
        self.assertEqual(pool_opt2.active.committed_patches, ["P1", "P2"])
        self.assertEqual(pool_opt2.active.worker_state, {"k1": 1, "k2": 2})

        # --- Option 1: Zygote-only (no idle GPU standby; spawns + replays on crash) ---
        pool_opt1 = ActiveStandbyWorkerPool(
            holder=holder,
            kv_cache_bytes=6 * GIB,
            scratch_bytes=2 * GIB,
            initial_pid_base=50000,
            sync_mode=StandbySyncMode.OPTION_1_ZYGOTE_ONLY,
        )
        self.assertIsNone(pool_opt1.standby)
        pool_opt1.evaluate_patch(
            "P1", lambda worker, _: worker.worker_state.__setitem__("k1", 10), accept_on_pass=True
        )
        res_opt1 = pool_opt1.evaluate_patch(
            "P2_crash",
            lambda worker, _: (_ for _ in ()).throw(StickyCudaError("device assert")),
        )
        self.assertTrue(res_opt1.failover_triggered)
        self.assertEqual(res_opt1.failover_replayed_patches, 1)
        self.assertEqual(pool_opt1.active.committed_patches, ["P1"])
        self.assertEqual(pool_opt1.active.worker_state, {"k1": 10})


if __name__ == "__main__":
    unittest.main()
