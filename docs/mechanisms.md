# Restart Boundaries, Interface & Tests

## Mechanisms by restart boundary

When an edit cannot be swapped in place, `rsibox` refuses before mutating state and names the cheapest restart boundary that can run it:

| Boundary | Mechanism | Edits handled | Implementation |
|---|---|---|---|
| **In place, Python** | Swap function code on every rank with journaled undo | Function & method bodies | [`core/livelab/kernel/swap.py`](../core/livelab/kernel/swap.py) |
| **In place, Triton** | Rebuild `@triton.jit` kernels & callers; restore binaries on undo | Triton kernel bodies | [`core/livelab/kernel/jit.py`](../core/livelab/kernel/jit.py) |
| **In place, CUDA graphs** | Selective `R1` recapture of affected graphs in live workers | Edits inside graph-captured paths | [`repos/vllm/livelab/adapter/recapture.py`](../repos/vllm/livelab/adapter/recapture.py) |
| **In place, C++/CUDA** | Compile edited `.cu` side-library, rebind `torch.ops._C` + `R1` recapture | Single-unit C++/CUDA kernels | [`repos/vllm/livelab/adapter/native.py`](../repos/vllm/livelab/adapter/native.py) |
| **In process, rebuild** | Re-import edited module and rebuild live objects | Constructors, new fields, module constants | [`repos/fastvideo/livelab/services/rebuild.py`](../repos/fastvideo/livelab/services/rebuild.py) |
| **Warm respawn (zygote)** | Fork from pre-imported server; weights served over CUDA IPC | Launch flags, startup code, refused edits | [`core/livelab/forksafe.py`](../core/livelab/forksafe.py), `repos/*/livelab/services/{zygote,weights,holder}.py` |
| **In-memory reshard** | Reshard held weights between TP and EP over CUDA IPC | Expert-parallelism layout switches | [`repos/vllm/livelab/services/reshard_live.py`](../repos/vllm/livelab/services/reshard_live.py) |

See [`docs/kernels-and-graphs.md`](kernels-and-graphs.md) for the deeper kernel and CUDA graph design.

## Lab interface & live-swap verification check

Every engine adapter implements [`core/livelab/INTERFACE.md`](../core/livelab/INTERFACE.md): `install(patch)`, `check(workload)`, `reset()`, `uninstall()`, and `status()` (`base`, `edited`, or `poisoned`).

The live-swap verification check ([`core/livelab/conformance/README.md`](../core/livelab/conformance/README.md)) verifies:

- **c0–c2:** Repeatable workload; no-op edit stays bit-exact; visible edit changes output and fires witnesses on every rank.
- **c3:** Live-edited output (and a chain of 3 edits) matches a fresh process started with the edit.
- **c4, c7, c9:** Undo restores output and code identity to base, step-by-step through chains, and across 20 install/undo cycles.
- **c5–c6:** Mid-install faults roll back all ranks; unsupported or silent no-op edits are refused without mutating state.
- **c8:** Crashed, hung, or externally tampered workers mark the lab `poisoned` until respawned at base.
- **c10:** Startup-only edits whose new code never ran are reported not proven (`zero_witness`).

**Limits:** A live check proves the edited path ran and matched on the chosen workload, not equivalence on all inputs. Engines need a Python control layer reachable between steps, and one agent operates one warm process at a time.

## Full CPU test suite

What CPU CI runs (`.github/workflows/cpu-tests.yml`); torch, vLLM, and Linux-only cases skip automatically on a laptop:

```sh
PYTHONPATH=core/livelab python3 -m rsibox doctor
python3 -m unittest discover -s core/livelab/kernel -t core/livelab
python3 -m unittest discover -s core/livelab/rsibox -t core/livelab
python3 -m unittest discover -s core/livelab -p 'test_forksafe.py'
python3 -m unittest discover -s core/livelab/survey -t core/livelab
(cd core/livelab && python3 -m unittest discover -s conformance -t .)
python3 -m unittest discover -s template/livelab/adapter -t template/livelab
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 scripts/verify_facts.py --strict
python3 .github/modal-ci/test_modal_ci.py -v
```
