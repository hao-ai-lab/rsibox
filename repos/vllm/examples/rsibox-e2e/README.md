# rsibox end to end on a real vLLM engine

GPU cases that drive a real `vllm serve` through the rsibox HTTP API only: the calls an agent
makes. The CPU tests in `core/livelab/rsibox/` prove the daemon's plumbing on the toy lab; these
prove that the plumbing does the right thing to a real engine.

## What is here

| File | What |
|---|---|
| `cases.json` | the step cases, as data: edits, refusals, lifecycle |
| `run.py` | runs the step cases against one box and judges every reply |
| `scenarios.py` | four box-level situations a step list cannot express |

Each case has an `id`, the `pain_point` it guards against, what it `proves` and what it
`does_not_prove`, and `steps`. A step is one API call with its expected result:

```json
{"op": "install", "as": "t1",
 "edits": [{"module": "vllm.v1.core.sched.utils",
            "old": "        or request.num_output_tokens >= request.max_tokens\n",
            "new": "        or request.num_output_tokens >= max(1, request.max_tokens - 4)\n"}],
 "expect": {"outcome": "committed"}},
{"op": "check", "expect": {"output": "ne:base", "proven": true}},
{"op": "uninstall", "to": "t1", "expect": {"undone": 1, "state": "base"}}
```

An edit names an anchor that must occur exactly once in the module's live text, so a case
fails loudly when vLLM's source has moved. The runner saves the base output before each case
and requires it again afterwards; a case that leaks an edit fails itself.

| Lens | Cases cover |
|---|---|
| edits | methods, static and class methods, properties, lambdas, generators, decorated functions; worker role, engine-core role and both in one install; chains with partial undo; code the workload never reaches (`proven` must be false); reverse ports of merged upstream PRs sent as diffs; the live-edited output against a restart with the same edit on disk (`fresh`) |
| refusals | syntax errors and garbage; comment- and docstring-only changes; module-level changes; closures; removed functions; defaults; constructor edits and new fields; a patch made from a stale text; a stale diff; two files of which one is refused (nothing may be installed); unknown modules and ids. Every refusal also proves the engine was left untouched |
| lifecycle | repeatability; no drift over install and uninstall cycles; an edit across many requests and across `reset state`; `reset respawn` while edited; transaction ids in status; receipts and events in order; the live text and its hash; the HTTP surface (401, 404, 400) changing nothing |

`scenarios.py`: verbs while the engine boots; a second verb while a long check runs; the
engine's worker killed from outside, then poison, refusal and `reset respawn`; `down` leaving
no process and no GPU memory.

## Run

The box is any `vllm --real` box (`core/livelab/rsibox/README.md`); the cases do not care how
it was launched, so the same file checks a local box, a Docker box and a gVisor box.

```sh
export FASTRSI_VLLM_MODEL=<model> FASTRSI_VLLM_VENV=<venv> FASTRSI_VLLM_WORK=<scratch>
rsibox up --launcher local --lab vllm --real --state <dir> && export RSIBOX=<dir>/box.json
rsibox wait

python3 run.py --out results.json --fresh-state <scratch>     # every step case
python3 run.py --only edits-check-stop-one-token-early        # one case
python3 scenarios.py --state <other-dir>                      # starts and stops its own box
```

`--fresh-state` enables the `fresh` steps: each starts a second engine on the same GPU from a
copy of the vLLM sources with the edit on disk, so the GPU must fit two engines. Without it
those cases are reported as skipped, never as passed.

The three cases that install a unified diff need `git` where the lab runs. In a container image
without it they fail at their first step with `patch_invalid`, which is the lab's real behaviour
there, not a flaw in the cases.

## Cost

One engine boot, then seconds per case. A `fresh` step and the `reset respawn` case each cost
an engine start. `scenarios.py` costs one boot, one respawn and a long check.

## What these cases do not prove

- Anything about engines other than vLLM, or about vLLM with `torch.compile` or CUDA graphs:
  the engine runs eager.
- That a no-op edit was really swapped: for edits that keep the output, the only evidence is
  the kernel's own witness. Witness counts are cumulative since the install, so a second
  `proven: true` in one case says nothing new.
- That `reset state` did anything, where a case only shows the edit survived it.
- Isolation. A case that passes inside gVisor shows the API works there, not that the sandbox
  holds.

The cases were written against vLLM 0.33 with a text model on one GPU, tensor-parallel 1.
