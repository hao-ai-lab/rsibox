# rsibox: a live lab as a service

A **lab** is the in-engine adapter that implements `INTERFACE.md`. A **box** is a lab that has
been started somewhere and made reachable: one warm engine behind one small HTTP daemon. An
agent needs only a URL and a token to edit the running engine.

Standard library only. It imports no engine: the daemon takes any `LabUnderTest` class
(`module:Class`, or a name `rsibox list` knows).

## Quick start

```sh
# from core/livelab (or with the package installed: `rsibox ...`)
python3 -m rsibox up --launcher local --lab toy --state /tmp/box1     # prints the box file
export RSIBOX=/tmp/box1/box.json                                      # url + token

python3 -m rsibox wait                       # blocks until the engine is warm; prints boot seconds
python3 -m rsibox status                     # base | edited | poisoned
python3 -m rsibox source toy.ops > ops.py    # the text the engine runs now
#   ... edit ops.py with any tool ...
python3 -m rsibox install --file toy.ops=ops.py
python3 -m rsibox check                      # output, proven, witness, cost_s
python3 -m rsibox uninstall
python3 -m rsibox events                     # every call so far, with its seconds
python3 -m rsibox down
```

`RSIBOX` names a box file. `RSIBOX_URL` plus `RSIBOX_TOKEN` work too, for an agent that runs
on another machine.

## What a patch is

`install` takes **the full new text of each changed module**, not a diff:

```json
{"files": {"toy.ops": "<whole new file>"}, "accept": [], "undo_resets": []}
```

The kernel finds the changed functions itself from the two texts and refuses a patch whose
base no longer matches what runs (`base_mismatch`). An agent edits a file with its usual tools
and sends the result; it never has to produce line numbers. Labs built on `BaseLab` also
accept `{"diff": "<unified diff>"}` (`rsibox install --diff x.patch`); the lab applies it to
its live text and refuses when it does not apply. The lab applies it with `git apply`, so `git`
must exist where the lab runs (a bare container image may lack it); without it every diff is
refused as `patch_invalid`. Full texts need nothing.

After a committed install the live text is the new base. Read it back with `source`.

A lab applies full texts to whatever is live, so a patch made from an older text would silently
drop the edits committed since. To prevent that, send the hashes you edited from:
`"base": {"toy.ops": "<hash from source>"}`. The box refuses the patch with `base_mismatch`
when a module's live hash differs, before the lab is called. Without `base` there is no such
check: use it whenever more than one writer can reach the box.

## HTTP

Every request carries `Authorization: Bearer <token>`. Bodies and replies are JSON.

| Call | Does |
|---|---|
| `POST /v1/install` | the lab's `install(patch)` |
| `POST /v1/check` | `check(workload)`; body `{"workload": ...}` or empty |
| `POST /v1/reset` | `reset(kind)`; `state` or `respawn` |
| `POST /v1/uninstall` | `uninstall(id)` |
| `GET /v1/status` | `status()` |
| `GET /v1/source?module=M` | `{"module", "text", "hash"}` of the live text |
| `GET /v1/info` | box state, lab spec, boot seconds, the call in flight, counts |
| `GET /v1/events?since=N` | calls after event `N`, each with verb, ok, outcome and seconds |
| `GET /v1/receipts` | the receipt file's records. There is no route that writes it |
| `POST /v1/close` | close the lab and stop the daemon |

| HTTP status | Means |
|---|---|
| 200 | the lab answered. `ok` in the body says whether the call succeeded |
| 401 | missing or wrong token. The lab was not called |
| 404 | unknown route |
| 409 | the box cannot run the call now: `booting`, `failed`, `closed`, or `busy` past the wait |
| 400 | the body is not a JSON object |

The reply of a verb is the lab's reply unchanged, plus `box: {seq, wall_s}`.

## Box states

`booting → ready → closed`, or `booting → failed`. The daemon listens before the engine
starts, so a client can watch a boot that takes minutes: `info` answers at once in every state
and reports `boot_s` when the engine is up. `base`, `edited` and `poisoned` are the lab's own
states and come only from `status`.

One verb runs at a time. `info` and `events` never wait. A second verb waits up to
`busy_wait_s` (default 5 s, `--busy-wait`) and then gets 409 with the verb in flight, so a
check that hangs cannot make the box look dead. A lab call that raises is answered as
`{"ok": false, "error", "raised": true}`.

## Launchers

A launcher starts the daemon somewhere and waits for its box file. It never speaks a verb.

| `--launcher` | Starts the daemon as | Needs |
|---|---|---|
| `local` | a child process | nothing |
| `docker` | `docker run` with the repo and the state folder mounted | Docker; `--image`; `--gpus` for a GPU engine |
| `docker --runtime runsc` | the same under gVisor | `runsc` registered as a Docker runtime |
| `gvisor` | `runsc run` on a generated OCI bundle, rootfs `/` behind a memory overlay | root (for example inside a privileged container) and a `runsc` binary |
| `slurm` | an `sbatch` job; the box file lands on the shared filesystem | Slurm; `--sbatch-arg` for partition, GPUs, time |

```sh
rsibox up --launcher docker --image IMG --gpus device=0 --lab vllm --real --state DIR \
          --env RSIBOX_VLLM_MODEL=/models/m --mount /models:/models:ro
rsibox up --launcher slurm --lab vllm --real --state /shared/box1 \
          --sbatch-arg=--gres=gpu:1 --sbatch-arg=--time=04:00:00 --setup 'source env.sh'
```

`up --dry-run` prints the command (or the sbatch script) and starts nothing.

The daemon binds `127.0.0.1` unless `--bind` says otherwise. Under Slurm it binds the node's
address; the box file then names the node, and you reach it directly or through an SSH tunnel.
The token is created by the daemon and stored in the box file (mode 0600).

`gvisor` passes the host's NVIDIA devices named by `CUDA_VISIBLE_DEVICES` into the sandbox
(gVisor's `nvproxy`). gVisor supports a fixed list of NVIDIA driver versions
(`runsc nvproxy list-supported-drivers`); `--allow-unsupported-driver` runs on others without
that guarantee. gVisor has no GPU proxy for AMD accelerators.

## What it costs

- Daemon start: well under a second; the engine's boot is the lab's own and is reported as
  `boot_s`.
- Per call: one local HTTP round trip. `box.wall_s` minus the lab's `cost_s` is the overhead
  the box adds.

## Contract

- `box.py` owns the box state, the call order, the event list and the token. The lab's
  `Session` still owns `base` / `edited` / `poisoned`. A launcher owns only how the daemon
  was started.
- The daemon never writes a receipt. It passes `receipts=<state>/receipts.jsonl` to the lab,
  whose shared `ReceiptWriter` appends to it, and serves the file read-only.
- No route runs a shell command. A patch is already code inside the engine, so isolation
  comes from where the box runs (container, gVisor), not from the API.

## Tests

```sh
python3 -m unittest discover -s rsibox -t . -v      # from core/livelab; CPU, real HTTP, the toy lab
```
