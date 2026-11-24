# Modal GPU CI (`modal-gpu` commit status)

Runs rsibox's torch, Triton, CUDA, Ray, and real-vLLM test suites on cached Modal GPU containers
and posts a single `modal-gpu` commit status for trusted commits.

## Architecture

- [`modal_app.py`](modal_app.py) defines Modal App `rsibox-gpu-ci` with two cached Python 3.12
  `debian_slim` images that build once and stay cached across CI runs:
  - `core_gpu_image`: `git`, `build-essential`, `libnuma1`, `torch==2.9.1` (`cu128`), `pytest`,
    `numpy`, `nvidia-ml-py`, `setuptools`, `wheel`, `ray`.
  - `vllm_gpu_image`: `git`, `build-essential`, `libnuma1`, `vllm`, `pytest`, `numpy`,
    `nvidia-ml-py`, `setuptools`.
- `run_core_gpu` and `run_vllm_gpu` execute in parallel on Modal GPUs (`--gpu`, default `L4`),
  keeping `repos/sglang/livelab` and `repos/vllm/livelab` strictly isolated per step.
- [`run_modal_ci.py`](run_modal_ci.py) is stdlib-only and orchestrates trust verification, archive
  creation, Modal execution, structured logs/summaries, retry settlement, and optional GitHub
  status posting (`--post`).
- [`modal_ci_loop.py`](modal_ci_loop.py) polls `main` and open same-repo PRs and invokes
  `run_modal_ci.py` for any head without a final result.
- [`parse_comment_cmd.py`](parse_comment_cmd.py) parses and validates `/run-gpu-ci` PR comments
  and `workflow_dispatch` inputs fail-closed against `ALL_SUITES` in `modal_app.py`.

## Security & Trust Boundary

- `run_modal_ci.py` fetches only `refs/heads/*` (never `refs/pull/*`) into `<work>/mirror.git` and
  refuses any SHA not contained in at least one branch of `hao-ai-lab/rsibox` (blocking fork PRs).
- `--pr N` verifies via `gh api` that `head.repo.full_name == hao-ai-lab/rsibox` and
  `head.sha == --sha`.
- Only a clean `git archive` tarball of the verified SHA is sent to Modal containers; GitHub tokens
  never enter Modal containers.
- Nothing is posted to GitHub before trust is verified. Refusals and pre-trust errors are non-final
  (`final=false`) and never burn retry attempts.

## Running One Commit

```sh
python3 .github/modal-ci/run_modal_ci.py \
  --sha <40-hex sha> \
  --work ~/rsibox-ci \
  --gpu L4
```

Useful flags:

- `--post`: post `pending` and final `modal-gpu` commit statuses via `gh api`.
- `--suites core-kernel,vllm-services`: run a subset of suites (unused Modal function is skipped).
- `--stages cpu,gpu,quickstart`: optionally run local CPU parity and README quickstart checks
  alongside Modal `gpu`.
- `--vllm-model <model>`: set `RSIBOX_VLLM_MODEL` and `RSIBOX_GPU_E2E=1` for `vllm-services`.
- `--modal-cmd "uv run --no-project --with modal modal"`: override the auto-detected Modal CLI.

Output goes to `<work>/runs/<sha12>-<UTC>/`: `summary.md`, `summary.json`, `logs/gpu/<step>.log`,
and `logs/modal-run.log`. Settlement records live in `<work>/results/<sha>.json`. Exit codes:
`0` success, `1` test failure, `2` refusal or runner error.

## Polling Loop

```sh
python3 .github/modal-ci/modal_ci_loop.py --once --dry-run
python3 .github/modal-ci/modal_ci_loop.py --interval 300 --post
```

Forwards `--gpu`, `--stages`, `--timeout`, `--max-attempts`, `--vllm-model`, `--suites`,
`--source`, and `--modal-cmd`, and checks `ps -ww` against `<work>/active/<sha>.pid` before
sending `SIGTERM` on timeout.

## GitHub Actions Trigger

[`.github/workflows/modal-gpu.yml`](../workflows/modal-gpu.yml) runs Modal GPU CI on demand using
repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` and posts the `modal-gpu` commit
status:

- **PR comment (`/run-gpu-ci`):** Any repository `OWNER`, `MEMBER`, or `COLLABORATOR` can comment
  on a same-repo PR to trigger a GPU run (acknowledged with an `eyes` reaction). Only the first
  line of the comment is parsed by [`parse_comment_cmd.py`](parse_comment_cmd.py); trailing lines
  are ignored:

  ```text
  /run-gpu-ci
  /run-gpu-ci --gpu H100 --suites core-kernel,vllm-services
  /run-gpu-ci --gpu A10G --suites vllm-services --vllm-model Qwen/Qwen3-0.6B
  ```

  Supported optional flags:
  - `--gpu <spec>`: Modal GPU spec matching `^[A-Za-z0-9][A-Za-z0-9:_-]{0,31}$` (default `L4`).
  - `--suites <csv>`: comma-separated suite names from `ALL_SUITES` in `modal_app.py` (default: all).
  - `--vllm-model <id>`: model ID matching `^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$` without `..`.

- **Manual dispatch (`workflow_dispatch`):** Trigger **Modal GPU CI** from the Actions tab (or
  `gh workflow run modal-gpu.yml`) with optional inputs `pr` (defaults to `main` when empty),
  `gpu` (default `L4`), `suites`, and `vllm_model`:

  ```sh
  gh workflow run modal-gpu.yml -f pr=206 -f gpu=L4 -f suites=core-kernel,vllm-services
  ```

- **Trust boundary:** Workflow scripts always run from the default branch (`ref: ${{ github.sha }}`,
  `persist-credentials: false`), never from PR head code. Fork PRs are refused before Modal is
  invoked. Pull requests touching `.github/modal-ci/**` or `.github/workflows/modal-gpu.yml` run
  only the offline stdlib self-test (`test_modal_ci.py -v`) with read-only permissions and zero
  secrets.

## Offline Tests

Run the stdlib golden suite (no Modal account, GPU, or network required):

```sh
python3 .github/modal-ci/test_modal_ci.py -v
```
