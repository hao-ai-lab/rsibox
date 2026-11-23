# Modal GPU CI (`modal-gpu` commit status)

Runs FastRSI's torch, Triton, CUDA, Ray, and real-vLLM test suites on cached Modal GPU containers
and posts a single `modal-gpu` commit status for trusted commits.

## Architecture

- [`modal_app.py`](modal_app.py) defines Modal App `fastrsi-gpu-ci` with two cached Python 3.12
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
  --work ~/fastrsi-ci \
  --gpu L4
```

Useful flags:

- `--post`: post `pending` and final `modal-gpu` commit statuses via `gh api`.
- `--suites core-kernel,vllm-services`: run a subset of suites (unused Modal function is skipped).
- `--stages cpu,gpu,quickstart`: optionally run local CPU parity and README quickstart checks
  alongside Modal `gpu`.
- `--vllm-model <model>`: set `FASTRSI_VLLM_MODEL` and `FASTRSI_GPU_E2E=1` for `vllm-services`.
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

## Optional GitHub Actions Trigger

To run the GPU suite from GitHub Actions on trusted same-repo pushes/PRs using repository secrets
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`:

```yaml
jobs:
  modal-gpu:
    if: github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install modal
      - env:
          MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
          MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
          TARGET_SHA: ${{ github.event.pull_request.head.sha || github.sha }}
        run: |
          python3 .github/modal-ci/run_modal_ci.py \
            --source "${{ github.server_url }}/${{ github.repository }}.git" \
            --sha "$TARGET_SHA" \
            --work "$RUNNER_TEMP/modal-ci" \
            --gpu L4
```

## Offline Tests

Run the stdlib golden suite (no Modal account, GPU, or network required):

```sh
python3 .github/modal-ci/test_modal_ci.py -v
```
