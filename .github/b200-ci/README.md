# B200 CI (`b200-gpu` commit status)

This replaces the self-attested `local-ci / ...` statuses. One runner on the B200 machine tests a
**trusted** commit in a clean checkout with fresh virtualenvs. It runs the torch, Triton, GPU and
vLLM cases that GitHub-hosted CPU jobs always skip, and builds a single `b200-gpu` commit status.

| Stage | Env | What runs |
|---|---|---|
| `cpu` | fresh venv with pytest only, `CUDA_VISIBLE_DEVICES=""` | the `cpu-tests.yml` suites (GitHub parity) |
| `gpu` | fresh venv with torch 2.9.1+cu128, Triton, numpy, pytest; `FASTRSI_TEST_GPU=1`; GPUs from the pool; `FASTRSI_VLLM_PYTHON` | the same suites, so torch, GPU and vLLM cases run |
| `quickstart` | `python -m venv` per Python (3.10, 3.11, 3.12), one clean clone each | the README `## Quickstart` block, verbatim |

## Trust

- The runner fetches only `refs/heads/*` and never `refs/pull/*`. It refuses a SHA that no branch of
  GindaChen/fastrsi contains, which covers fork PR heads (they exist only under `refs/pull/N/head`).
- `--pr N` also requires the PR's head repo to be GindaChen/fastrsi and its head SHA to equal `--sha`.
- Run the runner from a trusted copy (`~/fastrsi-ci/runner`), never from the commit under test.

## Run one commit (on B200)

```sh
~/.rack/bin/rack-run --agent claude --title "b200-ci <sha12> GPUs 0,1" --cluster b200 --node b200 -- \
  python3 ~/fastrsi-ci/runner/run_b200_ci.py --sha <40-hex sha> --gpus 0,1 --vllm-python /usr/bin/python3
```

- `--gpus auto:N` picks N idle GPUs from `--gpu-pool` (default `0,1,2,3`, the CI share of B200).
- Output goes to `~/fastrsi-ci/runs/<sha12>-<UTC>/`: `summary.md`, `summary.json`, and
  `logs/<stage>/<step>.log`. Envs are deleted afterwards (6.6 GB each) unless you pass `--keep-envs`.
- It prints the status it would post. `--post` posts it with `gh api` (a pending status first, then the result).
- Exit code 0 means success, 1 a test failure, 2 a refusal or runner error.

## Loop (not installed)

`b200_ci_loop.py --once --dry-run` lists the heads it would test: main, plus open same-repo PRs.
`fastrsi-b200-ci.service` is a systemd user unit for the loop. It is not installed or enabled, and it
does not post unless you add `--post` to `ExecStart`.
