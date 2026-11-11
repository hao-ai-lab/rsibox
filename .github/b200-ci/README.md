# B200 CI (`b200-gpu` commit status)

This replaces the self-attested `local-ci / ...` statuses. One runner on the B200 machine tests a
**trusted** commit in a clean checkout with fresh virtualenvs. It runs the torch, Triton, GPU and
vLLM cases that GitHub-hosted CPU jobs always skip, and builds a single `b200-gpu` commit status.

| Stage | Env | What runs |
|---|---|---|
| `cpu` | fresh venv with pytest only, `CUDA_VISIBLE_DEVICES=""` | the `cpu-tests.yml` suites (GitHub parity) |
| `gpu` | fresh venv with torch 2.9.1+cu128, Triton, numpy, pytest; `FASTRSI_TEST_GPU=1`; GPUs from the pool; `FASTRSI_VLLM_PYTHON` | the same suites, so torch, GPU and vLLM cases run |
| `quickstart` | `python -m venv` per Python (3.11, 3.12 by default), one clean clone each | every ` ```sh ` and ` ```bash ` block of the README `## Quickstart` section (up to the next `## ` heading), in order, verbatim |

## Trust

- The runner fetches only `refs/heads/*` and never `refs/pull/*`. It refuses a SHA that no branch of
  hao-ai-lab/fastrsi contains, which covers fork PR heads (they exist only under `refs/pull/N/head`).
- `--pr N` also requires the PR's head repo to be hao-ai-lab/fastrsi and its head SHA to equal `--sha`.
- Run the runner from a trusted copy (`~/fastrsi-ci/runner`), never from the commit under test.

## Run one commit (on B200)

```sh
~/.rack/bin/rack-run --agent claude --title "b200-ci <sha12> GPUs 0,1" --cluster b200 --node b200 -- \
  python3 ~/fastrsi-ci/runner/run_b200_ci.py --sha <40-hex sha> --gpus 0,1 --vllm-python /usr/bin/python3
```

- `--gpus auto:N` picks N idle GPUs from `--gpu-pool` (default `0,1,2,3`, the CI share of B200).
  Each chosen GPU is locked (`flock` on `~/fastrsi-ci/locks/gpu<N>.lock`) before anything is
  posted or built and stays locked until the run ends, so two CI runs never share a GPU. An
  auto pick is checked idle again under the lock and once more right before the `gpu` stage.
- Output goes to `~/fastrsi-ci/runs/<sha12>-<UTC>/`: `summary.md`, `summary.json`, and
  `logs/<stage>/<step>.log`. Envs are deleted afterwards (6.6 GB each) unless you pass `--keep-envs`.
- It prints the status it would post. `--post` posts it with `gh api` (a pending status first, then the result).
- Exit code 0 means success, 1 a test failure, 2 a refusal or runner error.
- `~/fastrsi-ci/results/<sha>.json` records the outcome, written atomically, with `final`:

  | Outcome | `final` | Status posted with `--post` |
  |---|---|---|
  | test result (success or failure) | true | the result |
  | refused before the pending status: SHA on no branch, PR head moved, GPUs busy or locked at pick time | false | none |
  | refused after it: a picked GPU became busy before the `gpu` stage | false | pending, saying it will retry |
  | runner error before the SHA is verified (fetch, mirror) | false, not counted | none |
  | runner error after that (env build, `--deadline-seconds`, SIGTERM) | true only on the last of `--max-attempts` (3) | pending "will retry" until then, then `error` |

  Nothing is posted on a SHA before it is verified to be on a branch of this repo.

## Loop (not installed)

`b200_ci_loop.py --once --dry-run` lists the heads it would test: main, plus open same-repo PRs.
It reruns a head until its result is final, and passes `--python`, `--quickstart-pythons`,
`--stages`, `--timeout`, `--max-attempts` and `--source` through to the runner. A run gets
`--deadline-seconds` just under the loop's `--max-wait`; if it still has not written a result
by then, the loop sends SIGTERM to the pid in `~/fastrsi-ci/active/<sha>.pid` (only if that
process's command line is still the runner for that SHA), before it starts the next head.
Each step runs in its own process group; a step cut by the deadline or by SIGTERM is killed
with its whole group and recorded as a runner-error attempt. Not guaranteed: a process that
leaves its group (`setsid`, a daemon) survives, and if the runner was SIGKILLed the loop
cannot stop anything. The GPU locks, not the loop, keep two runs off the same GPU.

`python3 .github/b200-ci/test_b200_ci.py -v` runs the runner and loop golden cases on CPU (real
git, real venvs; stand-ins for `nvidia-smi`, `rack-run` and the status call).
`fastrsi-b200-ci.service` is a systemd user unit for the loop. It is not installed or enabled, and it
does not post unless you add `--post` to `ExecStart`.
