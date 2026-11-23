# Local and Self-Hosted CI

rsibox's CI suite consists of 15 CPU-only checks across two workflows:

- **`.github/workflows/pr-rules.yml` (5 checks):** `rule-pr-title`, `rule-no-ai-attribution`,
  `rule-leak-check`, `docs-loop-record`, and `rule-cross-engine-imports`.
- **`.github/workflows/cpu-tests.yml` (10 checks):** `func-conformance-cases`,
  `func-livelab-framework`, `func-fastvideo-adapter`, `func-vllm-adapter`, `func-vllm-services`,
  `func-megatron-livelab`, `func-verl-adapter`, `func-miles-adapter`, `func-sglang-adapter`, and
  `func-modal-ci-runner`.

Both workflows support two execution modes so pull requests remain gated and green regardless of
whether GitHub-hosted runner minutes are used, a self-hosted runner is attached, or GitHub Actions
dispatch is paused at the account level.

---

## Mode 1: Self-Hosted Runner (`RSIBOX_CI_RUNNER=self-hosted`)

### Routing via `vars.RSIBOX_CI_RUNNER`

Every job in `cpu-tests.yml` and `pr-rules.yml` declares:

```yaml
runs-on: ${{ vars.RSIBOX_CI_RUNNER || 'ubuntu-latest' }}
```

- **When `RSIBOX_CI_RUNNER` is set** (for example `self-hosted` or `rsibox-cpu`): jobs route to
  the registered self-hosted runner at `$0/min` without consuming GitHub-hosted runner minutes.
- **When `RSIBOX_CI_RUNNER` is unset** (external forks or public clones): jobs fall back
  automatically to `ubuntu-latest` with zero workflow edits.

Configure the repository variable with the GitHub CLI:

```bash
# Route CI to the self-hosted runner
gh variable set RSIBOX_CI_RUNNER --body "self-hosted"

# Revert to GitHub-hosted ubuntu-latest runners
gh variable delete RSIBOX_CI_RUNNER
```

Both workflows also include `workflow_dispatch:` triggers so any branch or pull request (`-f pr=<n>`)
can be validated on demand:

```bash
gh workflow run cpu-tests.yml --ref <branch>
gh workflow run pr-rules.yml --ref <branch> -f pr=<pr-number>
```

### Workspace Idempotency and CPU Isolation

Self-hosted runners reuse a persistent working directory (`_work/rsibox/rsibox`) and run under a
host user account that may have GPU drivers or `~/.local` packages installed. To guarantee identical
behavior to a fresh `ubuntu-latest` VM:

1. **Clean checkouts without persisted tokens:** every `actions/checkout@v4` step sets
   `clean: true` and `persist-credentials: false`.
2. **Strict CPU and site-package isolation:** `cpu-tests.yml` sets workflow-level environment
   variables `PYTHONNOUSERSITE: "1"`, `CUDA_VISIBLE_DEVICES: ""`, and `RSIBOX_VLLM_PYTHON: ""`,
   preventing host `~/.local` packages (such as `torch` or `triton`) or GPUs from leaking into
   standard-library CPU tests.
3. **Ephemeral virtualenv for `pytest`:** `func-vllm-services` creates an isolated virtualenv inside
   `$RUNNER_TEMP/pytest-venv` (which the runner wipes after every job) rather than running
   `pip install` against the host or toolcache Python.
4. **Conditional `setup-python`:** `actions/setup-python@v5` runs only when
   `runner.environment == 'github-hosted'`, avoiding distro-specific binary downloads into
   `_work/_tool` on self-hosted Linux hosts that already provide Python 3.12.

### Self-Hosted Runner Setup (`~/rsibox-lab/actions-runner/`)

Install and register the official GitHub Actions runner outside the git checkout under
`~/rsibox-lab/actions-runner/` with a neutral runner name (`rsibox-cpu-runner-01`):

```bash
mkdir -p ~/rsibox-lab/actions-runner && cd ~/rsibox-lab/actions-runner
curl -fsSL -o actions-runner.tar.gz \
  https://github.com/actions/runner/releases/download/v2.338.2/actions-runner-linux-x64-2.338.2.tar.gz
echo "baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40  actions-runner.tar.gz" | sha256sum -c -
tar xzf actions-runner.tar.gz && rm -f actions-runner.tar.gz

REG_TOKEN=$(gh api -X POST repos/hao-ai-lab/rsibox/actions/runners/registration-token --jq '.token')
./config.sh --unattended \
  --url https://github.com/hao-ai-lab/rsibox \
  --token "$REG_TOKEN" \
  --name rsibox-cpu-runner-01 \
  --labels self-hosted,linux,x64,rsibox-cpu \
  --work _work \
  --replace
unset REG_TOKEN

cat <<'EOF' > ~/rsibox-lab/actions-runner/.env
PYTHONNOUSERSITE=1
CUDA_VISIBLE_DEVICES=
EOF

tmux new-session -d -s rsibox-gh-runner \
  "cd ~/rsibox-lab/actions-runner && ./run.sh > ~/rsibox-lab/actions-runner/runner.stdout.log 2>&1"
gh api repos/hao-ai-lab/rsibox/actions/runners --jq '.runners[] | {name, status, busy, labels: [.labels[].name]}'
```

---

## Mode 2: Local Commit-Status CI (`statuses/<sha>`)

Self-hosted runner jobs (`RSIBOX_CI_RUNNER=self-hosted`) dispatch and execute natively on GitHub
Actions at `$0/min` even when GitHub-hosted `ubuntu-latest` jobs are blocked by an account
spending-limit or payment hold (because GitHub enforces hosted-minute billing checks only on
GitHub-hosted runner labels). However, Mode 2 is needed whenever:

- open pull requests were branched before `vars.RSIBOX_CI_RUNNER` was added to
  `.github/workflows/cpu-tests.yml` and `.github/workflows/pr-rules.yml` (so their branch workflow
  definitions still hardcode `runs-on: ubuntu-latest` until rebased); or
- no self-hosted runner is currently online.

In those cases:

1. **Pause GitHub Actions check-run creation** so pushes do not attach synthetic `failure` check
   runs:
   ```bash
   gh workflow disable cpu-tests.yml
   gh workflow disable pr-rules.yml
   ```
2. **Run the local CI verifier (`~/rsibox-lab/ci/run_local_ci.py`)** against one or more pull
   requests:
   ```bash
   python3 ~/rsibox-lab/ci/run_local_ci.py <pr-number> [<pr-number> ...]
   ```
   For each pull request, `run_local_ci.py`:
   - fetches `origin/<base>` and `pull/<n>/head` into a clean detached worktree at the exact
     `headRefOid`;
   - executes all 5 `PR rules` checks and all 6 `CPU tests` checks with `PYTHONNOUSERSITE=1` and
     `CUDA_VISIBLE_DEVICES=""`;
   - posts real `pending` -> `success` / `failure` commit statuses to
     `POST /repos/<owner>/<repo>/statuses/<head_sha>` under the exact 11 workflow context names; and
   - upserts one sanitized collapsible receipt comment (`<!-- local-ci-receipt -->`) on the PR with
     per-job timings and output.
3. **Re-enable GitHub Actions dispatch** once self-hosted dispatch is unblocked and open pull
   requests include the `RSIBOX_CI_RUNNER` workflow routing:
   ```bash
   gh workflow enable cpu-tests.yml
   gh workflow enable pr-rules.yml
   ```

---

## CI Timing Comment on Every Pull Request

`.github/workflows/ci-timing.yml` runs after `CPU tests` finishes on a pull request and writes one
comment that says where the CI time went. It is updated in place on every push, so a pull request
has one timing comment.

| The comment shows | Source |
|---|---|
| Wall-clock from the first queue to the last job done, job count, total runner time | run and job timestamps |
| Per workflow: result, wall-clock, longest queue | run record |
| Per job, slowest first: result, runner, queued, ran, slowest step | job and step records |
| Per job: every step of 2 s or more (collapsed) | step records |

- **Resolution:** whole seconds per step, from the Actions API. A step is one test suite, so the
  comment says which suite cost how much, not which test case.
- **Trust:** the workflow is triggered by `workflow_run`, so it runs the script from the default
  branch with a token that may write pull request comments, and never checks out pull request code.
- **Cost:** one job of a few seconds per CI run. It runs no tests.
- **By hand:** print the table for any commit, or post it:

  ```bash
  python3 .github/scripts/ci_timing_comment.py --repo hao-ai-lab/rsibox --sha <sha>
  gh workflow run ci-timing.yml -f sha=<sha> -f pr=<pr-number>
  ```
