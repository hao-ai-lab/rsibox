# Local and Self-Hosted CI

FastRSI's CI suite consists of 11 CPU-only checks across two workflows:

- **`.github/workflows/pr-rules.yml` (5 checks):** `title`, `attribution`, `leak-check`,
  `loop-record`, and `cross-engine-imports`.
- **`.github/workflows/cpu-tests.yml` (6 checks):** `conformance-cases`, `livelab-framework`,
  `fastvideo-adapter`, `vllm-adapter`, `vllm-services`, and `megatron-livelab`.

Both workflows support two execution modes so pull requests remain gated and green regardless of
whether GitHub-hosted runner minutes are used, a self-hosted runner is attached, or GitHub Actions
dispatch is paused at the account level.

---

## Mode 1: Self-Hosted Runner (`FASTRSI_CI_RUNNER=self-hosted`)

### Routing via `vars.FASTRSI_CI_RUNNER`

Every job in `cpu-tests.yml` and `pr-rules.yml` declares:

```yaml
runs-on: ${{ vars.FASTRSI_CI_RUNNER || 'ubuntu-latest' }}
```

- **When `FASTRSI_CI_RUNNER` is set** (for example `self-hosted` or `fastrsi-cpu`): jobs route to
  the registered self-hosted runner at `$0/min` without consuming GitHub-hosted runner minutes.
- **When `FASTRSI_CI_RUNNER` is unset** (external forks or public clones): jobs fall back
  automatically to `ubuntu-latest` with zero workflow edits.

Configure the repository variable with the GitHub CLI:

```bash
# Route CI to the self-hosted runner
gh variable set FASTRSI_CI_RUNNER --body "self-hosted"

# Revert to GitHub-hosted ubuntu-latest runners
gh variable delete FASTRSI_CI_RUNNER
```

Both workflows also include `workflow_dispatch:` triggers so any branch or pull request (`-f pr=<n>`)
can be validated on demand:

```bash
gh workflow run cpu-tests.yml --ref <branch>
gh workflow run pr-rules.yml --ref <branch> -f pr=<pr-number>
```

### Workspace Idempotency and CPU Isolation

Self-hosted runners reuse a persistent working directory (`_work/fastrsi/fastrsi`) and run under a
host user account that may have GPU drivers or `~/.local` packages installed. To guarantee identical
behavior to a fresh `ubuntu-latest` VM:

1. **Clean checkouts without persisted tokens:** every `actions/checkout@v4` step sets
   `clean: true` and `persist-credentials: false`.
2. **Strict CPU and site-package isolation:** `cpu-tests.yml` sets workflow-level environment
   variables `PYTHONNOUSERSITE: "1"`, `CUDA_VISIBLE_DEVICES: ""`, and `FASTRSI_VLLM_PYTHON: ""`,
   preventing host `~/.local` packages (such as `torch` or `triton`) or GPUs from leaking into
   standard-library CPU tests.
3. **Ephemeral virtualenv for `pytest`:** `vllm-services` creates an isolated virtualenv inside
   `$RUNNER_TEMP/pytest-venv` (which the runner wipes after every job) rather than running
   `pip install` against the host or toolcache Python.
4. **Conditional `setup-python`:** `actions/setup-python@v5` runs only when
   `runner.environment == 'github-hosted'`, avoiding distro-specific binary downloads into
   `_work/_tool` on self-hosted Linux hosts that already provide Python 3.12.

### Self-Hosted Runner Setup (`~/fastrsi-lab/actions-runner/`)

Install and register the official GitHub Actions runner outside the git checkout under
`~/fastrsi-lab/actions-runner/` with a neutral runner name (`fastrsi-cpu-runner-01`):

```bash
mkdir -p ~/fastrsi-lab/actions-runner && cd ~/fastrsi-lab/actions-runner
curl -fsSL -o actions-runner.tar.gz \
  https://github.com/actions/runner/releases/download/v2.338.2/actions-runner-linux-x64-2.338.2.tar.gz
echo "baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40  actions-runner.tar.gz" | sha256sum -c -
tar xzf actions-runner.tar.gz && rm -f actions-runner.tar.gz

REG_TOKEN=$(gh api -X POST repos/GindaChen/fastrsi/actions/runners/registration-token --jq '.token')
./config.sh --unattended \
  --url https://github.com/GindaChen/fastrsi \
  --token "$REG_TOKEN" \
  --name fastrsi-cpu-runner-01 \
  --labels self-hosted,linux,x64,fastrsi-cpu \
  --work _work \
  --replace
unset REG_TOKEN

cat <<'EOF' > ~/fastrsi-lab/actions-runner/.env
PYTHONNOUSERSITE=1
CUDA_VISIBLE_DEVICES=
EOF

tmux new-session -d -s fastrsi-gh-runner \
  "cd ~/fastrsi-lab/actions-runner && ./run.sh > ~/fastrsi-lab/actions-runner/runner.stdout.log 2>&1"
gh api repos/GindaChen/fastrsi/actions/runners --jq '.runners[] | {name, status, busy, labels: [.labels[].name]}'
```

---

## Mode 2: Local Commit-Status CI (`statuses/<sha>`)

Self-hosted runner jobs (`FASTRSI_CI_RUNNER=self-hosted`) dispatch and execute natively on GitHub
Actions at `$0/min` even when GitHub-hosted `ubuntu-latest` jobs are blocked by an account
spending-limit or payment hold (because GitHub enforces hosted-minute billing checks only on
GitHub-hosted runner labels). However, Mode 2 is needed whenever:

- open pull requests were branched before `vars.FASTRSI_CI_RUNNER` was added to
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
2. **Run the local CI verifier (`~/fastrsi-lab/ci/run_local_ci.py`)** against one or more pull
   requests:
   ```bash
   python3 ~/fastrsi-lab/ci/run_local_ci.py <pr-number> [<pr-number> ...]
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
   requests include the `FASTRSI_CI_RUNNER` workflow routing:
   ```bash
   gh workflow enable cpu-tests.yml
   gh workflow enable pr-rules.yml
   ```
