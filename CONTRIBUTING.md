# Contributing to rsibox

Thank you for your interest in contributing to **rsibox**!

## Getting Started

`rsibox` requires Python 3.11+ and runs its core conformance and unit test suites on CPU with standard-library dependencies only.

```sh
git clone https://github.com/hao-ai-lab/rsibox.git
cd rsibox
python3 -m venv .venv && . .venv/bin/activate
pip install -e .

# Verify your local environment and run the toy sandbox verification suite
# (checks repeatability, live execution proof, fresh-process parity, safe rollback, and fault recovery)
python3 -m rsibox doctor
python3 -m rsibox conformance toy
```

## Running Tests

Before opening a pull request, run the CPU test suites relevant to your change (overview in [`docs/concepts.md`](docs/concepts.md), full CI matrix in [`.github/workflows/cpu-tests.yml`](.github/workflows/cpu-tests.yml)):

```sh
python3 -m unittest discover -s core/livelab/kernel -t core/livelab
python3 -m unittest discover -s core/livelab/rsibox -t core/livelab
(cd core/livelab && python3 -m unittest discover -s conformance -t .)
python3 scripts/verify_facts.py --strict
```

## Pull Request Conventions

- **Title format:** `<type>(<scope>): <summary>` (for example `fix(vllm): handle empty patch in live swap`). Valid types: `feat`, `fix`, `perf`, `docs`, `test`, `ci`, `refactor`, `chore`.
- **Description:** Fill out `.github/pull_request_template.md`, including what you verified and what you did not verify.
- **No environment leaks:** Never commit internal IPs, private hostnames, local user paths, credentials, or raw cluster logs (CI runs `.github/scripts/leak-check.sh`).
- **Full contributor & agent rules:** See [`AGENTS.md`](AGENTS.md) for the complete architecture boundaries, loop record requirements when changing `repos/<engine>/`, and testing standards.

## Adding a New Engine

Start from the scaffold in [`template/`](template/) and review [`core/livelab/INTERFACE.md`](core/livelab/INTERFACE.md) and [`core/livelab/conformance/README.md`](core/livelab/conformance/README.md).
