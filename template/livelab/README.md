# Live lab template

What `toolsmith_online` copies to bootstrap a lab for a new engine.

- `INTERFACE.md`: the fixed verbs and receipt format.
- `livelab.py`: a working stub on a toy CPU engine (to port from `legacy/v0`:
  `src/fastrsi/core.py`, `agent.py`, `engines/toy.py`).
- `conformance/`: the cases every lab must pass before its receipts count.
- `kernel/`: the engine-agnostic transaction kernel (locate, swap, journal, findings, witness,
  receipt, coordinator) with golden tests on a toy engine. See `kernel/README.md`.
