# Live lab template

What `toolsmith_online` copies to bootstrap a lab for a new engine.

- `INTERFACE.md`: the fixed verbs and receipt format (version 0). `INTERFACE-v1.md` is the
  draft of version 1, on top of the transaction kernel; it is not adopted yet.
- `livelab.py`: a working stub on a toy CPU engine (to port from `legacy/v0`:
  `src/fastrsi/core.py`, `agent.py`, `engines/toy.py`).
- `conformance/`: the cases c0-c10 every lab must pass before its receipts count, runnable on
  a toy lab: `python3 -m conformance --lab toy`. See `conformance/README.md`.
- `kernel/`: the engine-agnostic transaction kernel (locate, swap, journal, findings, witness,
  receipt, coordinator, session) with golden tests on a toy engine. See `kernel/README.md`.

`python3 -m unittest` from here runs the kernel's golden tests and the conformance cases.
