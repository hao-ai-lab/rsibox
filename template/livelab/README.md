# Live lab template

The blueprint every engine's live lab follows. No engine code lives here: each engine's lab,
its kernel and its adapter are in `repos/<engine>/livelab/`.

- `INTERFACE.md`: the fixed verbs and receipt format (version 0). `INTERFACE-v1.md` is the
  draft of version 1, on top of the transaction kernel; it is not adopted yet.
- `conformance/`: the cases c0-c10 every lab must pass before its receipts count:
  `python3 -m conformance --lab module:Class`. See `conformance/README.md`.

`python3 -m unittest discover -s conformance -t .` from here tests the cases themselves against
small fake labs: a lab that lacks a property must fail the case that names it.
