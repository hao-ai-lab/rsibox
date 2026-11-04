# Live lab interface, version 0 (draft)

Every `repos/<engine>/livelab/` implements these verbs. How it implements them is up to the engine.
Changing this file is a breaking change: bump the version.

| Verb | Does | Returns |
|---|---|---|
| `start` | Starts the engine with the lab loaded | Engine id |
| `stop` | Stops it | |
| `install <patch>` | Installs a patch; every function declares its group | Install id, or refusal with a reason |
| `undo` | Restores the code from before the last install | Proof that the base code is back |
| `status` | `base`, `edited` or `poisoned`, and whether each edited function has run | |
| `witness` | Which processes ran each new function | |
| `capture <fn>` | Records real inputs to a function for offline replay | Capture path |
| `profile <group> <steps>` | Times or traces a function group for N steps | Trace path |
| `receipt` | The record of the last command, in the fixed format | Receipt path |

## Receipt

Written only by the shared receipt writer, never by engine code or agents. Carries the interface
version, engine, commit, patch hash, groups, witness result, and any measurement.

<!-- TODO: exact schema, and the shared receipt writer + state machine each lab imports. -->
