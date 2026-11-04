# Docs

Docs mirror the repo tree, so each doc has exactly one obvious home.

```
docs/
  README.md        this index
  decisions/       one numbered file per decision: NNNN-short-name.md
  core/            how the skill and the toolsmith prompts work
  livelab/         the interface, receipts, conformance
  repos/<engine>/  engine-specific design notes
```

Rules:

- Make a folder only when its first doc arrives. No empty placeholders.
- A doc about one engine goes under `repos/<engine>/`, never at the top level.
- A decision that changes the layout or the interface gets a file in `decisions/`.
- Add every new doc to the index below.

## Index

- [decisions/0001-skill-first-restart.md](decisions/0001-skill-first-restart.md)
