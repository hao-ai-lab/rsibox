# 0001: Restart around a skill and per-engine live labs

Date: 2026-11-04

## Decision

- The core of FastRSI is agent-facing text: one skill (`core/livelab_overview.md`) and two
  subagent prompts (`core/toolsmith_offline.md`, `core/toolsmith_online.md`).
- Each engine gets its own live lab under `repos/<engine>/livelab/`. The interface is fixed
  (`template/livelab/INTERFACE.md`); the implementation is per engine.
- Trust comes from conformance, not shared code: a lab's receipts count only after it passes
  `template/livelab/conformance/`. Receipts are written by one shared writer that agents do not edit.
- Labs are bootstrapped from `template/livelab/` by `toolsmith_online`.

## Why

- v0 handled each engine with a hand-written plugin, and agents in our FastVideo A/B
  (Oct 20) built their own keep-alive and replay tools anyway.
- Agent-built checks can be wrong (a v0 install reported success while changing nothing), so the
  receipt path must stay outside agent control.

## What happened to v0

The Python package, vLLM and SGLang plugins, conformance suite and receipts are on the `legacy/v0`
branch. They are the source for the template stub and the conformance port.
