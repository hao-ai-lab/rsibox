---
name: formal
description: Run bounded TLC verification, mutant/witness checks, spec pin/anchor/assumption gates, and Mermaid trace rendering across RSIBox TLA+ models.
---

# Formal TLA+ Verification Gate

1. Verify spec integrity, symbol anchors, and non-vacuous assumptions:
   - `python3 core/livelab/formal/gates/check_spec_pins.py`
   - `python3 core/livelab/formal/gates/check_anchors.py`
   - `python3 core/livelab/formal/gates/assumptions_check.py`
2. Run TLC across all 9 models, `Bug*` mutants, and `Never*` witnesses:
   - `python3 core/livelab/formal/run_formal.py --bootstrap --jar /tmp/tla2tools-rsibox.jar --output /tmp/formal-runs`
3. Render any violation or witness `trace.json` as Mermaid:
   - `python3 core/livelab/formal/gates/render_mermaid.py <trace.json>`
4. When intentionally updating a `.tla` spec, refresh `SPEC_PINS.sha256` with `--update` and keep `anchors.tsv` aligned.
