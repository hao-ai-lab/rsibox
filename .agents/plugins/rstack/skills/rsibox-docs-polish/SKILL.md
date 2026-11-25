---
name: rsibox-docs-polish
description: Keep RSIBox public documentation crisp, accurate, and plain-English while enforcing docs_lint.py and facts_sync.py gates.
---

# Public Docs Polish & Lint Gate

1. Write plain-English prose in `README.md`, `CONTRIBUTING.md`, `docs/*.md`, and `core/livelab/**/README.md` — drop nested internal symbols, duplicate bullets, and redundant parenthetical artifacts while preserving exact measured numbers and `docs/engines.md` tables.
2. Keep `docs/concepts.md`, `docs/README.md`, and `README.md` aligned with real repository files and authoritative trace counts.
3. Run `python3 .agents/plugins/rstack/scripts/docs_lint.py` to verify zero broken relative links, zero broken `#anchor` links, and zero duplicate parenthetical artifacts.
4. Run `python3 .agents/plugins/rstack/scripts/facts_sync.py` to verify `verify_facts.py --strict`, `build_site.py --check`, `scripts/test_*.py`, and `site/facts/delivery-matrix.json`.
