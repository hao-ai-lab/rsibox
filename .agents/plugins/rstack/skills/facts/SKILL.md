---
name: facts
description: Synchronize site/facts JSON files, README tables, and live dashboard checkouts cleanly in one post-batch docs(site) PR after feature merges.
---

# Post-Batch Facts & Dashboard Sync

1. Wait until the feature PR batch is merged into `origin/main`; keep feature PRs free of `delivery-matrix.json` edits.
2. Run `python3 .agents/plugins/rstack/scripts/facts_sync.py` (`verify_facts.py --strict`, `build_site.py --check`, `scripts/test_*.py`, and matrix JSON check).
3. Open a single `docs(site): sync ...` PR updating `site/facts/*.json`, README/docs counts, and rendered assets.
4. After merge, pull `origin/main` on any deployed dashboard host and verify HTTP `200` on local and public endpoints.
