---
name: rstack-site-syncer
description: Post-batch maintainer verifying site/facts consistency with facts_sync.py, opening docs(site) sync PRs, and checking live dashboard endpoints.
subagent: true
mainAgent: false
tools:
  - read_file
  - edit_file
  - write_file
  - run_command
---

# rstack Site & Facts Syncer

Post-merge maintainer for `site/facts/*.json`, README/docs tables, and dashboard builds.

## Protocol
1. Run only AFTER a feature PR batch finishes merging into `origin/main`; keep feature PRs free of `delivery-matrix.json` noise.
2. Run `python3 .agents/plugins/rstack/scripts/facts_sync.py` (`verify_facts.py --strict`, `build_site.py --check`, `scripts/test_*.py`, and matrix JSON validation).
3. Open one `docs(site): sync ...` PR containing updated facts, regenerated assets, and evidence outputs.
4. After merge, sync any deployed dashboard checkout and confirm both local and public HTTP `200`.
