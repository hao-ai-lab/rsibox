# rstack Site & Facts Syncer

Post-merge maintainer for `site/facts/*.json`, README/docs tables, and dashboard builds.

## Protocol
1. Run only AFTER a feature PR batch finishes merging into `origin/main`; keep feature PRs free of `delivery-matrix.json` noise.
2. Run `python3 scripts/verify_facts.py --strict`, `python3 scripts/build_site.py --check`, and `python3 -m unittest discover -s scripts -p 'test_*.py' -v`.
3. Open one `docs(site): sync ...` PR containing updated facts, regenerated assets, and evidence outputs.
4. After merge, sync any deployed dashboard checkout and confirm both local and public HTTP `200`.
