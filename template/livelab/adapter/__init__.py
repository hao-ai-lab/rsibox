"""Minimal onboarding skeleton for `repos/<new-engine>/livelab/adapter/__init__.py`.

When copied to `repos/<new-engine>/livelab/adapter/__init__.py`, importing this package puts
`<repo>/core/livelab` (or `FASTRSI_LIVELAB` when set) on `sys.path` so `from kernel import ...`,
`import ray_participants`, and `import forksafe` resolve directly from the shared runtime in
`core/livelab/`.
"""
import os
import sys

# From repos/<new-engine>/livelab/adapter/__init__.py, four parent hops reach <repo>/.
# From template/livelab/adapter/__init__.py in-place, three parent hops reach <repo>/.
_HERE = os.path.dirname(os.path.realpath(__file__))
_CANDIDATES = (
    os.path.join(_HERE, "..", "..", "..", "..", "core", "livelab"),
    os.path.join(_HERE, "..", "..", "..", "core", "livelab"),
)
LIVELAB = os.path.abspath(
    os.environ.get("FASTRSI_LIVELAB")
    or next((candidate for candidate in _CANDIDATES if os.path.isdir(candidate)), _CANDIDATES[0])
)
if LIVELAB not in sys.path:
    sys.path.insert(0, LIVELAB)
