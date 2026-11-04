"""Golden cases for the vLLM live lab extension. CPU only, no vLLM needed.

The module under test mimics vLLM's shape: a function decorated with a `functools.wraps`
decorator whose wrapper closes over the inner function, plus a "registered op" that holds the
wrapper object, the way `direct_register_custom_op` does.

Run: python3 repos/vllm/livelab/test_lab_ext.py -v
"""
import functools, sys, types, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastrsi_lab_ext import LabExtension  # noqa: E402

BASE = '''
import functools
def eager_break(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        return fn(*a, **k)
    return wrapper

@eager_break
def indexer(x):
    return x + 1
'''
PATCHED = '''
import functools
from math import sqrt
def eager_break(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        return fn(*a, **k)
    return wrapper

@eager_break
def indexer(x):
    return int(sqrt(x)) * 100
'''


def fresh_module():
    mod = types.ModuleType("toy_engine")
    mod.__file__ = "toy_engine.py"
    exec(BASE, mod.__dict__)
    sys.modules["toy_engine"] = mod
    registered_op = mod.indexer          # what the op registry keeps: the wrapper object
    return mod, registered_op


class Install(unittest.TestCase):
    def setUp(self):
        self.mod, self.op = fresh_module()
        self.lab = LabExtension()

    def test_install_changes_what_the_registered_op_runs(self):
        """The op registry holds the wrapper; after install it must run the new code."""
        self.assertEqual(self.op(16), 17)
        r = self.lab.fastrsi_install("toy_engine", "indexer", PATCHED)
        self.assertNotEqual(r["base_hash"], r["new_hash"])
        self.assertEqual(self.op(16), 400)

    def test_witness_counts_only_new_code(self):
        self.op(1)
        self.assertEqual(self.lab.fastrsi_witness("toy_engine")["hits"], {})
        self.lab.fastrsi_install("toy_engine", "indexer", PATCHED)
        self.op(4); self.op(9)
        self.assertEqual(self.lab.fastrsi_witness("toy_engine")["hits"], {"toy_engine.indexer": 2})

    def test_new_import_is_added(self):
        """The patch imports `sqrt`, which the running module never had."""
        self.assertNotIn("sqrt", self.mod.__dict__)
        self.lab.fastrsi_install("toy_engine", "indexer", PATCHED)
        self.assertIn("sqrt", self.mod.__dict__)

    def test_uninstall_restores_base(self):
        self.lab.fastrsi_install("toy_engine", "indexer", PATCHED)
        self.lab.fastrsi_uninstall("toy_engine", "indexer")
        self.assertEqual(self.op(16), 17)

    def test_swapping_the_wrapper_would_be_a_silent_noop(self):
        """Why the rule exists: rebinding the module attribute leaves the op on old code."""
        ns = {}
        exec(PATCHED, ns)
        self.mod.indexer = ns["indexer"]          # the naive "install"
        self.assertEqual(self.op(16), 17)          # registered op still runs base code


if __name__ == "__main__":
    unittest.main()
