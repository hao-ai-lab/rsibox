"""Golden tests of the JIT rebuild (jit.py through swap, the participant and the session).

CPU, standard library only: `TRITON` below is a stand-in runtime with Triton's shape where
the rebuild depends on it (3.5 / 3.8 `runtime/jit.py`, `autotuner.py`): the source text
`_src` from `inspect.getsourcelines`, `_unsafe_update_src`, `hash` memoising `cache_key`
(which folds in callee keys and merges callee `used_global_vals` before reading the callee's
key, as 3.5 does), compiled kernels in `device_caches` keyed by argument types (never by
text), a disk cache keyed by `cache_key`, `pre_run_hooks` run on every launch, callees inlined
at compile, and `Autotuner.cache`. "Compiling" builds a Python function from `_src`, so what
a launch runs is what the text said at compile time. The real-Triton case at the end runs only
when `FASTRSI_TEST_GPU=1` is set with triton and a GPU.

Run from the repo root:  python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v
"""
import importlib.util
import itertools
import os
import sys
import unittest

from . import coordinator as C
from . import jit as jit_module
from . import journal as journal_module
from . import locate as L
from . import session as SS
from .test_detectors import temp_modules
from .test_locate import at

_n = itertools.count()

TRITON = '''import ast, collections, hashlib, inspect, re, textwrap

DISK = {}         # cache_key -> compiled kernel (Triton's on-disk cache)
COMPILES = []     # names of the kernels compiled, in order
_MISSING = object()


class constexpr:
    pass


class Config:
    def __init__(self, kwargs):
        self.kwargs = kwargs


def _resolve(node, scope):
    if isinstance(node, ast.Name):
        return scope.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve(node.value, scope)
        return getattr(base, node.attr, None) if base is not None else None
    return None


class JITFunction:
    def __init__(self, fn):
        self.fn, self.signature = fn, inspect.signature(fn)
        self.raw_src, self.starting_line_number = inspect.getsourcelines(fn)
        src = textwrap.dedent("".join(self.raw_src))
        self._src = src[re.search(r"^def\\s+\\w+\\s*\\(", src, re.MULTILINE).start():]
        self.hash, self.used_global_vals, self.pre_run_hooks = None, {}, []
        self.device_caches = collections.defaultdict(lambda: ({}, {}, "cpu", "stand-in"))
        self.__globals__, self.__name__, self.__module__ = fn.__globals__, fn.__name__, fn.__module__

    def parse(self):
        return ast.parse(self._src)

    def _unsafe_update_src(self, src):
        self.hash, self._src = None, src

    def _names(self):
        return [n for n in ast.walk(self.parse())
                if isinstance(n, (ast.Name, ast.Attribute)) and isinstance(n.ctx, ast.Load)]

    @property
    def cache_key(self):
        if self.hash is not None:
            return self.hash
        g, h, used = self.__globals__, hashlib.sha256(self._src.encode()), {}
        for n in self._names():
            v = _resolve(n, g)
            if isinstance(n, ast.Name) and isinstance(v, (int, float)):
                used[(n.id, id(g))] = (v, g)
            elif isinstance(v, JITFunction) and v is not self:
                used.update(v.used_global_vals)        # merged before the callee's key (3.5)
                h.update(v.cache_key.encode())
        self.used_global_vals = used
        self.hash = h.hexdigest() + str(self.starting_line_number)
        return self.hash

    def _build(self):
        """What the device runs: a function made from _src now, callees inlined."""
        assert "__fastrsi_hit__" not in self._src, "the witness call reached the kernel text"
        me, inl = self, {}

        class Inline(ast.NodeTransformer):
            def _sub(self, n):
                v = _resolve(n, me.__globals__)
                if isinstance(v, JITFunction) and v is not me and isinstance(n.ctx, ast.Load):
                    name = f"__inline{len(inl)}"
                    inl[name] = v._build()
                    return ast.copy_location(ast.Name(name, ast.Load()), n)
                return self.generic_visit(n)
            visit_Name = visit_Attribute = _sub

        tree = ast.fix_missing_locations(Inline().visit(self.parse()))
        ns = {}
        exec(compile(tree, f"<kernel {self.__name__}>", "exec"), {**self.__globals__, **inl}, ns)
        return ns[self.fn.__name__]

    def run(self, *args, grid=None, warmup=False, **kw):
        for hook in self.pre_run_hooks:
            hook(*args, **kw)
        cache = self.device_caches["cpu"][0]
        spec = tuple(type(a).__name__ for a in args)     # never the source text
        kernel = cache.get(spec)
        if kernel is None:
            kernel = DISK.get(self.cache_key)
            if kernel is None:
                kernel = DISK[self.cache_key] = self._build()
                COMPILES.append(self.__name__)
            cache[spec] = kernel
        for (name, _), (val, g) in self.used_global_vals.items():
            if g.get(name, _MISSING) != val:
                raise RuntimeError(f"Global variable {name} has changed since compile")
        return kernel(*args, **kw)

    def __getitem__(self, grid):
        return lambda *a, **k: self.run(*a, grid=grid, **k)


class Autotuner:
    def __init__(self, fn, configs, key):
        self.fn, self.configs, self.keys, self.cache, self.benchmarks = fn, configs, key, {}, 0

    def run(self, *args, **kw):
        names = list(self.fn.signature.parameters)
        key = tuple(dict(zip(names, args))[k] for k in self.keys)
        if key not in self.cache:
            self.benchmarks += 1
            self.cache[key] = self.configs[0]
        return self.fn.run(*args, **{**kw, **self.cache[key].kwargs})

    def __getitem__(self, grid):
        return lambda *a, **k: self.run(*a, grid=grid, **k)


def jit(fn):
    return JITFunction(fn)


def autotune(configs, key):
    return lambda fn: Autotuner(fn, configs, key)
'''

LEAF = '''import {t} as triton


@triton.autotune(configs=[triton.Config({{"BLOCK": 4}})], key=["n"])
@triton.jit
def scale_kernel(x, n, BLOCK: triton.constexpr):
    return [v * 2 for v in x[:n]]


def launch(x):
    return scale_kernel[(1,)](x, len(x))
'''

ACT = '''import {t} as triton

SCALE = 2


@triton.jit
def act_fn(v):
    return v * SCALE
'''

MLP = '''import {t} as triton
import {act} as acts


@triton.autotune(configs=[triton.Config({{}})], key=["n"])
@triton.jit
def mlp_kernel(x, n):
    return [acts.act_fn(v) + 1 for v in x[:n]]


def launch(x):
    return mlp_kernel[(1,)](x, len(x))
'''


def lab(test, files):
    """Import the stand-in runtime and the kernel modules; a session over one participant."""
    names = {k: f"kt_jit_{k}{next(_n)}" for k in ("t", *files)}
    texts = {"t": TRITON, **{k: v.format(**names) for k, v in files.items()}}
    mods = dict(zip(texts, temp_modules(test, {names[k]: texts[k] for k in texts})))
    p = C.LocalParticipant("rank0")
    sess = SS.Session(C.Coordinator([p], phase2_deadline_s=None),
                      [names[k] for k in files])
    return names, texts, mods, p, sess


def install(sess, module, source, accept=()):
    return sess.install({"modules": {module: {"source": source}}, "accept": list(accept)})


def findings(res):
    return (res.get("receipt") or {}).get("findings") or res.get("findings", [])


def checks(res, advisory=None):
    fs = findings(res)
    return sorted(f["check"] for f in fs if advisory is None or bool(f.get("advisory")) == advisory)


class JitRebuildTest(unittest.TestCase):

    def test_leaf_autotuned_kernel_edit_chained_edit_and_undo(self):
        """Input: `@triton.autotune @triton.jit scale_kernel` (v * 2), launched once. Edit 1
        (v * 3), then edit 2 on top (v * 5), then undo both. Expected: each install commits
        with an advisory jit_rebuild; each launch after it compiles the new text once, re-tunes
        once and proves the kernel's witness; undo puts the exact prior compiled kernel and
        tuning choice back (no compile, no benchmark); verify() passes at base, after each
        edit and after each undo. Fails without the rebuild: blocking jit_target, nothing
        changes on the device. Not: a real GPU (RealTritonTest)."""
        names, texts, m, p, sess = lab(self, {"leaf": LEAF})
        tri, leaf, mod = m["t"], m["leaf"], names["leaf"]
        key = L.key_of(mod, "scale_kernel", at(texts["leaf"], "@triton.autotune"))
        self.assertEqual((leaf.launch([1, 2, 3]), tri.COMPILES), ([2, 4, 6], ["scale_kernel"]))
        base_kernel = leaf.scale_kernel.fn.device_caches["cpu"][0][("list", "int")]
        edit1 = texts["leaf"].replace("v * 2", "v * 3")
        r1 = install(sess, mod, edit1)
        self.assertEqual((r1["outcome"], checks(r1, advisory=True)),
                         ("committed", ["jit_rebuild"]), r1)
        self.assertTrue(sess.verify()["ok"])
        self.assertEqual(leaf.launch([1, 2, 3]), [3, 6, 9])
        pr = sess.co.prove(r1["txn"])
        self.assertEqual((pr["ok"], pr["receipt"]["witness"]["rank0"]), (True, {key: 1}))
        r2 = install(sess, mod, edit1.replace("v * 3", "v * 5"))
        self.assertEqual(r2["outcome"], "committed", r2)
        self.assertEqual((leaf.launch([1, 2, 3]), sess.co.prove(r2["txn"])["ok"]),
                         ([5, 10, 15], True))
        self.assertEqual((len(tri.COMPILES), leaf.scale_kernel.benchmarks), (3, 3))
        self.assertTrue(sess.uninstall()["ok"])                          # undo edit 2
        self.assertEqual((leaf.launch([1, 2, 3]), len(tri.COMPILES)), ([3, 6, 9], 3))
        self.assertTrue(sess.uninstall()["ok"])                          # undo edit 1
        jf = leaf.scale_kernel.fn
        self.assertEqual((leaf.launch([1, 2, 3]), len(tri.COMPILES), leaf.scale_kernel.benchmarks),
                         ([2, 4, 6], 3, 3))
        self.assertIs(jf.device_caches["cpu"][0][("list", "int")], base_kernel)
        self.assertEqual((sess.state(), jf.pre_run_hooks, p.at_base()["ok"]), ("base", [], True))
        self.assertNotIn("v * 5", jf._src)

    def test_helper_edit_resets_caller_in_another_module(self):
        """Input: `act_fn` (v * SCALE, SCALE = 2) in module A, inlined into the autotuned
        `mlp_kernel` of module B; both launched. The edit (module A only) makes act_fn
        `v + 10`, dropping SCALE; then SCALE is rebound to 7. Expected: commit; the outer
        launch runs the new helper (v + 11) without a stale-global error, and proves act_fn's
        witness through mlp_kernel's launch hooks; undo restores both kernels with no
        compile. Fails without resetting the caller: mlp_kernel keeps (or reloads from the
        disk cache) the old binary; without clearing used_global_vals: "Global variable
        SCALE has changed". Not: callers held only inside classes or closures."""
        names, texts, m, p, sess = lab(self, {"act": ACT, "mlp": MLP})
        tri, acts, mlp = m["t"], m["act"], m["mlp"]
        self.assertEqual(mlp.launch([1, 2]), [3, 5])
        n0 = len(tri.COMPILES)
        r = install(sess, names["act"], texts["act"].replace("v * SCALE", "v + 10"))
        self.assertEqual(r["outcome"], "committed", r)
        rebuild = [f for f in findings(r) if f["check"] == "jit_rebuild"][0]
        self.assertIn(f"{names['mlp']}.mlp_kernel", rebuild["observed"])
        acts.SCALE = 7
        self.assertEqual(mlp.launch([1, 2]), [12, 13])
        key = L.key_of(names["act"], "act_fn", at(texts["act"], "@triton.jit"))
        pr = sess.co.prove(r["txn"])
        self.assertEqual((pr["ok"], pr["receipt"]["witness"]["rank0"]), (True, {key: 1}))
        acts.SCALE = 2
        n1 = len(tri.COMPILES)
        self.assertTrue(sess.uninstall()["ok"])
        self.assertEqual((mlp.launch([1, 2]), len(tri.COMPILES), n1 > n0), ([3, 5], n1, True))
        self.assertEqual((mlp.mlp_kernel.fn.pre_run_hooks, sess.state()), ([], "base"))

    def test_zero_launch_and_capture_only_are_not_proven(self):
        """Input: the leaf edit committed twice: once with no launch, once launched only
        inside `with witness.capture():` (a CUDA graph recapture; replays run no hooks).
        Expected: prove is not ok, zero_witness both times; the second with capture calls > 0
        and route R1. Pain point: a graph captured before the edit keeps replaying the old
        kernel while the edit reads as proven."""
        names, texts, m, p, sess = lab(self, {"leaf": LEAF})
        mod = names["leaf"]
        m["leaf"].launch([1])
        for captured in (False, True):
            r = install(sess, mod, texts["leaf"].replace("v * 2", "v * 4"))
            self.assertEqual(r["outcome"], "committed", r)
            if captured:
                with p.witness.capture():
                    m["leaf"].launch([1])
            pr = sess.co.prove(r["txn"])
            f = pr["receipt"]["findings"]
            self.assertEqual((pr["ok"], [x["check"] for x in f]), (False, ["zero_witness"]))
            self.assertEqual(f[0].get("route"), "R1" if captured else None)
            self.assertTrue(sess.uninstall()["ok"])

    def test_signature_or_decorator_change_is_refused(self):
        """Input: edits of the leaf kernel that add a parameter, or change the autotune key.
        Expected: abort with a blocking jit_target saying what changed; nothing mutated (the
        kernel text and the participant's journal as before). Pain point: Triton fixed the
        parameter binding and the tuning configs when it built the objects; an in-place
        rebuild would launch with the old ones."""
        names, texts, m, p, sess = lab(self, {"leaf": LEAF})
        jf = m["leaf"].scale_kernel.fn
        src = jf._src
        for new, word in ((texts["leaf"].replace("n, BLOCK", "n, extra, BLOCK"), "parameters"),
                          (texts["leaf"].replace('key=["n"]', 'key=["x"]'), "decorators")):
            r = install(sess, names["leaf"], new, accept=["outside_changes"])
            fs = [f for f in findings(r) if f["check"] == "jit_target"]
            self.assertEqual((r["outcome"], len(fs), fs[0].get("advisory", False) if fs else None),
                             ("aborted", 1, False), r)
            self.assertIn(word, fs[0]["observed"])
            self.assertEqual((jf._src, p.journal.entries, sess.state()), (src, [], "base"))

    def test_added_jit_helper_is_built_from_the_patch_text_and_removed_on_undo(self):
        """Input: a module with `@triton.jit step` called by `scale_kernel`; one transaction
        adds `@triton.jit def bump(v): return v + 100` (not in the file on disk) and edits
        both `step` and `scale_kernel` to call `bump`. Expected: commit (getsourcelines reads
        the patch text: no OSError, the right lines); one launch runs bump twice inlined and
        counts each of the three witness keys once (no duplicate caller hook); undo removes
        bump and the base kernel runs. Fails without seeding linecache: getsourcelines reads
        the file's lines at bump's line number."""
        chain = LEAF.replace("\n\n@triton.autotune", "\n\n@triton.jit\ndef step(v):\n"
                             "    return v * 2\n\n\n@triton.autotune").replace("[v * 2", "[step(v)")
        names, texts, m, p, sess = lab(self, {"leaf": chain})
        leaf = m["leaf"]
        self.assertEqual(leaf.launch([1, 2]), [2, 4])
        new = texts["leaf"].replace(
            "\n\n@triton.jit\ndef step", "\n\n@triton.jit\ndef bump(v):\n    return v + 100\n"
            "\n\n@triton.jit\ndef step").replace("return v * 2", "return bump(v)").replace(
            "[step(v)", "[bump(step(v))")
        r = install(sess, names["leaf"], new)
        self.assertEqual(r["outcome"], "committed", r)
        self.assertEqual(leaf.bump._src, "def bump(v):\n    return v + 100\n")
        self.assertEqual(leaf.launch([1, 2]), [201, 202])
        pr = sess.co.prove(r["txn"])
        self.assertEqual((pr["ok"], sorted(pr["receipt"]["witness"]["rank0"].values())),
                         (True, [1, 1, 1]), pr)
        self.assertTrue(sess.uninstall()["ok"])
        self.assertEqual((hasattr(leaf, "bump"), leaf.launch([1, 2]), sess.state()),
                         (False, [2, 4], "base"))

    def test_helper_fn_edit_invalidates_outer_kernel_counts_witness_and_restores(self):
        """Input: `@triton.jit helper_fn` called by `@triton.jit outer_kernel`. Editing
        `helper_fn` discovers `outer_kernel` via `discover_triton_callers`, invalidates
        `outer_kernel`'s device cache, counts launches of `outer_kernel` toward
        `helper_fn`'s witness, and restores cleanly on undo."""
        helper_src = (
            "import {t} as triton\n\n"
            "@triton.jit\n"
            "def helper_fn(val):\n"
            "    return val * 3\n\n"
            "@triton.jit\n"
            "def outer_kernel(items, count):\n"
            "    return [helper_fn(item) + 1 for item in items[:count]]\n\n"
            "def launch(items):\n"
            "    return outer_kernel[(1,)](items, len(items))\n"
        )
        names, texts, mods, participant, session = lab(self, {"kern": helper_src})
        kern_mod = mods["kern"]
        mod_name = names["kern"]
        self.assertEqual(kern_mod.launch([2, 4]), [7, 13])
        self.assertTrue(bool(kern_mod.outer_kernel.device_caches["cpu"][0]))

        discovered = jit_module.discover_triton_callers(
            {"helper_fn"}, modules=[kern_mod]
        )
        self.assertEqual([caller.fn.__name__ for caller in discovered], ["outer_kernel"])

        edited_src = texts["kern"].replace("val * 3", "val * 10")
        install_res = install(session, mod_name, edited_src)
        self.assertEqual(install_res["outcome"], "committed", install_res)
        self.assertEqual(kern_mod.outer_kernel.device_caches, {})

        self.assertEqual(kern_mod.launch([2, 4]), [21, 41])
        helper_key = L.key_of(
            mod_name, "helper_fn", at(texts["kern"], "@triton.jit")
        )
        prove_res = session.co.prove(install_res["txn"])
        self.assertTrue(prove_res["ok"])
        self.assertEqual(prove_res["receipt"]["witness"]["rank0"][helper_key], 1)

        self.assertTrue(session.uninstall()["ok"])
        self.assertEqual(kern_mod.launch([2, 4]), [7, 13])
        self.assertEqual(kern_mod.outer_kernel.pre_run_hooks, [])
        self.assertTrue(participant.at_base()["ok"])

    def test_helper_imported_under_an_alias_resets_its_caller(self):
        """Input: module A defines `@triton.jit act_fn` (v * SCALE); module B imports it as
        `from A import act_fn as act` and its autotuned `mlp_kernel` calls `act(v)`; launched
        once. The edit (module A only) makes act_fn `v + 10`. Expected: jit_rebuild names
        B.mlp_kernel as a reset caller, the next launch runs the new helper (v + 11) and
        proves act_fn's witness; undo gives the base result. Pain point: the caller search
        filtered callers by the helper's own name in their text, so an alias was never
        searched: mlp_kernel kept its old compiled kernel and the proof read zero_witness.
        Not: a caller held only in a class or a closure."""
        aliased = MLP.replace("import {act} as acts", "from {act} import act_fn as act")
        names, texts, m, p, sess = lab(self, {"act": ACT, "mlp": aliased.replace(
            "acts.act_fn(v)", "act(v)")})
        mlp = m["mlp"]
        self.assertEqual(mlp.launch([1, 2]), [3, 5])
        r = install(sess, names["act"], texts["act"].replace("v * SCALE", "v + 10"))
        self.assertEqual(r["outcome"], "committed", r)
        rebuild = [f for f in findings(r) if f["check"] == "jit_rebuild"][0]
        self.assertIn(f"{names['mlp']}.mlp_kernel", rebuild["observed"])
        self.assertEqual(mlp.launch([1, 2]), [12, 13])
        key = L.key_of(names["act"], "act_fn", at(texts["act"], "@triton.jit"))
        pr = sess.co.prove(r["txn"])
        self.assertEqual((pr["ok"], pr["receipt"]["witness"]["rank0"]), (True, {key: 1}))
        self.assertTrue(sess.uninstall()["ok"])
        self.assertEqual((mlp.launch([1, 2]), sess.state()), ([3, 5], "base"))

    def test_added_kernel_calling_an_added_helper_proves_the_helper(self):
        """Input: a module with only `launch(x)`; one edit adds `@triton.jit added_helper`
        (v * 3), `@triton.jit added_kernel` calling it, and makes launch call added_kernel.
        Expected: commit; launch([2, 4]) = [6, 12]; the proof is ok with every witness key
        (helper, kernel, launch) at 1. Pain point: an added helper was hooked only into the
        rebuilt kernels whose text names it, never into another added kernel, so its witness
        read 0 (zero_witness) though every launch ran it inlined. Not: a real GPU."""
        base = "import {t} as triton\n\n\ndef launch(x):\n    return x\n"
        names, texts, m, p, sess = lab(self, {"add": base})
        new = texts["add"].replace(
            "\n\n\ndef launch(x):\n    return x\n",
            "\n\n\n@triton.jit\ndef added_helper(v):\n    return v * 3\n\n\n"
            "@triton.jit\ndef added_kernel(x, n):\n    return [added_helper(v) for v in x[:n]]\n"
            "\n\ndef launch(x):\n    return added_kernel[(1,)](x, len(x))\n")
        r = install(sess, names["add"], new)
        self.assertEqual(r["outcome"], "committed", r)
        self.assertEqual(m["add"].launch([2, 4]), [6, 12])
        pr = sess.co.prove(r["txn"])
        self.assertEqual((pr["ok"], sorted(pr["receipt"]["witness"]["rank0"].values())),
                         (True, [1, 1, 1]), pr["receipt"]["findings"])
        self.assertTrue(sess.uninstall()["ok"])
        self.assertEqual((m["add"].launch([2, 4]), sess.state()), ([2, 4], "base"))

    def test_selective_compiled_frame_invalidation_and_restore(self):
        """Input: a target callable carrying per-callable TorchDynamo cache attributes
        (`_torchdynamo_orig_callable`, `_torchdynamo_cache`, `_dynamo_cache_entries`).
        Expected: `invalidate_compiled_frames` clears the Dynamo attributes without
        disturbing `_dynamo_marked_constant`, and restores them cleanly on undo."""
        def original_fn(val):
            return val + 1

        def compiled_target(val):
            return val + 1

        compiled_target._torchdynamo_orig_callable = original_fn
        compiled_target._torchdynamo_cache = {"guard_0": "compiled_code_0"}
        compiled_target._dynamo_cache_entries = ["entry_a", "entry_b"]
        compiled_target._dynamo_marked_constant = True

        journal = journal_module.Journal("rank0")
        mark = journal.mark("frame_invalidation")
        result = jit_module.invalidate_compiled_frames([compiled_target], journal=journal)

        self.assertIn(compiled_target.__qualname__, result["invalidated"])
        self.assertFalse(hasattr(compiled_target, "_torchdynamo_orig_callable"))
        self.assertEqual(compiled_target._torchdynamo_cache, {})
        self.assertEqual(compiled_target._dynamo_cache_entries, [])
        self.assertTrue(compiled_target._dynamo_marked_constant)

        self.assertTrue(journal.undo_to(mark)["ok"])
        self.assertIs(compiled_target._torchdynamo_orig_callable, original_fn)
        self.assertEqual(
            compiled_target._torchdynamo_cache, {"guard_0": "compiled_code_0"}
        )
        self.assertEqual(
            compiled_target._dynamo_cache_entries, ["entry_a", "entry_b"]
        )


def _gpu():
    if os.environ.get("FASTRSI_TEST_GPU") != "1":
        return False
    if importlib.util.find_spec("triton") is None or importlib.util.find_spec("torch") is None:
        return False
    import torch
    return torch.cuda.is_available()


REAL = '''import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + i, tl.load(x_ptr + i, mask=i < n) + 1.0, mask=i < n)


def launch(x):
    import torch
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(x.numel(), 128),)](x, out, x.numel(), BLOCK=128)
    return out
'''


@unittest.skipUnless(_gpu(), "set FASTRSI_TEST_GPU=1 with triton and a GPU")
class RealTritonTest(unittest.TestCase):

    def test_real_kernel_edit_and_undo(self):
        """Input: a real @triton.jit kernel (x + 1) launched on the GPU; the edit makes it
        x + 2. Expected: commit, the launch gives x + 2 and proves the witness; undo gives
        x + 1 again from the restored compiled kernel (device cache entry identical)."""
        import torch
        name = f"kt_jit_real{next(_n)}"
        (mod,) = temp_modules(self, {name: REAL})
        x = torch.arange(1000, device="cuda", dtype=torch.float32)
        self.assertTrue(torch.equal(mod.launch(x), x + 1))
        cache = dict(next(iter(mod.add_kernel.device_caches.values()))[0])
        p = C.LocalParticipant("rank0")
        sess = SS.Session(C.Coordinator([p], phase2_deadline_s=None), [name])
        r = install(sess, name, REAL.replace("+ 1.0", "+ 2.0"))
        self.assertEqual(r["outcome"], "committed", r)
        self.assertTrue(torch.equal(mod.launch(x), x + 2))
        self.assertTrue(sess.co.prove(r["txn"])["ok"])
        self.assertTrue(sess.uninstall()["ok"])
        self.assertTrue(torch.equal(mod.launch(x), x + 1))
        now = next(iter(mod.add_kernel.device_caches.values()))[0]
        self.assertEqual({k: id(v) for k, v in now.items()}, {k: id(v) for k, v in cache.items()})


if __name__ == "__main__":
    sys.exit(unittest.main())
