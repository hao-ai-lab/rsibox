"""Golden tests of the Ray actor reach (`ray_participants`): edits to `@ray.remote` class
methods and to files loaded without a `sys.modules` entry, through the kernel's
`LocalParticipant` and `Coordinator`.

CPU, standard library only. A stand-in Ray (`fake_remote`, `FakeActor`) reproduces what real
Ray leaves in a process: the module holds an ActorClass handle, and the actor runs a copy of
the class whose methods read a globals dict of their own (cloudpickle by value). The last case
runs the same edit through real Ray actors when `ray` is installed and skips otherwise.

Run from the repo root:  python3 -m unittest discover -s core/livelab/kernel -t core/livelab -v
"""
import importlib
import importlib.util
import itertools
import marshal
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest

import ray_participants as RP

from . import coordinator as C

_n = itertools.count()
LIVELAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORKER = '''import {decorator_module}


def helper(x):
    return x + 1


@{decorator_module}.remote
class Worker:
    def __init__(self):
        self.k = 3

    def score(self):
        return helper(self.k)
'''
REWARD = '''def bonus(text):
    return 0.5


def compute_score(text):
    return len(text) + bonus(text)
'''


def fake_remote(cls):
    """What `@ray.remote` leaves in the module: a handle; the class sits in its metadata."""
    modified = type(cls.__name__, (cls,), {"__ray_actor_class__": cls,
                                            "__module__": cls.__module__})
    handle = type(f"ActorClass({cls.__name__})", (), {})()
    handle.__dict__["__ray_metadata__"] = types.SimpleNamespace(modified_class=modified)
    return handle


def by_value(cls):
    """The class as an actor process gets it: cloudpickle cannot pickle it by reference (the
    module's name now holds the handle), so it rebuilds the functions over a fresh globals dict
    with the module's dunders and the names they read."""
    module = vars(sys.modules[cls.__module__])
    names = set()

    def collect(code):
        names.update(code.co_names)
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                collect(const)
    globals_dict = {k: module[k] for k in ("__name__", "__file__", "__package__") if k in module}
    body = {}
    for key, value in vars(cls).items():
        if key in ("__dict__", "__weakref__"):
            continue
        if isinstance(value, types.FunctionType):
            collect(value.__code__)
            copy = types.FunctionType(value.__code__, globals_dict, value.__name__,
                                      value.__defaults__, value.__closure__)
            copy.__qualname__, copy.__kwdefaults__ = value.__qualname__, value.__kwdefaults__
            value = copy
        body[key] = value
    globals_dict.update({n: module[n] for n in names if n in module})
    return type(cls.__name__, cls.__bases__, body)


class FakeActor:
    """One actor: an instance of a by-value copy, reached through `__ray_call__.remote`."""

    def __init__(self, handle):
        original = vars(handle)["__ray_metadata__"].modified_class.__ray_actor_class__
        copy = by_value(original)
        modified = type(copy.__name__, (copy,), {"__ray_actor_class__": copy,
                                                 "__module__": copy.__module__})
        self.instance = modified()
        self.__ray_call__ = types.SimpleNamespace(
            remote=lambda fn, *args: fn(self.instance, *args))

    def call(self, method):
        return getattr(self.instance, method)()


def same_ref(ref, timeout_s):
    return ref


# a process that hosts no actor (the driver): `__ray_call__` runs fn(None, ...)
NO_ACTOR = types.SimpleNamespace(__ray_call__=types.SimpleNamespace(
    remote=lambda fn, *args: fn(None, *args)))


class RayParticipantsTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="fastrsi-ray-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        with open(os.path.join(self.dir, "fakeray.py"), "w") as handle:
            handle.write("from kernel.test_ray_participants import fake_remote as remote\n")
        sys.path.insert(0, self.dir)
        self.addCleanup(sys.path.remove, self.dir)

    def module(self, decorator_module="fakeray"):
        name = f"rp_worker_{next(_n)}"
        self.source = WORKER.format(decorator_module=decorator_module)
        with open(os.path.join(self.dir, name + ".py"), "w") as handle:
            handle.write(self.source)
        self.addCleanup(sys.modules.pop, name, None)
        return name, importlib.import_module(name)

    def edit(self, *pairs):
        text = self.source
        for old, new in pairs:
            self.assertEqual(text.count(old), 1, old)
            text = text.replace(old, new)
        return text

    def run_txn(self, participants, plan):
        coordinator = C.Coordinator(participants, phase2_deadline_s=None)
        return coordinator, coordinator.run(plan)

    def test_plain_participant_commits_but_the_actor_keeps_running_the_old_method(self):
        """The pain point. Input: an actor of `@ray.remote class Worker`; plan: score() returns
        helper(k) * 10. A plain LocalParticipant in the actor's process commits ok, yet the
        actor still returns 4, not 40: the kernel swapped the class in the module's metadata,
        which no actor runs. Proves why the view is needed; not: real Ray (last case)."""
        name, module = self.module()
        actor = FakeActor(module.Worker)
        local = C.LocalParticipant("plain")
        _, out = self.run_txn([local], {"modules": {name: {
            "source": self.edit(("return helper(self.k)", "return helper(self.k) * 10"))}}})
        self.assertEqual(out["outcome"], "committed")
        self.assertEqual(actor.call("score"), 4)

    def test_actor_runs_the_edited_method_and_undo_restores_its_code_object(self):
        """Input: the same actor and plan through RayActorParticipant. Expected: committed;
        score() returns 40; the copy's `#actor` key is witnessed (1 call) and the module's dead
        class raises no zero_witness; undo brings back 4 and the copy's exact code object.
        Fails without `actor_view` (score stays 4, as in the case above)."""
        name, module = self.module()
        actor = FakeActor(module.Worker)
        base_code = type(actor.instance).score.__code__
        participant = RP.RayActorParticipant("worker", actor, get=same_ref)
        coordinator, out = self.run_txn([participant], {"modules": {name: {
            "source": self.edit(("return helper(self.k)", "return helper(self.k) * 10"))}}})
        self.assertEqual(out["outcome"], "committed", out)
        self.assertEqual(actor.call("score"), 40)
        proof = coordinator.prove(out["txn"])
        counts = proof["receipt"]["witness"]["worker"]
        self.assertEqual([n for k, n in counts.items() if RP.ACTOR in k], [1], counts)
        self.assertTrue(proof["ok"], proof["receipt"]["findings"])
        self.assertEqual(coordinator.undo(out["txn"])["status"], "undone")
        self.assertEqual(actor.call("score"), 4)
        self.assertIs(type(actor.instance).score.__code__, base_code)

    def test_one_edit_to_a_module_helper_and_an_actor_method(self):
        """Input: one plan that changes helper() (module level, x + 2) and Worker.score (* 10).
        Expected: score() returns (3 + 2) * 10 = 50: helper is swapped in the module the copy
        reads it from, score in the copy; both keys witnessed. Not: edits to the class header
        (the kernel refuses those as outside changes)."""
        name, module = self.module()
        actor = FakeActor(module.Worker)
        participant = RP.RayActorParticipant("worker", actor, get=same_ref)
        coordinator, out = self.run_txn([participant], {"modules": {name: {"source": self.edit(
            ("return x + 1", "return x + 2"),
            ("return helper(self.k)", "return helper(self.k) * 10"))}}})
        self.assertEqual(out["outcome"], "committed", out)
        self.assertEqual(actor.call("score"), 50)
        proof = coordinator.prove(out["txn"])
        counts = proof["receipt"]["witness"]["worker"]
        self.assertEqual(counts[f"{name}:helper:4"], 1)
        self.assertEqual(counts[f"{name}{RP.ACTOR}:Worker.score:13"], 1)
        self.assertTrue(proof["ok"], proof["receipt"]["findings"])
        coordinator.undo(out["txn"])
        self.assertEqual(actor.call("score"), 4)

    def test_one_actor_that_cannot_take_the_edit_aborts_it_on_every_actor(self):
        """Input: two actors; the second runs a score() that differs from the file (someone
        patched it). Expected: the second votes no (base_mismatch), outcome aborted, and the
        first still returns 4. Then, with the second restored, the same plan commits on both
        (40, 40) and one undo returns both to 4."""
        name, module = self.module()
        first, second = FakeActor(module.Worker), FakeActor(module.Worker)
        copy = type(second.instance).__ray_actor_class__
        good = copy.score.__code__
        scratch = {}
        exec(compile(self.edit(("return helper(self.k)", "return helper(self.k) + 1")),
                     module.__file__, "exec"), scratch)
        copy.score.__code__ = RP.actor_class(scratch["Worker"]).score.__code__
        self.assertEqual(second.call("score"), 5)
        participants = [RP.RayActorParticipant(f"worker{i}", actor, get=same_ref)
                        for i, actor in enumerate((first, second))]
        plan = {"modules": {name: {
            "source": self.edit(("return helper(self.k)", "return helper(self.k) * 10"))}}}
        _, out = self.run_txn(participants, plan)
        self.assertEqual(out["outcome"], "aborted")
        checks = {f["check"] for f in out["receipt"]["findings"]}
        self.assertIn("base_mismatch", checks)
        self.assertEqual(first.call("score"), 4)
        copy.score.__code__ = good
        coordinator, out = self.run_txn(participants, plan)
        self.assertEqual(out["outcome"], "committed", out)
        self.assertEqual((first.call("score"), second.call("score")), (40, 40))
        coordinator.undo(out["txn"])
        self.assertEqual((first.call("score"), second.call("score")), (4, 4))

    def test_reward_file_loaded_without_a_sys_modules_entry(self):
        """Input: a reward file loaded the way verl loads a custom reward function
        (spec_from_file_location + exec_module, no sys.modules entry, only the function kept);
        plan: bonus() returns 2.0. Expected: a plain participant cannot even name the module
        (vote no); through `serve` with the plan's `file`, compute_score("abc") goes 3.5 ->
        5.0, and undo brings back 3.5. A file no process loaded is `unloaded` (ask before
        run: such a plan would commit and change nothing); a process without it abstains."""
        path = os.path.join(self.dir, f"reward_{next(_n)}.py")
        with open(path, "w") as handle:
            handle.write(REWARD)
        spec = importlib.util.spec_from_file_location("custom_module_x", path)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        compute_score = loaded.compute_score
        del loaded
        name = RP.file_module(path)
        self.addCleanup(sys.modules.pop, name, None)
        plan = {"modules": {name: {"file": path,
                                   "source": REWARD.replace("return 0.5", "return 2.0")}}}
        _, out = self.run_txn([C.LocalParticipant("plain")], plan)
        self.assertEqual(out["outcome"], "aborted")
        participant = RP.RayActorParticipant("reward", NO_ACTOR, get=same_ref)
        self.assertEqual(RP.unloaded([participant], plan), [])
        coordinator, out = self.run_txn([participant], plan)
        self.assertEqual(out["outcome"], "committed", out)
        self.assertEqual(compute_score("abc"), 5.0)
        self.assertTrue(coordinator.prove(out["txn"])["ok"])
        coordinator.undo(out["txn"])
        self.assertEqual(compute_score("abc"), 3.5)
        unloaded = os.path.join(self.dir, "never_loaded.py")
        with open(unloaded, "w") as handle:
            handle.write(REWARD)
        absent_plan = {"id": "t-absent", "modules": {
            RP.file_module(unloaded): {"file": unloaded, "source": REWARD}}}
        self.assertEqual(RP.unloaded([participant], absent_plan), [RP.file_module(unloaded)])
        reply = participant.prepare(absent_plan)
        self.assertEqual((reply["ok"], reply["absent"], reply["abstain"]), (True, True, True))
        self.assertEqual(participant.commit("t-absent")["state"], "committed")
        fp = participant.fingerprint([name, RP.file_module(unloaded),
                                      {"name": name, "file": path},
                                      {"name": RP.file_module(unloaded), "file": unloaded}])
        self.assertTrue(fp["ok"], fp)
        self.assertTrue(any(k.startswith(name + ":") for k in fp["code"]), fp)
        self.assertFalse(any(RP.file_module(unloaded) in k for k in fp["code"]), fp)

    def test_base_source_handles_disk_edited_file_and_preserves_chained_order(self):
        """Input: a module whose file on disk is edited BEFORE `prepare` (CLI edit-in-place
        workflow), with `base_source` carrying the original pre-edit snapshot; then a second
        chained edit on the same actor method.
        Expected: edit 1 diffs against `base_source` (not the overwritten disk file) and swaps
        `score` 4 -> 40; edit 2 diffs against edit 1 (not `__base__`) and swaps 40 -> 400;
        undo restores 40 then 4."""
        name, module = self.module()
        with open(module.__file__) as handle:
            base_src = handle.read()
        actor = FakeActor(module.Worker)
        participant = RP.RayActorParticipant("worker", actor, get=same_ref)
        v1 = self.edit(("return helper(self.k)", "return helper(self.k) * 10"))
        v2 = self.edit(("return helper(self.k)", "return helper(self.k) * 100"))
        # Overwrite the file on disk BEFORE calling prepare (simulates editing the repo file).
        with open(module.__file__, "w") as handle:
            handle.write(v1)
        coordinator = C.Coordinator([participant], phase2_deadline_s=None)
        out1 = coordinator.run({"modules": {name: {"source": v1, "base_source": base_src}}})
        self.assertEqual(out1["outcome"], "committed", out1)
        self.assertEqual(actor.call("score"), 40)
        # Second chained edit on disk + plan still carrying base_source must NOT shadow out1.
        with open(module.__file__, "w") as handle:
            handle.write(v2)
        out2 = coordinator.run({"modules": {name: {"source": v2, "base_source": base_src}}})
        self.assertEqual(out2["outcome"], "committed", out2)
        self.assertEqual(actor.call("score"), 400)
        self.assertEqual(coordinator.undo(out2["txn"])["status"], "undone")
        self.assertEqual(actor.call("score"), 40)
        self.assertEqual(coordinator.undo(out1["txn"])["status"], "undone")
        self.assertEqual(actor.call("score"), 4)

    def test_unloaded_fans_out_across_actors_in_parallel(self):
        """8 x 50 ms RayActorParticipant.loaded checks fan out concurrently (< 250 ms,
        peak_in_flight >= 4) and report only modules loaded by no actor."""
        name, module = self.module()
        actors = [FakeActor(module.Worker) for _ in range(8)]
        lock = threading.Lock()
        state = {"in_flight": 0, "peak_in_flight": 0}

        def slow_get(ref, _timeout_s):
            with lock:
                state["in_flight"] += 1
                state["peak_in_flight"] = max(state["peak_in_flight"], state["in_flight"])
            try:
                time.sleep(0.05)
                return ref
            finally:
                with lock:
                    state["in_flight"] -= 1

        participants = [RP.RayActorParticipant(f"w{i}", actor, get=slow_get)
                        for i, actor in enumerate(actors)]
        missing_mod = f"fastrsi_unloaded_test_{next(_n)}"
        plan = {"modules": {
            name: {"source": self.source},
            missing_mod: {"source": REWARD},
        }}
        t0 = time.monotonic()
        unloaded_mods = RP.unloaded(participants, plan)
        elapsed = time.monotonic() - t0
        self.assertEqual(unloaded_mods, [missing_mod])
        self.assertGreaterEqual(state["peak_in_flight"], 4)
        self.assertLess(elapsed, 0.25, f"RP.unloaded took {elapsed:.3f}s")

        missing_path = os.path.join(self.dir, f"missing_{next(_n)}.py")
        with open(missing_path, "w") as handle:
            handle.write(REWARD)
        file_mod = RP.file_module(missing_path)
        fast_participant = RP.RayActorParticipant("w0_file", actors[0], get=lambda ref, _: ref)
        file_unloaded = RP.unloaded([fast_participant], {"modules": {
            name: {"source": self.source},
            file_mod: {"file": missing_path, "source": REWARD},
        }})
        self.assertEqual(file_unloaded, [file_mod])

    @unittest.skipUnless(importlib.util.find_spec("ray"), "ray is not installed")
    def test_real_ray_actors_run_the_edit_and_undo(self):
        """Input: two real Ray actors (local cluster, 2 CPUs) of `@ray.remote class Worker`;
        plan: score() * 10 through RayActorParticipant and a Coordinator. Expected: both return
        40 after commit, the copy's key is witnessed in both processes, undo returns 4."""
        import ray
        name, module = self.module(decorator_module="ray")
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False,
                 runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join([LIVELAB, self.dir])}})
        self.addCleanup(ray.shutdown)
        actors = [module.Worker.remote() for _ in range(2)]
        participants = [RP.RayActorParticipant(f"worker{i}", actor, timeout_s=60)
                        for i, actor in enumerate(actors)]
        coordinator = C.Coordinator(participants, phase2_deadline_s=None)
        from . import session as SS
        session = SS.Session(coordinator, [name])
        self.assertEqual(session.state(), "base", session.poisoned())
        out = session.install({"modules": {name: {
            "source": self.edit(("return helper(self.k)", "return helper(self.k) * 10"))}}})
        self.assertEqual(out["outcome"], "committed", out)
        self.assertEqual(ray.get([a.score.remote() for a in actors]), [40, 40])
        proof = coordinator.prove(out["txn"])
        self.assertTrue(proof["ok"], proof["receipt"]["findings"])
        undo_res = session.uninstall()
        self.assertTrue(undo_res["ok"], undo_res)
        self.assertEqual(session.state(), "base", session.poisoned())
        self.assertEqual(ray.get([a.score.remote() for a in actors]), [4, 4])


class FileNamespaceNegativeCacheTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="fastrsi-negcache-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        RP._clear_absent_cache()
        self.addCleanup(RP._clear_absent_cache)
        self.scan_count = 0
        orig_all_objects = RP.L.all_objects

        def counting_all_objects():
            self.scan_count += 1
            return orig_all_objects()

        RP.L.all_objects = counting_all_objects
        self.addCleanup(setattr, RP.L, "all_objects", orig_all_objects)

    def _make_file(self, stem):
        path = os.path.realpath(os.path.join(self.dir, f"{stem}_{next(_n)}.py"))
        with open(path, "w") as handle:
            handle.write("def marker():\n    return 42\n")
        self.addCleanup(sys.modules.pop, RP.file_module(path), None)
        return path

    def test_repeat_miss_and_unrelated_code_load_avoid_rescan(self):
        """(a) Repeat miss scans only once; (f) compiling/executing an unrelated .py file does
        NOT evict the cached absent path."""
        path = self._make_file("neg_absent")
        self.assertEqual(RP.file_namespace(path), (None, None))
        self.assertEqual(self.scan_count, 1)
        self.assertEqual(RP.file_namespace(path), (None, None))
        self.assertEqual(self.scan_count, 1)

        other = self._make_file("neg_unrelated")
        ns = {}
        exec(compile("def other_fn():\n    return 1\n", other, "exec"), ns)
        self.assertEqual(RP.file_namespace(path), (None, None))
        self.assertEqual(self.scan_count, 1)

    def test_compile_exec_invalidates_and_finds_namespace(self):
        """(b) compile() + exec() for `path` invalidates the negative cache and finds it."""
        path = self._make_file("neg_compile")
        self.assertEqual(RP.file_namespace(path), (None, None))
        self.assertEqual(self.scan_count, 1)

        ns = {}
        exec(compile("def marker():\n    return 42\n", path, "exec"), ns)
        name, mod = RP.file_namespace(path)
        self.assertEqual(name, RP.file_module(path))
        self.assertIsNotNone(mod)
        self.assertEqual(mod.marker(), 42)
        self.assertEqual(self.scan_count, 2)

    def test_precompiled_exec_invalidates_via_exec_audit(self):
        """(c) Pre-compiled code object executed later invalidates via the `exec` audit event."""
        path = self._make_file("neg_precompiled")
        code_obj = compile("def marker():\n    return 99\n", path, "exec")
        self.assertEqual(RP.file_namespace(path), (None, None))
        self.assertEqual(self.scan_count, 1)

        ns = {}
        exec(code_obj, ns)
        name, mod = RP.file_namespace(path)
        self.assertEqual(name, RP.file_module(path))
        self.assertEqual(mod.marker(), 99)
        self.assertEqual(self.scan_count, 2)

    def test_code_replace_invalidates_via_code_new(self):
        """(d) code.replace(co_filename=path) invalidates via the `code.__new__` audit event."""
        path = self._make_file("neg_replace")
        other = self._make_file("neg_replace_src")
        base_code = compile("def marker():\n    return 7\n", other, "exec")
        self.assertEqual(RP.file_namespace(path), (None, None))
        self.assertEqual(self.scan_count, 1)
        self.assertIn(path, RP._ABSENT_FILES)

        # Replace co_filename on both module code and inner function code object.
        new_consts = tuple(
            c.replace(co_filename=path) if isinstance(c, types.CodeType) else c
            for c in base_code.co_consts
        )
        rebound = base_code.replace(co_filename=path, co_consts=new_consts)
        self.assertNotIn(path, RP._ABSENT_FILES)
        ns = {}
        exec(rebound, ns)
        name, mod = RP.file_namespace(path)
        self.assertEqual(name, RP.file_module(path))
        self.assertEqual(mod.marker(), 7)
        self.assertEqual(self.scan_count, 2)

    def test_marshal_loads_invalidates_and_finds_namespace(self):
        """(e) marshal.loads(marshal.dumps(code)) invalidates the cache before exec."""
        path = self._make_file("neg_marshal")
        raw = marshal.dumps(compile("def marker():\n    return 123\n", path, "exec"))
        self.assertEqual(RP.file_namespace(path), (None, None))
        self.assertEqual(self.scan_count, 1)
        self.assertIn(path, RP._ABSENT_FILES)

        loaded_code = marshal.loads(raw)
        self.assertNotIn(path, RP._ABSENT_FILES)
        ns = {}
        exec(loaded_code, ns)
        name, mod = RP.file_namespace(path)
        self.assertEqual(name, RP.file_module(path))
        self.assertEqual(mod.marker(), 123)
        self.assertEqual(self.scan_count, 2)

    def test_sys_modules_registration_hits_fast_path_without_rescan(self):
        """(g) Registering a module directly in sys.modules hits the fast path with 0 scans."""
        path = self._make_file("neg_sysmod")
        self.assertEqual(RP.file_namespace(path), (None, None))
        self.assertEqual(self.scan_count, 1)

        mod_name = f"fastrsi_direct_sysmod_{next(_n)}"
        mod = types.ModuleType(mod_name)
        mod.__file__ = path
        sys.modules[mod_name] = mod
        self.addCleanup(sys.modules.pop, mod_name, None)

        found_name, found_mod = RP.file_namespace(path)
        self.assertEqual((found_name, found_mod), (mod_name, mod))
        self.assertEqual(self.scan_count, 1)

    def test_ast_compile_preserves_negative_cache_and_exec_evicts_target(self):
        """(h) CPython `compile(ast.AST, filename, 'exec')` emits filename=None in PEP 578;
        it must NOT wipe unrelated cached misses, while `exec()` of the compiled target still
        evicts via `co_filename`, and `_ACTIVE_SCANS` still protects mid-scan races."""
        import ast
        path = self._make_file("neg_ast_target")
        unrelated = self._make_file("neg_ast_unrelated")
        self.assertEqual(RP.file_namespace(path), (None, None))
        self.assertEqual(self.scan_count, 1)

        # Compiling an AST for an unrelated file must not evict `path`.
        unrelated_code = compile(ast.parse("def other():\n    return 1\n"), unrelated, "exec", dont_inherit=True)
        self.assertIn(path, RP._ABSENT_FILES)
        self.assertEqual(RP.file_namespace(path), (None, None))
        self.assertEqual(self.scan_count, 1)
        unrelated_ns = {}
        exec(unrelated_code, unrelated_ns)
        self.assertIn(path, RP._ABSENT_FILES)

        # Compiling an AST for `path` keeps cache until `exec()`, which evicts via `co_filename`.
        target_code = compile(ast.parse("def marker():\n    return 314\n"), path, "exec", dont_inherit=True)
        self.assertIn(path, RP._ABSENT_FILES)
        target_ns = {}
        exec(target_code, target_ns)
        self.assertNotIn(path, RP._ABSENT_FILES)
        name, mod = RP.file_namespace(path)
        self.assertEqual(name, RP.file_module(path))
        self.assertEqual(mod.marker(), 314)
        self.assertEqual(self.scan_count, 2)

        # While `_ACTIVE_SCANS > 0`, even AST compile bumps `_CODE_EPOCH` so mid-scan races stay safe.
        with RP._ABSENT_LOCK:
            before_epoch = RP._CODE_EPOCH
            RP._ACTIVE_SCANS += 1
        try:
            compile(ast.parse("x = 1\n"), unrelated, "exec", dont_inherit=True)
        finally:
            with RP._ABSENT_LOCK:
                RP._ACTIVE_SCANS -= 1
                after_epoch = RP._CODE_EPOCH
        self.assertGreater(after_epoch, before_epoch)


if __name__ == "__main__":
    unittest.main()
