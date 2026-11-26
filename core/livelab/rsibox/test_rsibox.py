"""Golden cases for rsibox: the real daemon over real HTTP on the kernel's toy lab (CPU).

Each case names the failure it guards against. Cases marked "found on a real engine" encode
bugs the first vLLM runs exposed and the toy lab alone did not. What these do not prove: anything
about a real engine; that is a GPU receipt. The launchers are tested in `test_launch.py`.

    python3 -m unittest discover -s rsibox -t . -v      # from core/livelab
"""
import contextlib
import difflib
import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest

from kernel import toy_lab

from rsibox import box as B
from rsibox import cli
from rsibox import client as C
from rsibox import server as S

LIVELAB = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def edited_ops(text):
    """The toy edit: `scale` multiplies by one more, so the workload's output changes."""
    old = "def scale(x):\n    return x * STEP_SCALE\n"
    assert text.count(old) == 1
    return text.replace(old, "def scale(x):\n    return x * (STEP_SCALE + 1)\n")


class Served:
    """A daemon on an ephemeral port around `factory`, plus a client for it."""

    def __init__(self, factory, **box_kw):
        self.state = tempfile.mkdtemp(prefix="rsibox-test-")
        self.box = B.Box(factory, spec="test", state_dir=self.state, **box_kw)
        self.daemon = S.Daemon(self.box, state_dir=self.state).start()
        self.client = C.Client(self.daemon.url, self.daemon.token, timeout=30)

    def close(self):
        self.box.close()
        self.daemon.stop()
        self.daemon.wait()
        shutil.rmtree(self.state, ignore_errors=True)


class EditLoop(unittest.TestCase):
    def setUp(self):
        self.s = Served(toy_lab.ToyLab)
        self.addCleanup(self.s.close)
        self.s.box.boot()

    def test_source_edit_install_check_uninstall_over_http(self):
        """Pain point: an install that reports success while the engine still runs old code."""
        c = self.s.client
        base = c.check()
        src = c.source("toy.ops")
        self.assertTrue(src["ok"], src)
        self.assertEqual(src["hash"], B.text_hash(src["text"]))

        done = c.install({"files": {"toy.ops": edited_ops(src["text"])}})
        self.assertEqual((done["http"], done["ok"], done["outcome"]), (200, True, "committed"), done)
        after = c.check()
        self.assertTrue(after["proven"], after)                    # the edited code ran
        self.assertNotEqual(after["output"], base["output"])       # and changed the result
        self.assertEqual(c.status()["state"], "edited")
        self.assertEqual(c.source("toy.ops")["text"], edited_ops(src["text"]))   # live text is the new base

        self.assertEqual(c.uninstall()["undone"], [done["id"]])
        self.assertEqual(c.check()["output"], base["output"])
        self.assertEqual(c.status()["state"], "base")

        kinds = [r["kind"] for r in c.receipts()["receipts"]]
        self.assertEqual(kinds, ["txn", "proof", "undo"])          # written by the lab's writer
        verbs = [e["verb"] for e in c.events()["events"]]
        self.assertEqual(verbs[:3], ["boot", "check", "install"])
        self.assertEqual(c.events(since=len(verbs))["events"], [])

    def test_a_patch_made_from_a_stale_text_is_refused_and_keeps_the_newer_edit(self):
        """Pain point: two writers. B edits the text it read before A's commit; a full-text
        install of B would silently drop A's edit."""
        c = self.s.client
        old = c.source("toy.ops")
        a_text = edited_ops(old["text"])
        self.assertTrue(c.install({"files": {"toy.ops": a_text}, "base": {"toy.ops": old["hash"]}})["ok"])
        after_a = c.check()["output"]
        b_text = old["text"].replace("return x + 1", "return x + 2")
        self.assertNotEqual(b_text, old["text"])
        stale = c.install({"files": {"toy.ops": b_text}, "base": {"toy.ops": old["hash"]}})
        self.assertEqual((stale["ok"], stale["outcome"], stale["codes"]), (False, "refused", ["base_mismatch"]))
        self.assertEqual(c.source("toy.ops")["text"], a_text)      # A's edit is still live
        self.assertEqual(c.check()["output"], after_a)
        self.assertEqual(len(c.status()["txns"]), 1)

    def test_wrong_token_is_refused_before_the_lab_runs(self):
        """Pain point: an open port that lets any local user edit the engine."""
        before = len(self.s.box.events)
        for token in ("", "nope", self.s.daemon.token + "x"):
            bad = C.Client(self.s.daemon.url, token)
            self.assertEqual(bad.status()["http"], 401)
            self.assertEqual(bad.install({"files": {"toy.ops": "x = 1\n"}})["http"], 401)
            self.assertEqual(bad.info()["http"], 401)
        self.assertEqual(len(self.s.box.events), before)           # no call reached the lab

    def test_a_patch_without_files_or_diff_is_refused(self):
        reply = self.s.client.install({"accept": ["anything"]})
        self.assertEqual((reply["ok"], reply["outcome"], reply["codes"]), (False, "refused", ["patch_invalid"]))
        self.assertEqual(self.s.client.status()["state"], "base")

    def test_no_route_writes_receipts_or_runs_commands(self):
        """Pain point: an agent that edits its own receipt. Only GET exists for receipts."""
        c = self.s.client
        for name in ("receipts", "events", "info", "source", "exec", "shell"):
            c_post = C.Client(c.url, c.token)
            C.POSTS = C.POSTS + (name,)
            try:
                self.assertEqual(c_post.request(name, {"x": 1})["http"], 404, name)
            finally:
                C.POSTS = C.POSTS[:-1]


class Lifecycle(unittest.TestCase):
    def test_a_slow_boot_is_visible_and_verbs_wait_for_it(self):
        """Found on a real engine: vLLM takes minutes to boot; the daemon must answer meanwhile."""
        gate = threading.Event()

        def slow(receipts=None):
            gate.wait(10)
            return toy_lab.ToyLab(receipts=receipts)

        s = Served(slow)
        self.addCleanup(s.close)
        s.box.start()
        self.assertEqual(s.client.info()["state"], "booting")
        early = s.client.check()
        self.assertEqual((early["http"], early["state"]), (409, "booting"))
        gate.set()
        info = s.client.wait(timeout=10, poll_s=0.02)
        self.assertEqual((info["ok"], info["state"]), (True, "ready"))
        self.assertIsNotNone(info["boot_s"])
        self.assertTrue(s.client.check()["ok"])

    def test_a_failed_boot_keeps_the_daemon_up_with_the_error(self):
        """Found on a real engine: vLLM failed to start (no C compiler); the box must say why."""
        def broken(receipts=None):
            raise RuntimeError("engine start failed: no C compiler")

        s = Served(broken)
        self.addCleanup(s.close)
        s.box.boot()
        info = s.client.wait(timeout=5, poll_s=0.02)
        self.assertEqual((info["ok"], info["state"]), (False, "failed"))
        self.assertIn("no C compiler", info["boot_error"])
        self.assertEqual(s.client.status()["http"], 409)

    def test_a_hung_check_does_not_make_the_box_look_dead(self):
        """Pain point: one stuck call that blocks every other request, status included."""
        release = threading.Event()

        class Hangs(toy_lab.ToyLab):
            def check(self, workload=None):
                release.wait(10)
                return {"ok": True, "output": None, "proven": None, "witness": {}, "findings": [], "cost_s": 0.0}

        s = Served(Hangs, busy_wait_s=0.1)
        self.addCleanup(s.close)
        self.addCleanup(release.set)
        s.box.boot()
        threading.Thread(target=s.client.check, daemon=True).start()
        deadline = time.monotonic() + 5
        while s.client.info()["in_flight"] is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(s.client.info()["in_flight"]["verb"], "check")     # info never waits
        busy = s.client.status()
        self.assertEqual((busy["http"], busy["state"], busy["in_flight"]["verb"]), (409, "busy", "check"))
        release.set()

    def test_a_lab_call_that_raises_is_answered_not_dropped(self):
        class Raises(toy_lab.ToyLab):
            def reset(self, kind="state"):
                raise ValueError("boom")

        s = Served(Raises)
        self.addCleanup(s.close)
        s.box.boot()
        reply = s.client.reset()
        self.assertEqual((reply["http"], reply["ok"], reply["raised"]), (200, False, True))
        self.assertIn("boom", reply["error"])
        self.assertTrue(s.client.check()["ok"])                    # the box still works

    def test_source_reads_a_lab_that_keeps_its_text_like_the_vllm_lab(self):
        """Found on a real engine: the vLLM lab has no `live_text`; `source` returned nothing,
        and the agent then sent the error text as the patch."""
        class VllmShaped:
            def __init__(self, receipts=None):
                self.base_text = {"pkg.mod": "x = 1\n"}

            def _live_source(self, module):
                return {"pkg.mod": "x = 2\n"}.get(module)

            def close(self):
                pass

        s = Served(VllmShaped)
        self.addCleanup(s.close)
        s.box.boot()
        self.assertEqual(s.client.source("pkg.mod")["text"], "x = 2\n")
        missing = s.client.source("pkg.other")
        self.assertEqual((missing["ok"], "text" in missing), (False, False))


class FreshCheckAndMetadataTests(unittest.TestCase):
    def test_fresh_check_round_trip_and_errors_over_http_client_and_cli(self):
        """Box, Client, and `rsibox fresh-check` run fresh subprocess checks with base,
        file patches, unified diffs, and custom workloads, and fail cleanly on bad/unsupported labs."""
        s = Served(toy_lab.ToyLab)
        self.addCleanup(s.close)
        s.box.boot()
        c = s.client

        live_base = c.check()
        fresh_base = c.fresh_check()
        self.assertEqual((fresh_base["http"], fresh_base["ok"]), (200, True), fresh_base)
        self.assertEqual(fresh_base["output"], live_base["output"])
        self.assertEqual(fresh_base["output_digest"], live_base["output_digest"])
        self.assertTrue(fresh_base["exact_match"])

        src = c.source("toy.ops")
        new_ops = edited_ops(src["text"])
        # Commit an edit first so live text differs from base, then verify fresh_check still works with base hash.
        self.assertTrue(c.install({"files": {"toy.ops": new_ops}, "base": {"toy.ops": src["hash"]}})["ok"])
        live_edit = c.check()
        fresh_edit = c.fresh_check({"files": {"toy.ops": new_ops}, "base": {"toy.ops": src["hash"]}})
        self.assertTrue(fresh_edit["ok"], fresh_edit)
        self.assertEqual(fresh_edit["output"], live_edit["output"])
        self.assertEqual(fresh_edit["output_digest"], live_edit["output_digest"])
        self.assertFalse(fresh_edit["exact_match"])

        diff_text = "".join(difflib.unified_diff(
            src["text"].splitlines(True), new_ops.splitlines(True),
            fromfile="a/toy/ops.py", tofile="b/toy/ops.py",
        ))
        fresh_diff = c.fresh_check({"diff": diff_text})
        self.assertTrue(fresh_diff["ok"], fresh_diff)
        self.assertEqual(fresh_diff["output"], live_edit["output"])

        custom = c.fresh_check({"files": {"toy.ops": new_ops}}, workload={"inputs": [1, 2]})
        self.assertTrue(custom["ok"], custom)
        self.assertEqual(len(custom["output"]["rank0"]), 2)
        self.assertIsNone(custom["exact_match"])

        refused = c.fresh_check({"accept": []})
        self.assertEqual((refused["http"], refused["ok"], refused["outcome"], refused["codes"]),
                         (200, False, "refused", ["patch_invalid"]))
        bad_syntax = c.fresh_check({"files": {"toy.ops": "def scale(x):\n    return x +\n"}})
        self.assertEqual((bad_syntax["http"], bad_syntax["ok"]), (200, False))
        self.assertTrue(bad_syntax["error"])

        box_file = os.path.join(s.state, S.BOX_FILE)
        ops_file = os.path.join(s.state, "ops_edit.py")
        diff_file = os.path.join(s.state, "ops.patch")
        bad_patch_file = os.path.join(s.state, "bad_patch.json")
        with open(ops_file, "w", encoding="utf-8") as fh:
            fh.write(new_ops)
        with open(diff_file, "w", encoding="utf-8") as fh:
            fh.write(diff_text)
        with open(bad_patch_file, "w", encoding="utf-8") as fh:
            json.dump({"accept": []}, fh)

        def run_cli(*argv):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli.main(list(argv))
            return rc, json.loads(buf.getvalue())

        rc_base, out_base = run_cli("fresh-check", "--box", box_file)
        self.assertEqual(rc_base, 0)
        self.assertEqual(out_base["output"], live_base["output"])

        rc_file, out_file = run_cli(
            "fresh-check", "--box", box_file, "--file", f"toy.ops={ops_file}",
            "--workload", '{"inputs":[1,2]}',
        )
        self.assertEqual(rc_file, 0)
        self.assertEqual(out_file["output"], custom["output"])

        rc_diff, out_diff = run_cli("fresh-check", "--box", box_file, "--diff", diff_file)
        self.assertEqual(rc_diff, 0)
        self.assertEqual(out_diff["output"], live_edit["output"])

        rc_bad, out_bad = run_cli("fresh-check", "--box", box_file, "--patch", bad_patch_file)
        self.assertEqual(rc_bad, 2)
        self.assertEqual(out_bad["codes"], ["patch_invalid"])

        class NoFreshLab:
            def __init__(self, receipts=None):
                pass

            def close(self):
                pass

        class NotImplLab(toy_lab.ToyLab):
            def fresh_check(self, patch, workload=None):
                raise NotImplementedError("stub")

        for factory in (NoFreshLab, NotImplLab):
            s_err = Served(factory)
            self.addCleanup(s_err.close)
            s_err.box.boot()
            rep = s_err.client.fresh_check()
            self.assertEqual((rep["http"], rep["ok"], rep["codes"]), (200, False, ["unsupported"]))
            self.assertTrue(rep["error"])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_err = cli.main(["fresh-check", "--box", os.path.join(s_err.state, S.BOX_FILE)])
            self.assertEqual(rc_err, 1)
            self.assertEqual(json.loads(buf.getvalue())["codes"], ["unsupported"])

    def test_status_identity_modules_and_check_timing_and_proven_semantics(self):
        """Status surfaces process-invariant `identity.modules` / `box.modules_digest` vs
        process-local `fingerprint`; check surfaces `timing` and `proven_semantics` + `exact_match`."""
        class TimedToyLab(toy_lab.ToyLab):
            def check(self, workload=None):
                rep = super().check(workload)
                if rep.get("ok"):
                    rep["seconds"] = 0.012345
                return rep

        s = Served(TimedToyLab)
        self.addCleanup(s.close)
        s.box.boot()
        c = s.client

        st0 = c.status()
        self.assertEqual(st0["box"]["fingerprint_scope"], "process_local")
        self.assertTrue(st0["box"]["modules_digest"])

        self.assertTrue(c.reset("respawn")["ok"])
        st1 = c.status()
        self.assertNotEqual(st1["fingerprint"], st0["fingerprint"])
        self.assertEqual(st1["identity"]["modules"], st0["identity"]["modules"])
        self.assertEqual(st1["box"]["modules_digest"], st0["box"]["modules_digest"])

        base = c.check()
        self.assertIsNone(base["proven"])
        self.assertTrue(base["exact_match"])
        self.assertEqual(base["box"]["timing"]["workload_s"], 0.012345)
        self.assertEqual(base["box"]["timing"]["verb_s"], base["cost_s"])
        self.assertEqual(base["box"]["timing"]["wall_s"], base["box"]["wall_s"])
        self.assertNotIn("proven_semantics", base["box"])

        src = c.source("toy.ops")["text"]
        self.assertTrue(c.install({"files": {"toy.ops": edited_ops(src)}})["ok"])
        diverged = c.check()
        self.assertTrue(diverged["proven"])
        self.assertEqual(diverged["box"]["proven_semantics"], "witness_executed_only")
        self.assertFalse(diverged["exact_match"])
        self.assertNotEqual(diverged["output_digest"], base["output_digest"])

        self.assertTrue(c.uninstall()["ok"])
        noop_src = src.replace("    return x * STEP_SCALE\n", "    y = x * STEP_SCALE\n    return y\n")
        self.assertTrue(c.install({"files": {"toy.ops": noop_src}})["ok"])
        noop_chk = c.check()
        self.assertTrue(noop_chk["proven"])
        self.assertEqual(noop_chk["box"]["proven_semantics"], "witness_executed_only")
        self.assertTrue(noop_chk["exact_match"])
        self.assertEqual(noop_chk["output_digest"], base["output_digest"])

    def test_fresh_check_resolves_diff_for_files_only_baselab_without_host_imports(self):
        """Regression P1-1 & P2-1: labs with `files()` whose `fresh_check()` reads only
        `patch['files']` must still resolve `patch['diff']` via `_base_source`/`base_text`
        (never triggering host GPU package imports), strip `diff`, and reject bad types."""
        from unittest import mock
        from rsibox import harness as H

        base_src = toy_lab.toy.source("ops")
        new_src = edited_ops(base_src)
        diff_text = "".join(difflib.unified_diff(
            base_src.splitlines(True), new_src.splitlines(True),
            fromfile="a/gpu_pkg/ops.py", tofile="b/gpu_pkg/ops.py",
        ))
        seen_patches = []

        class FilesOnlyBaseLab:
            def __init__(self, receipts=None):
                self.base_text = {"gpu_pkg.ops": base_src}

            def _base_source(self, key):
                return self.base_text[key]

            def files(self, patch, against="live"):
                return H.resolve_patch_files(patch, lambda k: self.base_text.get(k), against=against)

            def fresh_check(self, patch, workload=None):
                seen_patches.append(patch)
                if patch is not None and "diff" in patch:
                    raise AssertionError("Box must strip 'diff' once resolved to 'files'")
                src = ((patch or {}).get("files") or {}).get("gpu_pkg.ops", base_src)
                ns = {}
                exec(compile(src, "<gpu_pkg.ops>", "exec"), ns)
                return {"ok": True, "output": [ns["scale"](x) for x in (workload or {}).get("inputs", [1, 3])]}

            def close(self):
                pass

        s = Served(FilesOnlyBaseLab)
        self.addCleanup(s.close)
        s.box.boot()
        c = s.client

        with mock.patch.object(H, "read_module", side_effect=AssertionError("host import forbidden")):
            base = c.fresh_check()
            self.assertEqual((base["http"], base["ok"], base["exact_match"]), (200, True, True), base)
            edited = c.fresh_check({"diff": diff_text})
            self.assertEqual((edited["http"], edited["ok"], edited["exact_match"]), (200, True, False), edited)
            self.assertNotEqual(edited["output"], base["output"])
            self.assertEqual(seen_patches[-1], {"files": {"gpu_pkg.ops": new_src}})

            diff_file = os.path.join(s.state, "gpu.patch")
            with open(diff_file, "w", encoding="utf-8") as fh:
                fh.write(diff_text)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli.main(["fresh-check", "--box", os.path.join(s.state, S.BOX_FILE), "--diff", diff_file])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(buf.getvalue())["output"], edited["output"])

        for bad_patch in (
            "not_a_dict",
            {"diff": "garbage"},
            {"diff": 123},
            {"files": "bad"},
            {"files": {}},
            {"files": {"gpu_pkg.ops": 123}},
            {"files": {"gpu_pkg.ops": new_src}, "diff": diff_text},
        ):
            rep = c.fresh_check(bad_patch)
            self.assertEqual((rep["http"], rep["ok"], rep["outcome"], rep["codes"]),
                             (200, False, "refused", ["patch_invalid"]), bad_patch)

    def test_box_mixed_key_digest_safety_and_fresh_check_timeout(self):
        """Regression P2-1 & P2-2: `Box.call()` handles mixed-type dict keys in `workload`
        and `output` without raising, and `Client.fresh_check` / `rsibox fresh-check --timeout`
        forward per-call HTTP timeouts cleanly."""
        from unittest import mock

        class MixedKeyLab(toy_lab.ToyLab):
            def check(self, workload=None):
                return {"ok": True, "output": {1: "a", "2": "b"}, "proven": None,
                        "witness": {}, "findings": [], "cost_s": 0.001}

            def fresh_check(self, patch, workload=None):
                return {"ok": True, "output": {1: "a", "2": "b"}}

        s = Served(MixedKeyLab)
        self.addCleanup(s.close)
        s.box.boot()

        status_code, reply = s.box.call("check", {"workload": {1: "x", "2": "y"}})
        self.assertEqual(status_code, 200)
        self.assertTrue(reply["ok"], reply)
        self.assertTrue(reply["exact_match"])
        self.assertTrue(reply["output_digest"])

        status_fc, reply_fc = s.box.call("fresh_check", {"workload": {1: "x", "2": "y"}})
        self.assertEqual(status_fc, 200)
        self.assertTrue(reply_fc["ok"], reply_fc)
        self.assertTrue(reply_fc["exact_match"])

        seen_timeouts = []
        orig_request = C.Client.request

        def spy_request(self_client, name, body=None, query=None, timeout=None):
            seen_timeouts.append((name, timeout))
            return orig_request(self_client, name, body=body, query=query, timeout=timeout)

        with mock.patch.object(C.Client, "request", autospec=True, side_effect=spy_request):
            rep = s.client.fresh_check(timeout=912.5)
            self.assertTrue(rep["ok"])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli.main(["fresh-check", "--box", os.path.join(s.state, S.BOX_FILE), "--timeout", "1234"])
            self.assertEqual(rc, 0)
        self.assertEqual(seen_timeouts, [("fresh_check", 912.5), ("fresh_check", 1234.0)])


if __name__ == "__main__":
    unittest.main()
