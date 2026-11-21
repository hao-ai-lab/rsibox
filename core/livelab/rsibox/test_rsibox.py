"""Golden cases for rsibox: the real daemon over real HTTP on the kernel's toy lab (CPU).

Each case names the failure it guards against. Cases marked "found on a real engine" encode
bugs the first vLLM runs exposed and the toy lab alone did not. What these do not prove: anything
about a real engine; that is a GPU receipt. The launchers are tested in `test_launch.py`.

    python3 -m unittest discover -s rsibox -t . -v      # from core/livelab
"""
import os
import shutil
import tempfile
import threading
import time
import unittest

from kernel import toy_lab

from rsibox import box as B
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


if __name__ == "__main__":
    unittest.main()
