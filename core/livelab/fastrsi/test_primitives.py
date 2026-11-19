"""Golden tests for `core/livelab/fastrsi/` low-level stdlib primitives:
`ipc`, `proc`, `hooks`, `harness`, and `bootstrap_livelab`.

Run from the repo root:
    python3 -m unittest discover -s core/livelab/fastrsi -p 'test_primitives.py' -t core/livelab -v
"""
import importlib
import itertools
import json
import linecache
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import fastrsi
from fastrsi import harness, hooks, ipc, proc

LIVELAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEQ = itertools.count(1)


class IpcTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fastrsi-ipc-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_unix_socket_callback_and_queue_modes_and_path_limit(self):
        """Input: real `UnixJsonServer` in callback mode and queue mode, plus an overlong path.
        Expected: callback answers valid JSON, returns error on malformed JSON without killing
        the server; queue mode delivers `(req, reply)` cleanly; paths > `MAX_SOCK_PATH` fail
        fast with ValueError before binding."""
        too_long = "/" + "a" * (ipc.MAX_SOCK_PATH + 10) + ".sock"
        self.assertIsNotNone(ipc.sock_path_error(too_long))
        with self.assertRaises(ValueError):
            ipc.UnixJsonServer(too_long, handler=lambda r: {"ok": True})

        sock_path = ipc.short_sock_path("test-cb")
        with ipc.UnixJsonServer(sock_path, handler=lambda r: {"ok": True, "echo": r["x"]}):
            self.assertEqual(ipc.unix_json_call(sock_path, {"x": 42}), {"ok": True, "echo": 42})
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as raw:
                raw.connect(sock_path)
                raw.sendall(b"{not-json\n")
                bad_reply = json.loads(raw.makefile("rb").readline())
            self.assertFalse(bad_reply["ok"])
            self.assertIn("bad request", bad_reply["error"])
            self.assertEqual(ipc.unix_json_call(sock_path, {"x": 7}), {"ok": True, "echo": 7})
        self.assertFalse(os.path.exists(sock_path))

        qsock = ipc.short_sock_path("test-q")
        with ipc.UnixJsonServer(qsock) as srv:
            def responder():
                req, reply = srv.next(timeout=3.0)
                reply({"ok": True, "v": req["v"] + 1})

            th = threading.Thread(target=responder, daemon=True)
            th.start()
            self.assertEqual(ipc.unix_json_call(qsock, {"v": 9}, timeout=3.0), {"ok": True, "v": 10})
            th.join(timeout=3.0)

    def test_file_channel_and_mailbox_round_trip_and_timeout(self):
        """Input: real `FileChannelServer` and `MailboxServer` over temp directories.
        Expected: monotonic replies succeed across sequential calls; a silent directory times
        out cleanly (`lost=True` for file channel, `TimeoutError` for mailbox)."""
        fc_dir = os.path.join(self.tmp, "fc")
        with ipc.FileChannelServer(fc_dir, handler=lambda c: {"ok": True, "op": c["op"]}, poll_s=0.002):
            r1 = ipc.file_channel_call(fc_dir, {"op": "ping"}, timeout=3.0, poll_s=0.002)
            r2 = ipc.file_channel_call(fc_dir, {"op": "status"}, timeout=3.0, poll_s=0.002)
            self.assertTrue(r1["ok"] and r1["op"] == "ping")
            self.assertTrue(r2["ok"] and r2["op"] == "status")
            self.assertIn("client_round_trip_s", r1)

        dead_fc = os.path.join(self.tmp, "fc-dead")
        os.makedirs(dead_fc, exist_ok=True)
        timed = ipc.file_channel_call(dead_fc, {"op": "ping"}, timeout=0.05, poll_s=0.005)
        self.assertFalse(timed["ok"])
        self.assertTrue(timed["lost"])

        mb_dir = os.path.join(self.tmp, "mb")
        q: queue.Queue = queue.Queue()
        with ipc.MailboxServer(mb_dir, q, poll_s=0.005):
            def drain():
                req, reply = q.get(timeout=3.0)
                reply({"ok": True, "got": req["verb"]})

            th = threading.Thread(target=drain, daemon=True)
            th.start()
            ans = ipc.mailbox_call(mb_dir, {"verb": "status"}, timeout=3.0, poll_s=0.005)
            self.assertEqual(ans, {"ok": True, "got": "status"})
            th.join(timeout=3.0)

        with self.assertRaises(TimeoutError):
            ipc.mailbox_call(os.path.join(self.tmp, "mb-dead"), {"verb": "ping"}, timeout=0.05, poll_s=0.005)


class ProcTests(unittest.TestCase):
    def test_stat_parsing_handles_comm_with_spaces_and_parens(self):
        """Input: `/proc/<pid>/stat` lines whose comm contains spaces and parentheses.
        Expected: `stat_state` and `stat_ppid` parse the fields after the LAST `')'`."""
        line = "12345 (python3 (worker 0)) S 9876 12345 12345 0 -1\n"
        self.assertEqual(proc.stat_state(line), "S")
        self.assertEqual(proc.stat_ppid(line), 9876)
        zombie = "12346 (a)b) Z 9876 12346 12346 0 -1\n"
        self.assertEqual(proc.stat_state(zombie), "Z")
        self.assertEqual(proc.stat_ppid(zombie), 9876)

    def test_descendants_and_stop_tree_terminate_parent_and_grandchild(self):
        """Input: a real subprocess spawning a grandchild process in its own session.
        Expected: `descendants()` finds the grandchild on both Linux (`/proc`) and macOS (`ps`),
        and `stop_tree()` terminates both processes cleanly."""
        child_script = (
            "import subprocess, sys, time\n"
            "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print(gc.pid, flush=True)\n"
            "time.sleep(60)\n"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", child_script],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            gc_pid = int(parent.stdout.readline().strip())
            self.assertIn(gc_pid, proc.descendants(parent.pid))
            self.assertTrue(proc.pid_alive(parent.pid))
            self.assertTrue(proc.pid_alive(gc_pid))
            self.assertTrue(proc.stop_tree(parent.pid, grace_s=1.0, kill_grace_s=2.0, include_pg=True))
            parent.wait(timeout=3.0)
            self.assertFalse(proc.pid_alive(gc_pid))
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=3.0)
            if parent.stdout:
                parent.stdout.close()

    def test_free_port_and_port_in_use(self):
        port = proc.free_port()
        self.assertFalse(proc.port_in_use(port))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", port))
            srv.listen(1)
            self.assertTrue(proc.port_in_use(port))


class HooksAndHarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fastrsi-hooks-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_stacked_import_hooks_and_overlay_compose_without_recursion(self):
        """Input: two `AfterImportHook`s stacked ahead of an `OverlayFinder`; importing a real
        package module on disk.
        Expected: no RecursionError; the module loads the overlaid text with original
        `co_filename` and populated `linecache`, and both callbacks fire after `exec_module`."""
        pkg = f"fastrsi_hook_pkg_{next(_SEQ)}"
        pkg_dir = os.path.join(self.tmp, pkg)
        os.makedirs(pkg_dir)
        with open(os.path.join(pkg_dir, "__init__.py"), "w", encoding="utf-8") as fh:
            fh.write("")
        mod_file = os.path.join(pkg_dir, "target.py")
        with open(mod_file, "w", encoding="utf-8") as fh:
            fh.write("def value():\n    return 'disk'\n")

        seen = []
        ov_text = "def value():\n    return 'overlay'\n"
        sys.path.insert(0, self.tmp)
        orig_meta = list(sys.meta_path)
        self.addCleanup(lambda: sys.path.remove(self.tmp) if self.tmp in sys.path else None)
        self.addCleanup(lambda: sys.meta_path.__setitem__(slice(None), orig_meta))
        self.addCleanup(lambda: [sys.modules.pop(k, None) for k in list(sys.modules) if k == pkg or k.startswith(pkg + ".")])
        self.addCleanup(lambda: linecache.cache.pop(mod_file, None))

        hooks.install_overlay(files={f"{pkg}.target": ov_text})
        hooks.install_import_hook(f"{pkg}.target", lambda m: seen.append(("cb1", m.value())))
        hooks.install_import_hook(f"{pkg}.target", lambda m: seen.append(("cb2", m.value())))

        mod = importlib.import_module(f"{pkg}.target")
        self.assertEqual(mod.value(), "overlay")
        self.assertEqual(mod.value.__code__.co_filename, mod_file)
        self.assertIn("overlay", "".join(linecache.getlines(mod_file)))
        self.assertEqual(seen, [("cb1", "overlay"), ("cb2", "overlay")])

    def test_package_overlay_handles_init_leaf_and_standalone_files(self):
        """Input: a real package with `__init__.py`, leaf module, `__pycache__`, and a standalone
        script outside the package.
        Expected: `PackageOverlay` strips `__pycache__`, overrides both `__init__.py` and leaf
        modules without mutating the source tree, writes standalone overrides, and cleans up."""
        pkg = f"ovpkg_{next(_SEQ)}"
        pkg_dir = os.path.join(self.tmp, pkg)
        os.makedirs(os.path.join(pkg_dir, "__pycache__"))
        with open(os.path.join(pkg_dir, "__init__.py"), "w", encoding="utf-8") as fh:
            fh.write("TAG = 'base_init'\n")
        with open(os.path.join(pkg_dir, "mod.py"), "w", encoding="utf-8") as fh:
            fh.write("X = 1\n")
        with open(os.path.join(pkg_dir, "__pycache__", "mod.pyc"), "w", encoding="utf-8") as fh:
            fh.write("stale")
        standalone = os.path.join(self.tmp, "reward.py")
        with open(standalone, "w", encoding="utf-8") as fh:
            fh.write("SCORE = 1\n")

        with harness.PackageOverlay(
            pkg,
            {pkg: "TAG = 'patched_init'\n", f"{pkg}.mod": "X = 2\n", standalone: "SCORE = 9\n"},
            src_dir=pkg_dir,
        ) as ov:
            root = ov.root
            self.assertFalse(os.path.exists(os.path.join(root, pkg, "__pycache__")))
            with open(os.path.join(root, pkg, "__init__.py"), encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "TAG = 'patched_init'\n")
            with open(os.path.join(root, pkg, "mod.py"), encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "X = 2\n")
            self.assertIn(standalone, ov.standalone)
            with open(ov.standalone[standalone], encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "SCORE = 9\n")
            with open(os.path.join(pkg_dir, "mod.py"), encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "X = 1\n")
        self.assertFalse(os.path.exists(root))

    def test_edit_once_and_run_fresh_subprocess(self):
        """Input: `edit_once` unique/ambiguous anchors and `run_fresh_subprocess` executing a
        real fresh Python module over a `PackageOverlay`.
        Expected: `edit_once` enforces uniqueness; `run_fresh_subprocess` parses `FRESH {...}`."""
        self.assertEqual(harness.edit_once("a = 1\nb = 2\n", "a = 1", "a = 9"), "a = 9\nb = 2\n")
        with self.assertRaises(AssertionError):
            harness.edit_once("x = 1\nx = 1\n", "x = 1", "x = 2")
        with self.assertRaises(AssertionError):
            harness.edit_once("a = 1\n", "missing", "y")

        pkg = f"fsub_{next(_SEQ)}"
        pkg_dir = os.path.join(self.tmp, pkg)
        os.makedirs(pkg_dir)
        with open(os.path.join(pkg_dir, "__init__.py"), "w", encoding="utf-8") as fh:
            fh.write("")
        with open(os.path.join(pkg_dir, "runner.py"), "w", encoding="utf-8") as fh:
            fh.write(
                "import json, sys\n"
                "VAL = 1\n"
                "spec = json.load(open(sys.argv[1]))\n"
                "print('FRESH ' + json.dumps({'output': spec['x'] + VAL}), flush=True)\n"
            )
        with harness.PackageOverlay(
            pkg,
            {f"{pkg}.runner": "import json, sys\nVAL = 10\nspec = json.load(open(sys.argv[1]))\nprint('FRESH ' + json.dumps({'output': spec['x'] + VAL}), flush=True)\n"},
            src_dir=pkg_dir,
        ) as ov:
            res = harness.run_fresh_subprocess(f"{pkg}.runner", {"x": 5}, ov.root, timeout=30.0)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["output"], 15)

    def test_bootstrap_livelab_resolves_path_without_mutating_environ(self):
        """Input: call `bootstrap_livelab` when `FASTRSI_LIVELAB` is unset.
        Expected: resolves `core/livelab` on `sys.path` and leaves `os.environ` untouched."""
        clean_env = {k: v for k, v in os.environ.items() if k != "FASTRSI_LIVELAB"}
        with mock.patch.dict(os.environ, clean_env, clear=True):
            resolved = fastrsi.bootstrap_livelab(__file__)
            self.assertEqual(resolved, os.path.abspath(LIVELAB))
            self.assertIn(resolved, sys.path)
            self.assertNotIn("FASTRSI_LIVELAB", os.environ)


if __name__ == "__main__":
    unittest.main()
