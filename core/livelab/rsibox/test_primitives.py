"""Golden tests for `core/livelab/rsibox/` low-level stdlib primitives:
`ipc`, `proc`, `hooks`, `harness`, and `bootstrap_livelab`.

Run from the repo root:
    python3 -m unittest discover -s core/livelab/rsibox -p 'test_primitives.py' -t core/livelab -v
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
import time
import unittest
from unittest import mock

import rsibox
from rsibox import diffs, harness, hooks, ipc, proc

LIVELAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEQ = itertools.count(1)


class IpcTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rsibox-ipc-")
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

    def test_file_channel_and_mailbox_alive_fn_fast_fails_on_real_sigkill(self):
        """Pain point: when a live engine worker is SIGKILLed right after a client starts polling
        a file channel (or mailbox) with a 60-180 s check timeout, polling without liveness checks
        stalls recovery for the entire timeout. With `alive_fn`, a real SIGKILL fails fast in
        milliseconds while preserving valid replies written just before exit and honoring `min_seq`."""
        fc_dir = os.path.join(self.tmp, "fc-kill")
        os.makedirs(fc_dir, exist_ok=True)
        # 1. Pre-exit dead peer fails immediately without writing ctl.json.
        pre_dead = ipc.file_channel_call(fc_dir, {"op": "status"}, timeout=60.0, alive_fn=lambda: False)
        self.assertFalse(pre_dead["ok"])
        self.assertTrue(pre_dead["lost"])
        self.assertIn("before send", pre_dead["error"])
        self.assertFalse(os.path.exists(os.path.join(fc_dir, "ctl.json")))

        # 2. Invalid arguments fail closed with TypeError / ValueError, and NaN timeout does not hang.
        with self.assertRaises(TypeError):
            ipc.file_channel_call(fc_dir, {"op": "status"}, alive_fn="not-callable")  # type: ignore[arg-type]
        for bad_min_seq in (True, 1.5, "10"):
            with self.assertRaises(TypeError):
                ipc.file_channel_call(fc_dir, {"op": "status"}, min_seq=bad_min_seq)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ipc.file_channel_call(fc_dir, {"op": "status"}, min_seq=-1)
        with self.assertRaises(ValueError):
            ipc.file_channel_call(fc_dir, {"op": "status"}, poll_s=-0.1)
        with self.assertRaises(ValueError):
            ipc.file_channel_call(fc_dir, {"op": "status"}, poll_s=float("nan"))
        with self.assertRaises(TypeError):
            ipc.mailbox_call(os.path.join(self.tmp, "mb-bad"), {"verb": "status"}, alive_fn=123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ipc.mailbox_call(os.path.join(self.tmp, "mb-bad"), {"verb": "status"}, poll_s=float("inf"))
        nan_fc = ipc.file_channel_call(os.path.join(self.tmp, "fc-nan"), {"op": "status"}, timeout=float("nan"), min_seq=0)
        self.assertFalse(nan_fc["ok"])
        self.assertTrue(nan_fc["lost"])
        with self.assertRaises(TimeoutError):
            ipc.mailbox_call(os.path.join(self.tmp, "mb-nan"), {"verb": "status"}, timeout=float("nan"))

        # 3. Real subprocess gets SIGKILLed mid-request -> file_channel_call fails fast (< 2.0 s vs 60 s timeout).
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(lambda: (sleeper.kill(), sleeper.wait()) if sleeper.poll() is None else None)
        ctl_path = os.path.join(fc_dir, "ctl.json")
        stop_killer = threading.Event()

        def kill_after_ctl():
            while not stop_killer.is_set() and not os.path.exists(ctl_path):
                time.sleep(0.002)
            if not stop_killer.is_set():
                sleeper.kill()
                sleeper.wait()

        killer = threading.Thread(target=kill_after_ctl, daemon=True)
        killer.start()
        t0 = time.perf_counter()
        try:
            lost = ipc.file_channel_call(
                fc_dir, {"op": "status"}, timeout=60.0, poll_s=0.005, min_seq=10,
                alive_fn=lambda: sleeper.poll() is None,
            )
            elapsed_s = time.perf_counter() - t0
        finally:
            stop_killer.set()
            killer.join(timeout=3.0)
        self.assertFalse(lost["ok"])
        self.assertTrue(lost["lost"])
        self.assertEqual(lost["seq"], 11)
        self.assertIn("before reply", lost["error"])
        self.assertLess(elapsed_s, 2.0, f"expected fast-fail on SIGKILL, took {elapsed_s:.3f}s")

        # 4. Reply written right before peer exits still succeeds (final reply check on death).
        race_dir = os.path.join(self.tmp, "fc-reply-then-exit")
        os.makedirs(os.path.join(race_dir, "reply"), exist_ok=True)
        checks = [0]

        def alive_after_reply() -> bool:
            checks[0] += 1
            if checks[0] == 1:
                return True  # alive before send
            ipc.atomic_write_json(os.path.join(race_dir, "reply", "1.json"), {"ok": True, "op": "final"})
            return False  # peer exits immediately after writing reply

        race_reply = ipc.file_channel_call(
            race_dir, {"op": "final"}, timeout=5.0, poll_s=0.005, alive_fn=alive_after_reply,
        )
        self.assertTrue(race_reply["ok"])
        self.assertEqual(race_reply["op"], "final")

        # 5. Mailbox fast-fail raises ConnectionError on dead peer without creating dir/request,
        # unlinks stale req on mid-call death, and preserves final replies written right before exit.
        mb_pre_dead = os.path.join(self.tmp, "mb-pre-dead")
        with self.assertRaises(ConnectionError):
            ipc.mailbox_call(mb_pre_dead, {"verb": "status"}, timeout=60.0, alive_fn=lambda: False)
        self.assertFalse(os.path.exists(mb_pre_dead))

        mb_dir = os.path.join(self.tmp, "mb-kill")
        sleeper_mb = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(lambda: (sleeper_mb.kill(), sleeper_mb.wait()) if sleeper_mb.poll() is None else None)
        stop_mb_killer = threading.Event()

        def kill_after_mb_req():
            while not stop_mb_killer.is_set():
                if os.path.isdir(mb_dir) and any(n.startswith("req-") for n in os.listdir(mb_dir)):
                    break
                time.sleep(0.002)
            if not stop_mb_killer.is_set():
                sleeper_mb.kill()
                sleeper_mb.wait()

        mb_killer = threading.Thread(target=kill_after_mb_req, daemon=True)
        mb_killer.start()
        t_mb = time.perf_counter()
        try:
            with self.assertRaises(ConnectionError):
                ipc.mailbox_call(mb_dir, {"verb": "status"}, timeout=60.0, poll_s=0.005, alive_fn=lambda: sleeper_mb.poll() is None)
            mb_elapsed = time.perf_counter() - t_mb
        finally:
            stop_mb_killer.set()
            mb_killer.join(timeout=3.0)
        self.assertLess(mb_elapsed, 2.0)
        self.assertEqual([n for n in os.listdir(mb_dir) if n.startswith("req-")], [])

        # Mailbox reply written right before peer exit still succeeds (after NFS listdir revalidation).
        mb_race = os.path.join(self.tmp, "mb-reply-then-exit")
        mb_checks = [0]

        def mb_alive_after_reply() -> bool:
            mb_checks[0] += 1
            if mb_checks[0] == 1:
                return True
            reqs = [n for n in os.listdir(mb_race) if n.startswith("req-") and n.endswith(".json")]
            self.assertEqual(len(reqs), 1)
            rid = reqs[0][4:-5]
            ipc.atomic_write_json(os.path.join(mb_race, f"resp-{rid}.json"), {"ok": True, "verb": "final"})
            return False

        mb_race_ans = ipc.mailbox_call(
            mb_race, {"verb": "final"}, timeout=5.0, poll_s=0.005, alive_fn=mb_alive_after_reply,
        )
        self.assertEqual(mb_race_ans, {"ok": True, "verb": "final"})

    def test_unix_json_call_and_stream_alive_fn_fast_fails_with_inherited_fd_on_sigkill(self):
        """Pain point: when a worker serving an AF_UNIX socket has a forked/spawned child holding
        an inherited copy of the connection FD, SIGKILLing the worker does NOT deliver EOF to the
        client socket — causing unix_json_call / unix_json_stream to hang for the full timeout.
        With `alive_fn`, SIGKILL fails fast in milliseconds while still preserving replies written
        immediately before peer exit."""
        sock_path = ipc.short_sock_path("test-unix-kill")
        self.addCleanup(lambda: os.unlink(sock_path) if os.path.exists(sock_path) else None)

        # 1. Eager validation & pre-dead peer fail closed at call time (without connecting or iterating).
        with self.assertRaises(TypeError):
            ipc.unix_json_call(sock_path, {"method": "ping"}, alive_fn="bad")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ipc.unix_json_call(sock_path, {"method": "ping"}, timeout=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ipc.unix_json_call(sock_path, {"method": "ping"}, timeout=-0.5)
        with self.assertRaises(ValueError):
            ipc.unix_json_call(sock_path, {"method": "ping"}, poll_s=-0.1)
        with self.assertRaises(ValueError):
            ipc.unix_json_call(sock_path, {"method": "ping"}, poll_s=float("nan"))
        with self.assertRaises(ValueError):
            ipc.unix_json_call(sock_path, {"method": "ping"}, poll_s=float("inf"))
        with self.assertRaises(ValueError):
            ipc.unix_json_call(sock_path, {"method": "ping"}, poll_s=10**400)
        with self.assertRaises(ValueError):
            ipc.unix_json_call(sock_path, {"method": "ping"}, timeout=10**400)
        with self.assertRaises(TimeoutError):
            ipc.unix_json_call(sock_path, {"method": "ping"}, timeout=float("nan"))
        with self.assertRaises(ConnectionError):
            ipc.unix_json_call(sock_path, {"method": "ping"}, timeout=30.0, alive_fn=lambda: False)
        # unix_json_stream validates eagerly BEFORE iteration!
        with self.assertRaises(TypeError):
            ipc.unix_json_stream(sock_path, {"method": "watch"}, alive_fn=123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ipc.unix_json_stream(sock_path, {"method": "watch"}, poll_s=-1.0)
        with self.assertRaises(ConnectionError):
            ipc.unix_json_stream(sock_path, {"method": "watch"}, timeout=30.0, alive_fn=lambda: False)

        # 2. Real server process spawns a child inheriting the accepted socket FD, then gets SIGKILLed!
        #    The child stays alive holding the FD open, so the kernel does NOT send EOF.
        server_code = (
            "import json, os, socket, subprocess, sys, time\n"
            "sock_path, mode = sys.argv[1], sys.argv[2]\n"
            "srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            "srv.bind(sock_path)\n"
            "srv.listen(4)\n"
            "print('READY', flush=True)\n"
            "conn, _ = srv.accept()\n"
            "conn.recv(4096)\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], pass_fds=(conn.fileno(),))\n"
            "print(f'ACCEPTED {child.pid}', flush=True)\n"
            "if mode == 'stream':\n"
            "    conn.sendall(b'{\"event\": \"started\"}\\n')\n"
            "time.sleep(30)\n"
        )

        for mode in ("call", "stream"):
            with self.subTest(mode=mode):
                if os.path.exists(sock_path):
                    os.unlink(sock_path)
                srv_proc = subprocess.Popen(
                    [sys.executable, "-c", server_code, sock_path, mode],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                child_pid_holder: list[int] = []

                def cleanup_procs(sp=srv_proc, cph=child_pid_holder):
                    if sp.poll() is None:
                        sp.kill()
                        sp.wait()
                    for cpid in cph:
                        try:
                            os.kill(cpid, 9)
                        except ProcessLookupError:
                            pass
                    if sp.stdout:
                        sp.stdout.close()
                    if sp.stderr:
                        sp.stderr.close()

                self.addCleanup(cleanup_procs)
                assert srv_proc.stdout is not None
                ready_line = srv_proc.stdout.readline().strip()
                self.assertEqual(ready_line, "READY")

                def kill_after_accept(sp=srv_proc, cph=child_pid_holder):
                    acc_line = sp.stdout.readline().strip()
                    if acc_line.startswith("ACCEPTED "):
                        cph.append(int(acc_line.split()[1]))
                    sp.kill()
                    sp.wait()

                killer = threading.Thread(target=kill_after_accept, daemon=True)
                killer.start()
                t0 = time.perf_counter()
                if mode == "call":
                    with self.assertRaises(ConnectionError):
                        ipc.unix_json_call(
                            sock_path, {"method": "generate"}, timeout=30.0, poll_s=0.01,
                            alive_fn=lambda sp=srv_proc: sp.poll() is None,
                        )
                else:
                    seen: list[dict] = []
                    with self.assertRaises(ConnectionError):
                        for ev in ipc.unix_json_stream(
                            sock_path, {"method": "watch"}, timeout=30.0, poll_s=0.01,
                            alive_fn=lambda sp=srv_proc: sp.poll() is None,
                        ):
                            seen.append(ev)
                    self.assertEqual(seen, [{"event": "started"}])
                elapsed = time.perf_counter() - t0
                killer.join(timeout=3.0)
                self.assertLess(elapsed, 2.0, f"{mode} stalled {elapsed:.3f}s despite SIGKILL (inherited FD)")

        # 3. Final-reply race guard: reply written right before alive_fn turns False still succeeds!
        race_sock = ipc.short_sock_path("test-unix-race")
        with ipc.UnixJsonServer(race_sock) as srv:
            def delayed_reply():
                req, reply = srv.next(timeout=3.0)
                time.sleep(0.03)
                reply({"ok": True, "got": req["method"]})

            checks = [0]
            replier = threading.Thread(target=delayed_reply, daemon=True)
            replier.start()

            def alive_turns_false_after_reply() -> bool:
                checks[0] += 1
                if checks[0] == 1:
                    return True
                replier.join(timeout=3.0)
                return False

            ans = ipc.unix_json_call(
                race_sock, {"method": "final"}, timeout=5.0, poll_s=0.01,
                alive_fn=alive_turns_false_after_reply,
            )
            self.assertEqual(ans, {"ok": True, "got": "final"})
            replier.join(timeout=3.0)

        # 4. P0-1 & P1-1..P1-5 regressions:
        #    (a) FD >= 1024 works without FD_SETSIZE ValueError (DefaultSelector),
        #    (b) >64 KiB chunked reply + poll_s=0.0 + timeout=inf,
        #    (c) timeout=0.0 raises TimeoutError (never ConnectionError) for both alive_fn=None and set,
        #    (d) clean EOF without trailing newline + stream clean EOF racing alive_fn()->False,
        #    (e) large sendall (> SO_SNDBUF) fails fast on dead peer instead of blocking until timeout,
        #    (f) unix_json_stream per-event inactivity timeout & clean early close (0 ResourceWarnings).
        big_payload = "A" * 131072
        hi_sock = ipc.short_sock_path("test-unix-hi")
        with ipc.UnixJsonServer(hi_sock, handler=lambda r: {"ok": True, "size": len(r.get("blob", "")), "echo": big_payload}):
            # Push socket FD past 1024 if RLIMIT_NOFILE allows it.
            duped: list[int] = []
            try:
                while len(duped) < 1040:
                    fd = os.dup(0)
                    duped.append(fd)
                    if fd >= 1030:
                        break
                ans_hi = ipc.unix_json_call(
                    hi_sock, {"method": "big", "blob": big_payload},
                    timeout=float("inf"), poll_s=0.01, alive_fn=lambda: True,
                )
                self.assertEqual(ans_hi["size"], len(big_payload))
                self.assertEqual(len(ans_hi["echo"]), len(big_payload))
            finally:
                for fd in duped:
                    os.close(fd)

        # timeout=0.0 against silent peer raises TimeoutError for both alive_fn=None and alive_fn!=None.
        t0_sock = ipc.short_sock_path("test-unix-t0")
        with ipc.UnixJsonServer(t0_sock):
            for af in (None, lambda: True):
                with self.assertRaises(TimeoutError):
                    ipc.unix_json_call(t0_sock, {"method": "slow"}, timeout=0.0, poll_s=0.0, alive_fn=af)

        # Clean EOF without trailing newline + stream clean EOF racing alive_fn()->False.
        eof_sock = ipc.short_sock_path("test-unix-eof")
        self.addCleanup(lambda: os.unlink(eof_sock) if os.path.exists(eof_sock) else None)
        eof_srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        eof_srv.bind(eof_sock)
        eof_srv.listen(4)
        self.addCleanup(eof_srv.close)

        c1_done, c2_done = threading.Event(), threading.Event()

        def serve_eof_cases():
            # Case 1: no-newline JSON then clean EOF.
            c1, _ = eof_srv.accept()
            with c1:
                c1.recv(4096)
                time.sleep(0.02)
                c1.sendall(b'{"ok": true, "eof": 1}')
            c1_done.set()
            # Case 2: stream burst in one send then clean EOF (no stop_event).
            c2, _ = eof_srv.accept()
            with c2:
                c2.recv(4096)
                time.sleep(0.02)
                c2.sendall(b'{"event": "a"}\n{"event": "b"}\n')
            c2_done.set()
            # Case 3: slow multi-tick stream (3 ticks 0.12s apart, total 0.36s > 0.25s timeout -> succeeds with inactivity timeout!).
            c3, _ = eof_srv.accept()
            with c3:
                c3.recv(4096)
                for i in range(3):
                    time.sleep(0.12)
                    c3.sendall(f'{{"event": "tick", "i": {i}}}\n'.encode("utf-8"))
                c3.sendall(b'{"event": "stopped"}\n')
            # Case 4: early close stream (never finishes, client closes after 1 event).
            c4, _ = eof_srv.accept()
            with c4:
                c4.recv(4096)
                c4.sendall(b'{"event": "first"}\n')
                time.sleep(0.2)

        eof_th = threading.Thread(target=serve_eof_cases, daemon=True)
        eof_th.start()
        c1_calls = [0]

        def alive_after_c1() -> bool:
            c1_calls[0] += 1
            if c1_calls[0] == 1:
                return True
            c1_done.wait(timeout=3.0)
            return False

        ans_eof = ipc.unix_json_call(
            eof_sock, {"method": "eof"}, timeout=5.0, poll_s=0.005,
            alive_fn=alive_after_c1,
        )
        self.assertEqual(ans_eof, {"ok": True, "eof": 1})
        c2_calls = [0]

        def alive_after_c2() -> bool:
            c2_calls[0] += 1
            if c2_calls[0] == 1:
                return True
            c2_done.wait(timeout=3.0)
            return False

        evs_eof = list(ipc.unix_json_stream(
            eof_sock, {"method": "burst"}, timeout=5.0, poll_s=0.005,
            alive_fn=alive_after_c2,
        ))
        self.assertEqual(evs_eof, [{"event": "a"}, {"event": "b"}])
        evs_inact = list(ipc.unix_json_stream(
            eof_sock, {"method": "ticks"}, timeout=0.25, poll_s=0.01, alive_fn=lambda: True,
        ))
        self.assertEqual(len(evs_inact), 4)
        self.assertEqual(evs_inact[-1]["event"], "stopped")
        gen_early = ipc.unix_json_stream(eof_sock, {"method": "early"}, timeout=5.0, poll_s=0.01, alive_fn=lambda: True)
        self.assertEqual(next(gen_early), {"event": "first"})
        gen_early.close()
        eof_th.join(timeout=3.0)

        # Large sendall (> SO_SNDBUF) against a peer that never reads and dies -> fast-fails with ConnectionError!
        snd_sock = ipc.short_sock_path("test-unix-snd")
        self.addCleanup(lambda: os.unlink(snd_sock) if os.path.exists(snd_sock) else None)
        snd_srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        snd_srv.bind(snd_sock)
        snd_srv.listen(2)
        self.addCleanup(snd_srv.close)
        accepted_conns: list[socket.socket] = []

        def accept_no_read():
            c, _ = snd_srv.accept()
            accepted_conns.append(c)

        acc_th = threading.Thread(target=accept_no_read, daemon=True)
        acc_th.start()
        snd_alive = [True]
        threading.Timer(0.05, lambda: snd_alive.__setitem__(0, False)).start()
        t_snd = time.perf_counter()
        with self.assertRaises(ConnectionError):
            ipc.unix_json_call(
                snd_sock, {"method": "huge", "blob": "x" * (4 * 1024 * 1024)},
                timeout=10.0, poll_s=0.01, alive_fn=lambda: snd_alive[0],
            )
        self.assertLess(time.perf_counter() - t_snd, 2.0)
        acc_th.join(timeout=2.0)
        for c in accepted_conns:
            c.close()


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

    def test_reap_all_preserves_kept_live_and_exited_children(self):
        """Input: a subprocess spawning one kept live child, one kept already-exited child
        (exit code 17), and one unkept job child, then calling `proc.reap_all(keep={...})`.
        Expected: only the unkept job child is killed/reaped; the kept live child survives and
        the kept exited child's status (17) is not stolen by `waitpid(-1)`."""
        script = (
            "import json, subprocess, sys, time\n"
            "from rsibox import proc\n"
            "slp = [sys.executable, '-c', 'import time; time.sleep(60)']\n"
            "helper = subprocess.Popen(slp)\n"
            "finished = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(17)'])\n"
            "time.sleep(0.2)\n"
            "job = subprocess.Popen(slp)\n"
            "reaped = proc.reap_all(timeout=5.0, keep={helper.pid, finished.pid})\n"
            "alive = helper.poll() is None\n"
            "helper.kill(); helper.wait()\n"
            "print(json.dumps([job.pid in reaped, alive, finished.wait()]))\n"
        )
        env = {**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=20)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout.strip()), [True, True, 17], out.stderr)

    def test_pg_alive_and_stop_session_guard_nonpositive_and_return_actual_state(self):
        """`pg_alive`, `stop_session`, and `kill_tree` reject `<= 0` without signaling the
        caller's own process group; `stop_session` caps post-SIGKILL wait and returns
        `not pg_alive(pid)`."""
        with mock.patch.object(proc.os, "killpg") as m_killpg, \
                mock.patch.object(proc.os, "kill") as m_kill:
            for bad in (0, -1, -os.getpgrp()):
                self.assertFalse(proc.pg_alive(bad))
                self.assertFalse(proc.stop_session(bad, grace=0.05))
                self.assertEqual(proc.kill_tree(bad), [])
            m_killpg.assert_not_called()
            m_kill.assert_not_called()

        # If process group stays alive even after SIGKILL, stop_session returns False and caps
        # the post-SIGKILL wait window relative to `grace`.
        signals_sent = []
        clock = {"t": 100.0}

        def fake_monotonic():
            return clock["t"]

        def fake_sleep(dt):
            clock["t"] += 2.0

        def fake_kill_tree(pid, sig, include_pg=True, proc_root="/proc"):
            signals_sent.append((pid, sig, include_pg, clock["t"]))
            return [pid]

        with mock.patch.object(proc, "pid_alive", return_value=False), \
                mock.patch.object(proc, "pg_alive", return_value=True), \
                mock.patch.object(proc, "kill_tree", side_effect=fake_kill_tree), \
                mock.patch.object(proc.time, "monotonic", side_effect=fake_monotonic), \
                mock.patch.object(proc.time, "sleep", side_effect=fake_sleep):
            ok = proc.stop_session(424242, grace=10.0)
        self.assertFalse(ok)
        self.assertEqual([s[1] for s in signals_sent], [proc.signal.SIGTERM, proc.signal.SIGKILL])
        # SIGTERM ran for 10s (100 -> 110); SIGKILL was capped at 5s (110 -> 116 <= 110 + 6).
        term_t, kill_t = signals_sent[0][3], signals_sent[1][3]
        self.assertEqual(kill_t - term_t, 10.0)
        self.assertLessEqual(clock["t"] - kill_t, 6.0)

        # Real session-leader child stops cleanly and returns True.
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        try:
            self.assertTrue(proc.pg_alive(child.pid))
            self.assertTrue(proc.stop_session(child.pid, grace=2.0, reap=child.poll))
            child.wait(timeout=3.0)
            self.assertFalse(proc.pg_alive(child.pid))
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=3.0)


class HooksAndHarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rsibox-hooks-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_stacked_import_hooks_and_overlay_compose_without_recursion(self):
        """Input: two `AfterImportHook`s stacked ahead of an `OverlayFinder`; importing a real
        package module on disk.
        Expected: no RecursionError; the module loads the overlaid text with original
        `co_filename` and populated `linecache`, and both callbacks fire after `exec_module`."""
        pkg = f"rsibox_hook_pkg_{next(_SEQ)}"
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
        """Input: call `bootstrap_livelab` when `RSIBOX_LIVELAB` is unset.
        Expected: resolves `core/livelab` on `sys.path` and leaves `os.environ` untouched."""
        clean_env = {k: v for k, v in os.environ.items() if k != "RSIBOX_LIVELAB"}
        with mock.patch.dict(os.environ, clean_env, clear=True):
            resolved = rsibox.bootstrap_livelab(__file__)
            self.assertEqual(resolved, os.path.abspath(LIVELAB))
            self.assertIn(resolved, sys.path)
            self.assertNotIn("RSIBOX_LIVELAB", os.environ)


class DiffsTests(unittest.TestCase):
    def test_apply_modifies_creates_and_refuses_invalid_or_stale_diffs(self):
        """Input: unified diffs that (a) edit a module, (b) create a new module and `__init__.py`,
        (c) fail context match, (d) delete a module, and (e) touch a non-Python file.
        Expected: valid diffs produce `{module: new_text}` with zero disk side effects;
        stale/deletion/non-Python diffs raise `ValueError`."""
        base = "def f(x):\n    return x + 1\n"
        edit_diff = (
            "--- a/pkg/mod.py\n"
            "+++ b/pkg/mod.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def f(x):\n"
            "-    return x + 1\n"
            "+    return x + 2\n"
        )
        self.assertEqual(diffs.paths(edit_diff), ["pkg/mod.py"])
        self.assertEqual(diffs.module_of("pkg/sub/__init__.py"), "pkg.sub")
        out = diffs.apply(edit_diff, lambda m: base if m == "pkg.mod" else None)
        self.assertEqual(out, {"pkg.mod": "def f(x):\n    return x + 2\n"})

        # Stale context refused without fuzz.
        with self.assertRaisesRegex(ValueError, "does not apply"):
            diffs.apply(edit_diff, lambda _m: "def f(x):\n    return x + 9\n")

        # Created module & package __init__.py supported.
        create_diff = (
            "--- /dev/null\n"
            "+++ b/pkg/sub/__init__.py\n"
            "@@ -0,0 +1 @@\n"
            "+FLAG = True\n"
        )
        self.assertEqual(diffs.apply(create_diff, lambda _m: None), {"pkg.sub": "FLAG = True\n"})

        # Deletion and non-Python files refused.
        del_diff = "--- a/pkg/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-X = 1\n"
        with self.assertRaisesRegex(ValueError, "deletes a module"):
            diffs.apply(del_diff, lambda _m: "X = 1\n")

        cfg_diff = "--- a/cfg.yaml\n+++ b/cfg.yaml\n@@ -1 +1 @@\n-a: 1\n+a: 2\n"
        with self.assertRaisesRegex(ValueError, "not modules"):
            diffs.apply(cfg_diff, lambda _m: None)


class HarnessHelpersTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rsibox-harness-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_append_jsonl_writes_lines_and_optionally_suppresses_errors(self):
        path = os.path.join(self.tmp, "receipts.jsonl")
        harness.append_jsonl(None, {"skip": 1})
        harness.append_jsonl(path, None)
        self.assertFalse(os.path.exists(path))
        harness.append_jsonl(path, {"n": 1})
        harness.append_jsonl(path, {"n": 2, "path": os.path.join(self.tmp, "x")})
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        self.assertEqual([r["n"] for r in rows], [1, 2])
        # Missing parent directory raises unless suppress_errors=True.
        bad = os.path.join(self.tmp, "missing-dir", "log.jsonl")
        with self.assertRaises(OSError):
            harness.append_jsonl(bad, {"ok": True})
        harness.append_jsonl(bad, {"ok": True}, suppress_errors=True)

    def test_source_stack_and_resolve_patch_files_handle_stacked_and_keyed_frames(self):
        disk = {"pkg.m": "X = 0\n"}
        d1 = "--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1 +1 @@\n-X = 0\n+X = 1\n"
        d2 = "--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1 +1 @@\n-X = 1\n+X = 2\n"

        # Plain frame mode (min_len=1 preserves base frame).
        frames = [{}]
        stack = harness.SourceStack(disk.get, frames, min_len=1)
        self.assertEqual(stack.resolve({"diff": d1}, against="live"), {"pkg.m": "X = 1\n"})
        stack.push({"pkg.m": "X = 1\n"})
        self.assertEqual(stack.resolve({"diff": d2}, against="live"), {"pkg.m": "X = 2\n"})
        self.assertEqual(stack.resolve({"diff": d1}, against="disk"), {"pkg.m": "X = 1\n"})
        with self.assertRaisesRegex(ValueError, "both"):
            stack.resolve({"files": {"pkg.m": "X = 9\n"}, "diff": d1})
        self.assertEqual(stack.pop(5), 1)
        self.assertEqual(len(frames), 1)
        self.assertEqual(stack.live_source("pkg.m"), "X = 0\n")

        # Keyed frame mode (`files_key="files"`, `min_len=0`, matching VllmLab._stack_files).
        keyed = []
        kstack = harness.SourceStack(disk.get, keyed, files_key="files", min_len=0)
        kstack.push({"id": "t1", "files": {"pkg.m": "X = 1\n"}})
        self.assertEqual(kstack.live_source("pkg.m"), "X = 1\n")
        self.assertEqual(kstack.resolve({"diff": d2}), {"pkg.m": "X = 2\n"})
        kstack.clear()
        self.assertEqual(keyed, [])

        # Missing base file raising FileNotFoundError maps cleanly to None for created modules.
        create = "--- /dev/null\n+++ b/pkg/new.py\n@@ -0,0 +1 @@\n+Y = 7\n"
        def raising_reader(_m):
            raise FileNotFoundError("absent")
        self.assertEqual(harness.resolve_patch_files({"diff": create}, raising_reader), {"pkg.new": "Y = 7\n"})

    def test_install_refusal_and_check_failure_shapes(self):
        f = {"check": "patch_invalid", "observed": "bad"}
        ref = harness.install_refusal(["patch_invalid"], [f], outcome="aborted", t0=time.perf_counter())
        self.assertEqual((ref["ok"], ref["id"], ref["outcome"], ref["codes"], ref["mechanisms"], ref["findings"]),
                         (False, None, "aborted", ["patch_invalid"], [], [f]))
        self.assertIsInstance(ref["cost_s"], float)
        ref_no_cost = harness.install_refusal(["patch_invalid"], [f])
        self.assertNotIn("cost_s", ref_no_cost)

        chk = harness.check_failure("timeout", [f], t0=time.perf_counter())
        self.assertEqual((chk["ok"], chk["error"], chk["output"], chk["proven"], chk["witness"], chk["findings"]),
                         (False, "timeout", None, None, {}, [f]))
        self.assertIsInstance(chk["cost_s"], float)
        chk_no_cost = harness.check_failure("lost", [f])
        self.assertNotIn("cost_s", chk_no_cost)

    def test_hooks_import_graph_dependents_and_purge_modules(self):
        import types
        tmp = tempfile.mkdtemp(prefix="rsibox-hooks-graph-")
        self.addCleanup(shutil.rmtree, tmp, True)
        pkg = os.path.join(tmp, "gpkg")
        os.makedirs(pkg)
        with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf-8") as fh:
            fh.write("")
        with open(os.path.join(pkg, "leaf.py"), "w", encoding="utf-8") as fh:
            fh.write("VAL = 1\n")
        with open(os.path.join(pkg, "mid.py"), "w", encoding="utf-8") as fh:
            fh.write("from . import leaf\n")
        with open(os.path.join(pkg, "top.py"), "w", encoding="utf-8") as fh:
            fh.write("import gpkg.mid\n")
        m_pkg = types.ModuleType("gpkg")
        m_pkg.__file__ = os.path.join(pkg, "__init__.py")
        m_leaf = types.ModuleType("gpkg.leaf")
        m_leaf.__file__ = os.path.join(pkg, "leaf.py")
        m_mid = types.ModuleType("gpkg.mid")
        m_mid.__file__ = os.path.join(pkg, "mid.py")
        m_top = types.ModuleType("gpkg.top")
        m_top.__file__ = os.path.join(pkg, "top.py")
        m_pkg.leaf, m_pkg.mid, m_pkg.top = m_leaf, m_mid, m_top
        mods = {"gpkg": m_pkg, "gpkg.leaf": m_leaf, "gpkg.mid": m_mid, "gpkg.top": m_top}
        cache = {}
        graph = hooks.import_graph(("gpkg",), mods, cache=cache)
        self.assertEqual(graph["gpkg.mid"], {"gpkg", "gpkg.leaf"})
        self.assertEqual(graph["gpkg.top"], {"gpkg.mid"})
        self.assertEqual(hooks.dependents({"gpkg.leaf"}, graph), {"gpkg.leaf", "gpkg.mid", "gpkg.top"})
        hooks.purge_modules(["gpkg.leaf", "gpkg.mid"], mods)
        self.assertNotIn("gpkg.leaf", mods)
        self.assertFalse(hasattr(m_pkg, "leaf"))
        self.assertTrue(hasattr(m_pkg, "top"))

    def test_purge_modules_preserves_non_module_attrs_and_resolve_import_clamps_level(self):
        """`purge_modules` preserves non-module attributes on a parent package when the target
        was not loaded (or was shadowed), and `_resolve_import` clamps `node.level` without
        negative-slice wraparound."""
        import ast
        import types

        m_pkg = types.ModuleType("upkg")
        m_pkg.CONST = 123
        sentinel = object()
        m_pkg.shadow = sentinel
        m_shadow = types.ModuleType("upkg.shadow")
        mods = {"upkg": m_pkg, "upkg.shadow": m_shadow}

        hooks.purge_modules(["upkg.CONST", "upkg.shadow"], mods)
        self.assertEqual(m_pkg.CONST, 123)
        self.assertIs(m_pkg.shadow, sentinel)
        self.assertNotIn("upkg.shadow", mods)

        over_bare = ast.parse("from .... import x").body[0]
        self.assertEqual(hooks._resolve_import("a.b.c", False, over_bare), [])
        over_mod = ast.parse("from ....sub import x").body[0]
        self.assertEqual(hooks._resolve_import("a.b.c", False, over_mod), ["sub", "sub.x"])


if __name__ == "__main__":
    unittest.main()
