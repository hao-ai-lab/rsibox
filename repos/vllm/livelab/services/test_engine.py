"""Golden cases for the engine launcher (engine.py). CPU only: a small fake `vllm` stands in for
the server, a stub of vLLM's argument parser (STUB_PARSER: the dests and defaults the launcher
reads) stands in for vLLM's when the launcher reads the argv, and a real zygote.py runs with
torch and vLLM stubbed out. The cases that need vLLM's real parser are in test_vllm_args.py.

Run: python3 -m pytest repos/vllm/livelab/services/test_engine.py -v
"""
import json, os, signal, socket, subprocess, sys, tempfile, time, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine, zygote  # noqa: E402

ENGINE = str(HERE / "engine.py")

# A stand-in for `vllm serve`: records its argv and two env variables, then serves /health
# (200) on --port after DELAY seconds, like a server that loads for a while.
FAKE_VLLM = r'''#!{py}
import http.server, json, os, sys, time
argv = sys.argv[1:]
open(os.environ["FAKE_OUT"], "w").write(json.dumps({{"argv": argv, "env": {{
    k: os.environ.get(k) for k in ("VLLM_CACHE_ROOT", "TRITON_CACHE_DIR")}}}}))
time.sleep(float(os.environ.get("FAKE_DELAY", "0")))
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404); self.end_headers()
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(argv[argv.index("--port") + 1])), H).serve_forever()
'''

# Starts the real zygote.py with torch stubbed (CUDA "not initialized") and vLLM made
# unimportable, so its import list fails fast and the zygote serves ping/check on a CPU box.
FAKE_PYTHON = r'''#!{py}
import runpy, sys, types
torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(is_initialized=lambda: False)
sys.modules["torch"] = torch
sys.modules["vllm"] = None
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''


# vLLM's parser, cut down to what the launcher reads (same dests, defaults, the model_tag
# positional and the -tp alias), for a CPU box without vLLM. On PYTHONPATH it shadows a real one.
# Like vLLM 0.15.1 with VLLM_LOGGING_LEVEL=DEBUG, it logs to stdout while it is imported.
STUB_PARSER = {
    "vllm/entrypoints/common/cli_args.py": '''
print("DEBUG [platforms/__init__.py:61] Checking if CUDA platform is available.")

def make_arg_parser(p):
    p.add_argument("model_tag", nargs="?")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host")
    p.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    p.add_argument("--load-format", default="auto")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--worker-cls", default="auto")
    p.add_argument("--max-model-len")
    p.add_argument("--model-loader-extra-config", type=__import__("json").loads, default={})
    return p
''',
    "vllm/utils/argparse_utils.py": "import argparse\n"
                                    "class FlexibleArgumentParser(argparse.ArgumentParser): pass\n",
}


def stub_env(d, **extra):
    """os.environ plus PYTHONPATH with STUB_PARSER first."""
    root = Path(d, "stubparser")
    for name, text in STUB_PARSER.items():
        p = root / name
        for parent in [x for x in p.relative_to(root).parents if str(x) != "."]:
            (root / parent).mkdir(parents=True, exist_ok=True)
            (root / parent / "__init__.py").touch()
        p.write_text(text)
    pp = os.environ.get("PYTHONPATH")
    return {**os.environ, "PYTHONPATH": str(root) + (os.pathsep + pp if pp else ""), **extra}


def _script(d, name, text):
    p = os.path.join(d, name)
    Path(p).write_text(text.format(py=sys.executable))
    os.chmod(p, 0o755)
    return p


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _short_dir():
    """AF_UNIX paths are limited to 107 bytes; the default temp dir can be longer."""
    return tempfile.mkdtemp(prefix="eng", dir="/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")


def _one_json_line(stdout):
    """engine.py's stdout contract: one JSON line (callers read it line by line)."""
    lines = stdout.splitlines()
    assert len(lines) == 1, f"{len(lines)} lines on stdout, expected one JSON line:\n{stdout}"
    return json.loads(lines[0])


def _stop(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_port_in_use_means_a_listener_answers():
    """A listening port is in use; a port that is only bound (no listen) or free is not: the
    check asks whether a /health probe would reach another server, not whether bind() works."""
    listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1)
    bound = socket.socket(); bound.bind(("127.0.0.1", 0))
    try:
        assert engine.port_in_use(listener.getsockname()[1]) is True
        assert engine.port_in_use(bound.getsockname()[1]) is False
        assert engine.port_in_use(_free_port()) is False
    finally:
        listener.close(); bound.close()


def test_busy_port_is_a_finding_and_nothing_starts():
    """Input: a listener holds 127.0.0.1:P; `engine.py start --port P`. Expect: a port_in_use
    finding, exit 2, the fake vllm never ran (without the check wait_healthy read the other
    server's /health and reported a start healthy after ~0.01 s). Not covered: a port taken
    after the check (a race). The finding is one JSON line on stdout (before: indented)."""
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]
    with tempfile.TemporaryDirectory() as d:
        fake = _script(d, "vllm", FAKE_VLLM)
        out = os.path.join(d, "ran.json")
        r = subprocess.run([sys.executable, ENGINE, "start", "m", "--port", str(port),
                            "--cache-root", d, "--log", os.path.join(d, "serve.log"),
                            "--vllm", fake], capture_output=True, text=True, timeout=60,
                           env=stub_env(d, FAKE_OUT=out))
        srv.close()
        assert r.returncode == 2, r.stdout + r.stderr
        assert _one_json_line(r.stdout)["finding"]["check"] == "port_in_use"
        assert not os.path.exists(out)


def test_wait_healthy_returns_none_as_soon_as_the_server_exits():
    """Input: a direct start whose server exits at once with an argument error (no Traceback in
    the log) and is not reaped yet, so /proc/<pid> still exists as a zombie. Expect: None within
    a few seconds, not after the full timeout (before: it polled /health until the timeout,
    an hour by default)."""
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "serve.log")
        with open(log, "w") as f:
            p = subprocess.Popen([sys.executable, "-c", "import sys; print('vllm: error: "
                                  "unrecognized arguments: --bogus'); sys.exit(2)"],
                                 stdout=f, stderr=subprocess.STDOUT)
        time.sleep(0.5)
        assert os.path.exists(f"/proc/{p.pid}")          # a zombie: exited, not reaped
        t = time.time()
        assert engine.wait_healthy(_free_port(), p.pid, log, timeout=30, poll=0.1) is None
        assert time.time() - t < 5
        p.wait()


def test_direct_start_passes_the_vllm_argv_through_and_reports_health_seconds():
    """Input: `engine.py start` with launcher flags and vLLM flags mixed, `-tp 2` and a port
    only vLLM's parser knows, and a fake vllm that answers /health after 1.5 s. Expect: the
    server got the vLLM argv verbatim and in order, plus `--host 127.0.0.1`, and nothing else
    (before: the launcher added its own `--tensor-parallel-size 1` and `--load-format auto`);
    cache roots under --cache-root; the report reads model, port and tp 2 back from the parser
    and healthy_after_s >= 1.5 on that port. Not: vLLM's real parser (test_vllm_args.py)."""
    with tempfile.TemporaryDirectory() as d:
        fake = _script(d, "vllm", FAKE_VLLM)
        out, port = os.path.join(d, "ran.json"), _free_port()
        vargv = ["m", "-tp", "2", "--port", str(port), "--enforce-eager"]
        r = subprocess.run([sys.executable, ENGINE, "start", "--cache-root",
                            os.path.join(d, "c"), vargv[0], "--log", os.path.join(d, "serve.log"),
                            *vargv[1:3], "--vllm", fake, "--timeout", "30", *vargv[3:]],
                           capture_output=True, text=True, timeout=60,
                           env=stub_env(d, FAKE_OUT=out, FAKE_DELAY="1.5"))
        rep = json.loads(r.stdout)
        _stop(rep["pid"])
        assert r.returncode == 0, r.stdout + r.stderr
        assert rep["healthy_after_s"] >= 1.5 and rep["port"] == port
        assert rep["vllm_args"]["args"]["model"] == "m"
        assert rep["vllm_args"]["args"]["tensor_parallel_size"] == 2
        ran = json.loads(Path(out).read_text())
        assert ran["argv"] == ["serve", *vargv, "--host", "127.0.0.1"]
        assert ran["env"]["VLLM_CACHE_ROOT"] == str(Path(d, "c", "vllm").resolve())


def test_an_argv_vllm_would_not_serve_as_meant_is_a_finding_and_nothing_starts():
    """Input: three starts: no model (only `--port`), a flag vLLM rejects, `--host 0.0.0.0`.
    Expect: exit 2, findings vllm_model (naming vLLM's default model), vllm_args (vLLM's
    message) and vllm_host (naming 0.0.0.0); the fake vllm never ran. Before: the launcher
    passed what it had and the server served vLLM's default model, or exited on the flag.
    Each refusal is one JSON line on stdout, as engine.py's docstring promises (before: an
    indented multi-line dump)."""
    with tempfile.TemporaryDirectory() as d:
        fake = _script(d, "vllm", FAKE_VLLM)
        out = os.path.join(d, "ran.json")
        seen = {}
        for vargv in (["--port", "8123"], ["m", "--no-such-flag"], ["m", "--host", "0.0.0.0"]):
            r = subprocess.run([sys.executable, ENGINE, "start", "--cache-root", d, "--log",
                                os.path.join(d, "serve.log"), "--vllm", fake, *vargv],
                               capture_output=True, text=True, timeout=60,
                               env=stub_env(d, FAKE_OUT=out))
            assert r.returncode == 2, r.stdout + r.stderr
            f = _one_json_line(r.stdout)["findings"][0]
            seen[f["check"]] = f["observed"]
        assert "Qwen/Qwen3-0.6B" in seen["vllm_model"]
        assert "--no-such-flag" in seen["vllm_args"]
        assert "0.0.0.0" in seen["vllm_host"]
        assert not os.path.exists(out)


def test_zygote_runs_with_the_engine_env():
    """Input: `engine.py zygote` starts a real zygote.py; then the zygote's own env check is
    asked about (1) the env a start with the same options builds and (2) one with another
    cache root. Expect: (1) ok with no env finding, also through `start --zygote-check`
    (exit 0); (2) an env finding naming VLLM_CACHE_ROOT, open and so not in
    "accepted_findings" (before: every finding was listed there, accepted or not); with
    accept=["env"] it is ok and the finding moves to "accepted_findings". Proves the zygote is
    started with engine_env(); not: a fork of a real `vllm serve` (GPU receipt)."""
    d = _short_dir()
    sock = os.path.join(d, "z.sock")
    zpy = _script(d, "python", FAKE_PYTHON)
    cache = os.path.join(d, "c")
    base = dict(os.environ)     # the launcher's env, fixed (pytest edits os.environ in-process)
    r = subprocess.run([sys.executable, ENGINE, "zygote", "--python", zpy, "--zygote", sock,
                        "--cache-root", cache, "--log", os.path.join(d, "z.log")],
                       capture_output=True, text=True, timeout=120, env=base)
    rep = json.loads(r.stdout)
    try:
        assert rep["ready"] is True, r.stdout + r.stderr + Path(d, "z.log").read_text()
        a, _ = engine.parse(["start", "--cache-root", cache, "--log", "x"])
        same = zygote.check(sock, engine.engine_env(a, base))
        assert same["ok"] is True and same["accepted_findings"] == [], same
        a2, _ = engine.parse(["start", "--cache-root", os.path.join(d, "other"), "--log", "x"])
        other = zygote.check(sock, engine.engine_env(a2, base))
        assert other["ok"] is False
        assert other["findings"][0]["check"] == "env"
        assert "VLLM_CACHE_ROOT" in other["findings"][0]["observed"]
        assert other["accepted_findings"] == [], other
        acc = zygote.check(sock, engine.engine_env(a2, base), accept=["env"])
        assert acc["ok"] is True and acc["findings"] == [], acc
        assert [f["check"] for f in acc["accepted_findings"]] == ["env"], acc
        chk = subprocess.run([sys.executable, ENGINE, "start", "m", "--port",
                              str(_free_port()), "--cache-root", cache, "--log",
                              os.path.join(d, "s.log"), "--zygote", sock, "--zygote-check"],
                             capture_output=True, text=True, timeout=60, env=base)
        assert chk.returncode == 0, chk.stdout + chk.stderr
        assert json.loads(chk.stdout)["zygote_check"]["ok"] is True
    finally:
        _stop(rep["pid"])


def test_zygote_parse_with_an_edit_it_cannot_refresh_is_a_stale_finding():
    """Input: a real zygote.py (torch stubbed, vLLM absent) that pre-imported a module outside
    the refresh packages; the module is then edited; `engine.py start --zygote S m`. Expect:
    exit 2 with a `stale` finding naming that file, nothing forked (before: the parse ran on
    the pre-edit code, so an edit to vLLM's argument code a refresh cannot reach was read with
    the old flags). With `--zygote-accept stale` the parse runs: here vLLM is absent, so the
    parse child fails to import its parser, a `zygote_parse` finding. Not: a refresh of an
    edited vLLM module (the next case)."""
    d = _short_dir()
    sock, cache = os.path.join(d, "z.sock"), os.path.join(d, "c")
    zpy = _script(d, "python", FAKE_PYTHON)
    lib = Path(d, "lib"); lib.mkdir()
    (lib / "zt_edited.py").write_text("X = 1\n")
    Path(d, "mods.txt").write_text("zt_edited\n")
    env = {**os.environ, "PYTHONPATH": str(lib)}
    r = subprocess.run([sys.executable, ENGINE, "zygote", "--python", zpy, "--zygote", sock,
                        "--cache-root", cache, "--log", os.path.join(d, "z.log"),
                        "--zygote-modules", os.path.join(d, "mods.txt")],
                       capture_output=True, text=True, timeout=120, env=env)
    rep = json.loads(r.stdout)
    try:
        assert rep["ready"] is True, r.stdout + r.stderr + Path(d, "z.log").read_text()
        (lib / "zt_edited.py").write_text("X = 2  # edited\n")
        start = [sys.executable, ENGINE, "start", "--cache-root", cache, "--log",
                 os.path.join(d, "s.log"), "--zygote", sock, "m", "--port", str(_free_port())]
        out = subprocess.run(start, capture_output=True, text=True, timeout=60, env=env)
        assert out.returncode == 2, out.stdout + out.stderr
        f = json.loads(out.stdout)["findings"]
        assert [x["check"] for x in f] == ["stale"] and "zt_edited.py" in f[0]["observed"]
        out = subprocess.run(start + ["--zygote-accept", "stale"], capture_output=True,
                             text=True, timeout=60, env=env)
        assert out.returncode == 2, out.stdout + out.stderr
        f = json.loads(out.stdout)["findings"]
        assert [x["check"] for x in f] == ["zygote_parse"] and "vllm" in f[0]["observed"]
    finally:
        _stop(rep["pid"])


def test_zygote_parse_reads_an_edit_to_vllms_argument_code():
    """Input: a real zygote.py (torch stubbed) that pre-imported STUB_PARSER as vLLM's parser
    (a `vllm` package, so its edits refresh); then three parses of `m --new-flag 1`: before any
    edit, after the parser gains --new-flag, after an edit that does not import. Expect: (1)
    vLLM's rejection `unrecognized arguments: --new-flag`; (2) ok, and the reply's "refreshed"
    names the parser's module (before: the child parsed with the zygote's pre-edit parser
    and rejected the flag); (3) `engine.py start --zygote` exits 2 with a `zygote_parse`
    finding carrying the SyntaxError (before: a `vllm_args` finding, "fix the vLLM flags"),
    and nothing started; (4) a start of `m` with another cache root: the parse passes, the
    zygote refuses the fork with an env finding, exit 2 and one JSON line on stdout (before:
    an indented multi-line dump). Not: a refresh of the real vLLM's argument modules (they are
    the same refresh as a fork's, test_zygote_refresh.py)."""
    d = _short_dir()
    sock, cache = os.path.join(d, "z.sock"), os.path.join(d, "c")
    zpy = _script(d, "python", FAKE_PYTHON.replace('sys.modules["vllm"] = None\n', ""))
    Path(d, "mods.txt").write_text("vllm.entrypoints.common.cli_args\nvllm.utils.argparse_utils\n")
    env = stub_env(d)
    cli = Path(d, "stubparser", "vllm/entrypoints/common/cli_args.py")
    r = subprocess.run([sys.executable, ENGINE, "zygote", "--python", zpy, "--zygote", sock,
                        "--cache-root", cache, "--log", os.path.join(d, "z.log"),
                        "--zygote-modules", os.path.join(d, "mods.txt")],
                       capture_output=True, text=True, timeout=120, env=env)
    rep = json.loads(r.stdout)
    try:
        assert rep["ready"] is True, r.stdout + r.stderr + Path(d, "z.log").read_text()
        argv = ["m", "--new-flag", "1"]
        before = zygote.parse(sock, argv)
        assert before["ok"] is False and "unrecognized arguments: --new-flag" in before["error"]
        assert not before.get("zygote_error"), before
        text = cli.read_text()
        cli.write_text(text.replace('    return p\n', '    p.add_argument("--new-flag")\n'
                                                     '    return p\n'))
        after = zygote.parse(sock, argv)
        assert after["ok"] is True and after["args"]["model"] == "m", after
        assert after["refreshed"]["changed"] == ["vllm.entrypoints.common.cli_args"], after
        cli.write_text(text + "\ndef broken(:\n")
        out = subprocess.run([sys.executable, ENGINE, "start", "--cache-root", cache, "--log",
                              os.path.join(d, "s.log"), "--zygote", sock, *argv],
                             capture_output=True, text=True, timeout=60, env=env)
        assert out.returncode == 2, out.stdout + out.stderr
        f = json.loads(out.stdout)["findings"]
        assert [x["check"] for x in f] == ["zygote_parse"], f
        assert "SyntaxError" in f[0]["observed"], f
        assert not os.path.exists(os.path.join(d, "s.log"))
        cli.write_text(text)
        out = subprocess.run([sys.executable, ENGINE, "start", "--cache-root",
                              os.path.join(d, "other"), "--log", os.path.join(d, "o.log"),
                              "--zygote", sock, "m", "--port", str(_free_port())],
                             capture_output=True, text=True, timeout=60, env=env)
        assert out.returncode == 2, out.stdout + out.stderr
        zf = _one_json_line(out.stdout)["zygote_findings"]
        assert [x["check"] for x in zf] == ["env"], zf
    finally:
        _stop(rep["pid"])


def test_fork_check_reads_vllms_spawn_fallback():
    """Input: a server log where vLLM fell back to spawn, and one where it did not. Expect:
    spawn_forced and a fork finding for the first; none for the second, and fork_check reports
    only that signal (before: it also reported a count of a "spawned interpreter" line and a
    FASTRSI_FORKSAFE "probe" that nothing on main prints or sets, so the count was always 0)."""
    with tempfile.TemporaryDirectory() as d:
        bad, good = os.path.join(d, "bad.log"), os.path.join(d, "good.log")
        Path(bad).write_text("WARNING We must use the `spawn` multiprocessing start method. "
                             "Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. Reasons: "
                             "CUDA is initialized\n")
        Path(good).write_text("INFO Application startup complete.\n")
        fb, fg = engine.fork_check(bad), engine.fork_check(good)
        assert fb["spawn_forced"] is True and engine.fork_finding(fb)["check"] == "fork"
        assert fg == {"spawn_forced": False} and engine.fork_finding(fg) is None
