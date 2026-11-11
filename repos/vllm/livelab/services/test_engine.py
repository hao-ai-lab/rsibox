"""Golden cases for the engine launcher (engine.py). CPU only: a small fake `vllm` stands in for
the server, a stub of vLLM's argument parser (STUB_PARSER: the dests and defaults the launcher
reads) stands in for vLLM's when the launcher reads the argv, and a real zygote.py runs with
torch and vLLM stubbed out. The cases that need vLLM's real parser are in test_vllm_args.py.

Run: python3 -m pytest repos/vllm/livelab/services/test_engine.py -v
"""
import json, os, signal, socket, subprocess, sys, tempfile, time
import pytest
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
    p.add_argument("-pp", "--pipeline-parallel-size", type=int, default=1)
    p.add_argument("-dp", "--data-parallel-size", type=int, default=1)
    p.add_argument("--enable-expert-parallel", action="store_true")
    p.add_argument("--load-format", default="auto")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--worker-cls", default="auto")
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--max-num-batched-tokens", type=int)
    p.add_argument("--max-num-seqs", type=int)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--enable-prefix-caching", action="store_true")
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("-q", "--quantization")
    p.add_argument("--moe-backend", default="auto")
    p.add_argument("--linear-backend", default="auto")
    p.add_argument("-cc", "--compilation-config", type=__import__("json").loads, default={})
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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only: needs /proc/<pid> (engine.py and zygote read process state from /proc)")
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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only: needs /proc/<pid> (engine.py and zygote read process state from /proc)")
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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only: needs /proc/<pid> (engine.py and zygote read process state from /proc)")
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
        a, _ = engine.parse(["start", "--zygote", sock, "--cache-root", cache, "--log", "x"])
        same = zygote.check(sock, engine.engine_env(a, base))
        assert same["ok"] is True and same["accepted_findings"] == [], same
        a2, _ = engine.parse(["start", "--zygote", sock, "--cache-root", os.path.join(d, "other"),
                              "--log", "x"])
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


def _stub_parser():
    """STUB_PARSER's vLLM parser, in this process, plus two flags as vLLM declares them: a
    boolean with a --no- form and a flag that takes one or more values."""
    import argparse, vllm_args
    code = {}
    exec(STUB_PARSER["vllm/entrypoints/common/cli_args.py"].split("\n", 2)[2], code)
    parser = code["make_arg_parser"](argparse.ArgumentParser(prog="vllm serve"))
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction)
    parser.add_argument("--served-model-name", nargs="+")
    return lambda argv: vllm_args.parse(argv, parser)


def test_an_override_that_moves_the_model_is_a_finding():
    """Input: running argvs whose model the merge cannot keep apart from a flag
    (`--trust-remote-code M`: a boolean before the positional model; `--served-model-name a b`:
    two values), and the overrides `--model N`, `--no-trust-remote-code` and
    `--served-model-name c`. Action: vllm_args.override, both argvs parsed, override_finding.
    Expect: a `vllm_args_override` finding for each (the merged argv would serve the old
    model, vLLM's default or "b"), and none for `M --trust-remote-code --port 8000` + `--model
    N`. Before: reconfigure stopped the engine and restarted it on the wrong model. Does not
    prove: a merge that keeps the model but moves another flag's values."""
    import vllm_args
    parse = _stub_parser()
    cases = (
        (["--trust-remote-code", "Qwen/Qwen3-8B", "--port", "8000"], ["--model", "N"]),
        (["--trust-remote-code", "Qwen/Qwen3-8B", "--port", "8000"], ["--no-trust-remote-code"]),
        (["--model", "Qwen/Qwen3-8B", "--served-model-name", "a", "b"],
         ["--served-model-name", "c"]),
    )
    for base, patch in cases:
        old, new = parse(base), parse(vllm_args.override(base, patch))
        assert old["ok"] and new["ok"], (old, new)
        finding = vllm_args.override_finding(patch, old, new)
        assert finding and finding["check"] == "vllm_args_override", (base, patch, new["args"])
    base, patch = ["Qwen/Qwen3-8B", "--trust-remote-code", "--port", "8000"], ["--model", "N"]
    new = parse(vllm_args.override(base, patch))
    assert new["args"]["model"] == "N"
    assert vllm_args.override_finding(patch, parse(base), new) is None


def test_a_reconfigure_value_vllm_cannot_run_is_a_finding():
    """Input: a merged argv with `--max-num-seqs 0` (vLLM's parser takes any int). Action:
    engine.value_finding. Expect: a `vllm_value` finding naming max_num_seqs; none for 8.
    Before: the live agent refused 0, so reconfigure fell back to a restart and stopped the
    running engine for a server that cannot run with that value."""
    parse = _stub_parser()
    finding = engine.value_finding(parse(["M", "--max-num-seqs", "0"]))
    assert finding["check"] == "vllm_value" and "max_num_seqs" in finding["observed"][0]
    assert engine.value_finding(parse(["M", "--max-num-seqs", "8"])) is None


def test_zygote_env_pins_cute_dsl_arch_without_fork(monkeypatch):
    """Input: a fake nvidia-smi that reports compute capability 10.0; the launcher arguments of
    `engine.py zygote --zygote S` and of `engine.py start --zygote S`, both without --fork.
    Action: engine_env() for each. Expect: both set CUTE_DSL_ARCH=sm_100a, so the zygote's
    import of CuTe DSL (through FlashInfer) does not call cuInit, and both envs agree (no `env`
    finding). Before: only --fork set it, and a zygote without --fork on a GPU host initialized
    the CUDA driver at import and stopped. Does not prove: the import on a real GPU (receipt)."""
    work_dir = _short_dir()
    smi_dir = os.path.join(work_dir, "bin")
    os.makedirs(smi_dir)
    _script(smi_dir, "nvidia-smi", "#!{py}\nprint('10.0')\n")
    monkeypatch.setenv("PATH", smi_dir + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    sock = os.path.join(work_dir, "z.sock")
    zyg, _ = engine.parse(["zygote", "--zygote", sock, "--cache-root", work_dir, "--log", "x"])
    srv, _ = engine.parse(["start", "--zygote", sock, "--cache-root", work_dir, "--log", "x"])
    direct, _ = engine.parse(["start", "--cache-root", work_dir, "--log", "x"])
    arch = [engine.engine_env(launcher_args).get("CUTE_DSL_ARCH") for launcher_args in (zyg, srv, direct)]  # not the env
    assert arch == ["sm_100a", "sm_100a", None]          # a direct start without --fork: unset


def test_zygote_child_applies_an_accepted_env_difference():
    """Input: a zygote started with KEEP=start, CHANGE=old, GONE=x whose imports then set KEEP
    (as torch or vLLM set variables at import), and a fork request with CHANGE=new, ADDED=a,
    GONE unset and KEEP=start, with the `env` finding accepted. Action: zygote._child in a real
    forked process, with a stand-in `vllm.entrypoints.cli.main` that records os.environ.
    Expect: the server sees CHANGE=new and ADDED=a, no GONE, and KEEP keeps its import-time
    value. Before: only VOLATILE keys were copied, so the server ran with CHANGE=old, no ADDED
    and GONE=x although the finding said accepting it applies them. Does not prove: which
    variables vLLM reads at import (an accepted one of those keeps the zygote's import)."""
    import types
    work_dir = _short_dir()
    out = os.path.join(work_dir, "env.json")
    pid = os.fork()
    if pid == 0:
        try:
            cli = types.ModuleType("vllm.entrypoints.cli.main")
            cli.main = lambda: Path(out).write_text(json.dumps(dict(os.environ)))
            sys.modules["vllm.entrypoints.cli.main"] = cli
            zygote.START_ENV = {"KEEP": "start", "CHANGE": "old", "GONE": "x"}
            os.environ.update(KEEP="set-by-import", CHANGE="old", GONE="x")
            req = {"argv": ["vllm", "serve"], "log": os.path.join(work_dir, "child.log"), "cwd": work_dir,
                   "env": {"KEEP": "start", "CHANGE": "new", "ADDED": "a"}, "accept": ["env"]}
            zygote._child(req, [])
        finally:
            os._exit(3)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, Path(work_dir, "child.log").read_text()
    seen = json.loads(Path(out).read_text())
    seen = {k: seen.get(k) for k in ("KEEP", "CHANGE", "ADDED", "GONE")}   # not the whole env
    assert seen == {"KEEP": "set-by-import", "CHANGE": "new", "ADDED": "a", "GONE": None}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only: needs /proc/<pid> (engine.py and zygote read process state from /proc)")
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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only: needs /proc/<pid> (engine.py and zygote read process state from /proc)")
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


# A stand-in `vllm serve` WITH the engine core's live agent: it builds a minimal
# vllm.v1.engine.core module (EngineCoreProc with the three methods live_reconfig patches),
# lets live_reconfig patch it exactly as in a real engine, and runs the busy loop in a child
# process (as vLLM's engine core is the API server's child). The scheduler is a plain object
# with vLLM's attribute names; the loop blocks on the input queue like vLLM's when idle.
FAKE_VLLM_LIVE = r'''#!{py}
import http.server, json, multiprocessing as mp, os, queue, sys, threading, time, types
argv = sys.argv[1:]
open(os.environ["FAKE_OUT"], "w").write(json.dumps({{"argv": argv}}))
def flag(name, default):
    return int(argv[argv.index(name) + 1]) if name in argv else default
def core_main():
    import live_reconfig
    mod = types.ModuleType("vllm.v1.engine.core")
    class EngineCoreProc:
        def __init__(self):
            self.input_queue = queue.Queue()
            self.scheduler = types.SimpleNamespace(
                max_num_running_reqs=flag("--max-num-seqs", 256),
                max_num_scheduled_tokens=flag("--max-num-batched-tokens", 8192),
                running=[], waiting=[])
            self.vllm_config = types.SimpleNamespace(
                scheduler_config=types.SimpleNamespace(enable_chunked_prefill=True),
                model_config=types.SimpleNamespace(max_model_len=flag("--max-model-len", 4096)),
                parallel_config=types.SimpleNamespace(data_parallel_size=1))
        def _handle_client_request(self, t, r):
            pass
        def _process_engine_step(self):
            return False
        def run_busy_loop(self):
            while True:
                self._handle_client_request(*self.input_queue.get())
                self._process_engine_step()
    mod.EngineCoreProc = EngineCoreProc
    live_reconfig._patch_core(mod)
    EngineCoreProc().run_busy_loop()
if __name__ == "__main__":
    sys.path.insert(0, os.environ["LIVELAB_SERVICES"])
    mp.set_start_method("fork")
    mp.Process(target=core_main, daemon=True).start()
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404); self.end_headers()
        def log_message(self, *a): pass
    http.server.HTTPServer(("127.0.0.1", flag("--port", 8000)), H).serve_forever()
'''


def _weights_daemon(work_dir):
    """nvidia-smi stub and a weights daemon socket for GPU-fake-0 (the --weights probe)."""
    from weights import socket_name
    smi_dir = os.path.join(work_dir, "bin")
    os.makedirs(smi_dir, exist_ok=True)
    _script(smi_dir, "nvidia-smi",
            "#!{py}\nimport sys\nids = sys.argv[sys.argv.index('-i') + 1].split(',') "
            "if '-i' in sys.argv else ['0']\nprint('\\n'.join(f'GPU-fake-{{i}}' for i in ids))\n")
    sock_dir = os.path.join(work_dir, "s")
    os.makedirs(sock_dir)
    daemon = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    daemon.bind(os.path.join(sock_dir, socket_name("GPU-fake-0")))
    daemon.listen(4)
    return smi_dir, sock_dir, daemon


def _reconfigure(cache_dir, log, pid, env, *flags):
    proc = subprocess.run(
        [sys.executable, ENGINE, "reconfigure", "--pid", str(pid), "--cache-root", cache_dir,
         "--log", log, *flags], capture_output=True, text=True, timeout=90, env=env)
    return proc, _one_json_line(proc.stdout)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only: needs /proc/<pid> (engine.py and zygote read process state from /proc)")
def test_reconfigure_without_a_live_consumer_restarts_and_says_so():
    """Input: an engine whose process has no live agent (the plain fake `vllm`), started with
    `--weights --kvplan` (`m --port P --max-model-len 4096 --enforce-eager`), then:
    1. `--max-num-seqs 4` (in_process_instant, a live field) -> nobody in the engine can apply
       it, so the launcher restarts: path "restart", restarted True, a new pid, and
       restart_reason names the missing live agent. Before: it wrote <pid>.reconfig.json that
       nothing read, recorded the new argv as the engine's, and reported restarted False.
    2. `--no-enforce-eager` (in_process_graph: no live consumer at all) -> restart too.
    3. `--gpu-memory-utilization 0.85` (warm_weights_ipc) -> restart with merged flags."""
    work_dir = _short_dir()
    smi_dir, sock_dir, daemon = _weights_daemon(work_dir)
    fake_vllm = _script(work_dir, "vllm", FAKE_VLLM)
    out_file = os.path.join(work_dir, "ran.json")
    cache_dir = os.path.join(work_dir, "c")
    port = _free_port()
    env = stub_env(work_dir, FAKE_OUT=out_file, CUDA_VISIBLE_DEVICES="0",
                   PATH=smi_dir + os.pathsep + os.environ["PATH"])
    start_proc = subprocess.run(
        [sys.executable, ENGINE, "start", "--cache-root", cache_dir, "--log",
         os.path.join(work_dir, "s1.log"), "--vllm", fake_vllm, "--weights",
         "--weights-socket-dir", sock_dir, "m", "--kvplan", "--port", str(port),
         "--max-model-len", "4096", "--enforce-eager"],
        capture_output=True, text=True, timeout=60, env=env)
    pids = [_one_json_line(start_proc.stdout)["pid"]]
    try:
        assert start_proc.returncode == 0, start_proc.stdout + start_proc.stderr
        for i, flags in enumerate((["--max-num-seqs", "4"], ["--no-enforce-eager"],
                                   ["--gpu-memory-utilization", "0.85"])):
            proc, rep = _reconfigure(cache_dir, os.path.join(work_dir, f"r{i}.log"), pids[-1],
                                     env, *flags)
            assert proc.returncode == 0, proc.stdout + proc.stderr
            rc = rep["reconfig"]
            assert rc["path"] == "restart" and rc["restarted"] is True, rc
            assert rc["replaced_pid"] == pids[-1] and rep["pid"] != pids[-1]
            assert zygote._exited(pids[-1]), "the replaced engine must be gone"
            assert rc["restart_reason"], rc
            assert rep["healthy_after_s"] is not None
            pids.append(rep["pid"])
        ran_argv = json.loads(Path(out_file).read_text())["argv"]
        assert ran_argv == [
            "serve", "m", "--port", str(port), "--max-model-len", "4096", "--max-num-seqs", "4",
            "--gpu-memory-utilization", "0.85",
            "--host", "127.0.0.1", "--load-format", "ipc_cache",
            "--model-loader-extra-config",
            json.dumps({"socket_dir": sock_dir, "mode": "zero_copy", "fallback": False}),
            "--worker-cls", "kvplan.KVPlanWorker",
        ]
        assert not list(Path(cache_dir, "engines").glob("*.reconfig.json")), \
            "no control file without a consumer"
    finally:
        for pid in pids:
            _stop(pid)
        daemon.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only: needs /proc/<pid> (engine.py and zygote read process state from /proc)")
def test_reconfigure_live_path_is_acknowledged_by_the_engine_core_agent():
    """Input: a fake `vllm serve` whose engine-core child runs live_reconfig's patched busy loop
    (FAKE_VLLM_LIVE), started with `--max-num-seqs 16 --max-num-batched-tokens 4096`, then:
    1. `--max-num-seqs 4` -> path "in_process_live", same pid, the agent's ack reads the
       scheduler back (target 4, max_num_running_reqs 4), and the recorded state has the new
       argv only after that ack;
    2. `--max-num-seqs 32` (above the boot value 16: the runner is sized for 16) -> not live:
       path "restart", restart_reason names the boot value."""
    work_dir = _short_dir()
    fake_vllm = _script(work_dir, "vllm", FAKE_VLLM_LIVE)
    out_file = os.path.join(work_dir, "ran.json")
    cache_dir = os.path.join(work_dir, "c")
    port = _free_port()
    env = stub_env(work_dir, FAKE_OUT=out_file, LIVELAB_SERVICES=str(HERE))
    start_proc = subprocess.run(
        [sys.executable, ENGINE, "start", "--cache-root", cache_dir, "--log",
         os.path.join(work_dir, "s1.log"), "--vllm", fake_vllm, "m", "--port", str(port),
         "--max-num-seqs", "16", "--max-num-batched-tokens", "4096"],
        capture_output=True, text=True, timeout=60, env=env)
    pids = [_one_json_line(start_proc.stdout)["pid"]]
    try:
        assert start_proc.returncode == 0, start_proc.stdout + start_proc.stderr
        state_file = Path(cache_dir, "engines", f"{pids[0]}.json")
        live_dir = json.loads(state_file.read_text())["live_dir"]
        for _ in range(100):
            if Path(live_dir, "ready.json").exists():
                break
            time.sleep(0.05)
        proc, rep = _reconfigure(cache_dir, os.path.join(work_dir, "r1.log"), pids[0], env,
                                 "--max-num-seqs", "4")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        rc = rep["reconfig"]
        assert rc["path"] == "in_process_live" and rc["restarted"] is False, rc
        assert rep["pid"] == pids[0] and not zygote._exited(pids[0])
        ack = rc["live_ack"]
        assert ack["ok"] and ack["before"]["max_num_running_reqs"] == 16
        assert ack["after"]["max_num_running_reqs"] == 4 and ack["after"]["target_max_num_seqs"] == 4
        assert json.loads(state_file.read_text())["vargv"][-2:] == ["--max-num-seqs", "4"]

        proc, rep = _reconfigure(cache_dir, os.path.join(work_dir, "r2.log"), pids[0], env,
                                 "--max-num-seqs", "32")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        rc = rep["reconfig"]
        assert rc["path"] == "restart" and "boot value 16" in rc["restart_reason"], rc
        pids.append(rep["pid"])
    finally:
        for pid in pids:   # each engine leads its session; its engine-core child is in it too
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                _stop(pid)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only: needs /proc/<pid> (engine.py and zygote read process state from /proc)")
def test_stop_engine_verifies_the_stop_and_never_signals_a_reused_pid():
    """(1) A process recorded with another start time is another process (pid reuse):
    _signal_if_same must not signal it. (2) stop_engine of an engine whose port another process
    keeps open returns stopped False with port_open True. Before: it returned stopped True
    after the SIGKILL path without checking anything."""
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        st = engine._start_time(victim.pid)
        assert st is not None
        assert engine._signal_if_same(victim.pid, st + 1, signal.SIGKILL) is False
        time.sleep(0.2)
        assert not zygote._exited(victim.pid), "a pid with another start time was signalled"
    finally:
        victim.kill()
        victim.wait()
    port = _free_port()
    other = socket.socket()
    other.bind(("127.0.0.1", port))
    other.listen(64)
    import threading

    def accept_all():
        try:
            while True:
                other.accept()[0].close()
        except OSError:
            pass
    threading.Thread(target=accept_all, daemon=True).start()
    eng = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                           start_new_session=True)
    try:
        r = engine.stop_engine(eng.pid, port=port, grace=0.5, kill_wait=0.5)
        eng.wait(timeout=5)
        assert r["stopped"] is False and r["port_open"] is True and r["alive"] == [], r
    finally:
        other.close()
        _stop(eng.pid)
    eng = subprocess.Popen([sys.executable, "-c",
                            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                            " time.sleep(60)"], start_new_session=True)
    try:
        time.sleep(0.3)
        r = engine.stop_engine(eng.pid, grace=0.5, kill_wait=3)
        assert r["stopped"] is True and r["alive"] == [], r
    finally:
        _stop(eng.pid)


def test_descendants_include_workers_reparented_out_of_the_engine_tree(tmp_path):
    """Input: a /proc tree (stat files in the kernel's format) with the engine 1000 (session
    and group leader), its child 1001, a worker 1002 whose parent died (ppid 1) but that stays
    in session 1000, and an unrelated process 1003. Action: _descendants(1000). Expect: 1001
    and 1002, not 1003. Before: only the ppid links were followed, so 1002, which still holds
    GPU memory, was never signalled. Does not prove: the signals (the Linux case below)."""
    for pid, ppid, pgrp, sid in ((1000, 1, 1000, 1000), (1001, 1000, 1000, 1000),
                                 (1002, 1, 1000, 1000), (1003, 1, 1003, 1003)):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "stat").write_text(
            f"{pid} (python3 (vllm)) S {ppid} {pgrp} {sid} " + " ".join(["0"] * 40) + "\n")
    assert sorted(engine._descendants(1000, proc_root=str(tmp_path))) == [1001, 1002]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only: needs /proc/<pid> (engine.py and zygote read process state from /proc)")
def test_stop_engine_stops_the_session_after_its_leader_exited():
    """Input: an engine leader (own session) that started a worker and exited; the worker
    stays in the leader's session with ppid 1. Action: stop_engine(leader pid). Expect: the
    worker is killed and stopped is True. Before: stop_engine returned at once with "was not
    running" and the worker kept running (holding the GPU, in a real engine)."""
    leader = subprocess.Popen(
        [sys.executable, "-c", "import subprocess, sys; print(subprocess.Popen([sys.executable,"
         " '-c', 'import time; time.sleep(60)']).pid, flush=True)"],
        stdout=subprocess.PIPE, text=True, start_new_session=True)
    worker = int(leader.stdout.readline())
    leader.wait(timeout=10)
    try:
        result = engine.stop_engine(leader.pid, grace=0.5, kill_wait=3)
        assert result["stopped"] is True, result
        assert zygote._exited(worker), "the leader's session member is still running"
    finally:
        _stop(worker)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only: needs /proc/<pid> (engine.py and zygote read process state from /proc)")
def test_reconfigure_refuses_tier2_or_bad_flags_before_stopping_running_engine():
    """Input: a running engine on port P (`--weights`), and three `reconfigure --pid PID` calls:
    (1) `--bogus-flag` (rejected by vLLM's parser);
    (2) `-tp 2` (`live_reshard_or_cold`, `subtier: tier2a_reshard`: changes TP sharding in
        WeightCacheKey);
    (3) `--kv-cache-dtype fp8` (`live_reshard_or_cold`, `subtier: tier2b_weight_format`: changes
        KV cache dtype in WeightCacheKey).
    Expect: all three exit 2 with findings (`vllm_args`, `weights_reconfig_tier2` for tier2a
    and tier2b), and the original engine PID stays alive and serving /health on port P!
    Then (4) with `--weights-accept-reconfig --kv-cache-dtype fp8`, preflight accepts the Tier-2
    change and replaces the engine."""
    from weights import socket_name
    work_dir = _short_dir()
    smi_dir = os.path.join(work_dir, "bin")
    os.makedirs(smi_dir, exist_ok=True)
    _script(smi_dir, "nvidia-smi",
            "#!{py}\nimport sys\nids = sys.argv[sys.argv.index('-i') + 1].split(',') "
            "if '-i' in sys.argv else ['0']\nprint('\\n'.join(f'GPU-fake-{{i}}' for i in ids))\n")
    sock_dir = os.path.join(work_dir, "s")
    os.makedirs(sock_dir)
    daemon = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    daemon.bind(os.path.join(sock_dir, socket_name("GPU-fake-0")))
    daemon.listen(4)
    fake_vllm = _script(work_dir, "vllm", FAKE_VLLM)
    out_file = os.path.join(work_dir, "ran.json")
    cache_dir = os.path.join(work_dir, "c")
    port = _free_port()
    env = stub_env(work_dir, FAKE_OUT=out_file, CUDA_VISIBLE_DEVICES="0",
                   PATH=smi_dir + os.pathsep + os.environ["PATH"])
    start_proc = subprocess.run(
        [sys.executable, ENGINE, "start", "--cache-root", cache_dir, "--log",
         os.path.join(work_dir, "s1.log"), "--vllm", fake_vllm, "--weights",
         "--weights-socket-dir", sock_dir, "m", "--port", str(port),
         "--max-model-len", "4096"],
        capture_output=True, text=True, timeout=60, env=env,
    )
    start_report = _one_json_line(start_proc.stdout)
    pid1 = start_report["pid"]
    pid2 = None
    try:
        assert start_proc.returncode == 0
        for patch_flags, want_check, want_subtier in (
            (["--bogus-flag"], "vllm_args", None),
            (["-tp", "2"], "weights_reconfig_tier2", "tier2a_reshard"),
            (["--kv-cache-dtype", "fp8"], "weights_reconfig_tier2", "tier2b_weight_format"),
            (["--max-num-seqs", "0"], "vllm_value", None),
        ):
            refused_proc = subprocess.run(
                [sys.executable, ENGINE, "reconfigure", "--pid", str(pid1),
                 "--cache-root", cache_dir, "--log", os.path.join(work_dir, "bad.log"),
                 *patch_flags],
                capture_output=True, text=True, timeout=60, env=env,
            )
            assert refused_proc.returncode == 2, refused_proc.stdout + refused_proc.stderr
            refused_report = _one_json_line(refused_proc.stdout)
            assert refused_report["findings"][0]["check"] == want_check, refused_report
            if want_subtier:
                assert refused_report["reconfig"]["tier"] == "live_reshard_or_cold", refused_report
                assert refused_report["reconfig"]["subtier"] == want_subtier, refused_report
            assert not zygote._exited(pid1), "running engine must stay up on preflight refusal"
            assert engine.port_in_use(port) is True
        accepted_proc = subprocess.run(
            [sys.executable, ENGINE, "reconfigure", "--pid", str(pid1),
             "--cache-root", cache_dir, "--log", os.path.join(work_dir, "s2.log"),
             "--weights-accept-reconfig", "--kv-cache-dtype", "fp8"],
            capture_output=True, text=True, timeout=60, env=env,
        )
        assert accepted_proc.returncode == 0, accepted_proc.stdout + accepted_proc.stderr
        accepted_report = _one_json_line(accepted_proc.stdout)
        pid2 = accepted_report["pid"]
        assert zygote._exited(pid1)
        assert accepted_report["reconfig"]["tier"] == "live_reshard_or_cold"
        assert accepted_report["reconfig"]["subtier"] == "tier2b_weight_format"
        assert accepted_report["vllm_args"]["args"]["kv_cache_dtype"] == "fp8"
    finally:
        _stop(pid1)
        if pid2:
            _stop(pid2)
        daemon.close()
