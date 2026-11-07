"""Golden cases for reading a `vllm serve` argv with vLLM's own parser (vllm_args.py), through the
launcher (engine.py) and through the zygote. They need a real vLLM: the python running pytest if
it imports vLLM, else the one in FASTRSI_VLLM_PYTHON. Without either (CPU CI) they skip. They run
on CPU; no case touches a GPU.

Run: FASTRSI_VLLM_PYTHON=<vLLM venv python> python3 -m pytest \
         repos/vllm/livelab/services/test_vllm_args.py -v
"""
import importlib.util, json, os, subprocess, sys, tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine, zygote  # noqa: E402
from test_engine import FAKE_VLLM, _free_port, _script, _short_dir, _stop  # noqa: E402

VLLM_PY = sys.executable if importlib.util.find_spec("vllm") else \
    os.environ.get("FASTRSI_VLLM_PYTHON")
pytestmark = pytest.mark.skipif(not VLLM_PY, reason="needs vLLM: run with FASTRSI_VLLM_PYTHON "
                                "set to a vLLM venv's python (CPU CI has no vLLM)")
ENGINE = str(HERE / "engine.py")

# One fresh process parses every argv below, then reports whether CUDA got initialized.
CASES = {
    "positional": ["Qwen/Qwen3-8B", "--port", "8123"],
    "model_flag": ["--model", "Qwen/Qwen3-8B"],
    "tp_short": ["M", "-tp", "2"],
    "tp_long": ["M", "--tensor-parallel-size", "2"],
    "tp_underscore": ["M", "--tensor_parallel_size=2"],
    "default_spelled": ["M", "--load-format", "auto"],
    "eager_worker": ["M", "--enforce-eager", "--worker-cls", "a.B"],
    "no_model": ["--port", "8123"],
    "bad_flag": ["M", "--no-such-flag"],
}
PARSE_ALL = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import vllm_args, zygote, torch
parser, _ = vllm_args.build()
cases = json.loads(sys.argv[2])
out = {k: vllm_args.parse(v, parser) for k, v in cases.items()}
ours = set(json.loads(sys.argv[3]))
theirs = {o for a in parser._actions for o in a.option_strings}
print(json.dumps({"cases": out, "overlap": sorted(ours & theirs),
                  "cuda_initialized": torch.cuda.is_initialized(),
                  "driver_initialized": zygote.driver_initialized()}))
'''


@pytest.fixture(scope="module")
def parsed():
    ours = [o for a in engine.launcher_parser()._actions for o in a.option_strings
            if o not in ("-h", "--help")]
    r = subprocess.run([VLLM_PY, "-c", PARSE_ALL, str(HERE), json.dumps(CASES),
                        json.dumps(ours)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_positional_model_resolves_to_the_given_model(parsed):
    """Input: `Qwen/Qwen3-8B --port 8123` and `--model Qwen/Qwen3-8B`. Expect: model
    Qwen/Qwen3-8B, explicit, in both. vLLM puts a positional model in model_tag and leaves
    `.model` at its default Qwen/Qwen3-0.6B; `vllm serve` copies model_tag over it, and so must
    we (reading `.model` naively serves the wrong model). Also: a missing model is not explicit."""
    for k in ("positional", "model_flag"):
        c = parsed["cases"][k]
        assert c["ok"] and c["args"]["model"] == "Qwen/Qwen3-8B", c
        assert "model" in c["explicit"]
    assert parsed["cases"]["positional"]["args"]["port"] == 8123
    no = parsed["cases"]["no_model"]
    assert no["args"]["model"] == "Qwen/Qwen3-0.6B" and "model" not in no["explicit"]


def test_tensor_parallel_size_in_every_spelling(parsed):
    """Input: `-tp 2`, `--tensor-parallel-size 2`, `--tensor_parallel_size=2`. Expect: 2 in all
    three (only vLLM's parser knows `-tp` and the underscore form)."""
    for k in ("tp_short", "tp_long", "tp_underscore"):
        assert parsed["cases"][k]["args"]["tensor_parallel_size"] == 2, parsed["cases"][k]
        assert "tensor_parallel_size" in parsed["cases"][k]["explicit"]


def test_explicit_means_given_even_at_the_default(parsed):
    """Input: `--load-format auto` (vLLM's default), and `--enforce-eager --worker-cls a.B`.
    Expect: load_format explicit in the first though it equals the default; enforce_eager and
    worker_cls explicit with their values in the second, and neither explicit in the first."""
    d = parsed["cases"]["default_spelled"]
    assert d["args"]["load_format"] == "auto" and "load_format" in d["explicit"]
    assert "worker_cls" not in d["explicit"] and "enforce_eager" not in d["explicit"]
    e = parsed["cases"]["eager_worker"]
    assert e["args"]["enforce_eager"] is True and e["args"]["worker_cls"] == "a.B"
    assert {"enforce_eager", "worker_cls"} <= set(e["explicit"])


def test_a_rejected_flag_is_vllms_message(parsed):
    """Input: `M --no-such-flag`. Expect: ok false, vLLM's own error line naming the flag."""
    b = parsed["cases"]["bad_flag"]
    assert b["ok"] is False and "unrecognized arguments: --no-such-flag" in b["error"], b


def test_parsing_does_not_initialize_cuda_and_flags_do_not_overlap(parsed):
    """In the fresh process that imported vLLM's parser and parsed every case: CUDA (torch) and
    the CUDA driver (cuInit) are not initialized, so a launcher or zygote that parses can still
    fork processes that use the GPU. And no launcher flag is also a `vllm serve` flag."""
    assert parsed["cuda_initialized"] is False
    assert parsed["driver_initialized"] is False
    assert parsed["overlap"] == []


def test_zygote_parse_equals_direct_parse():
    """Input: a real zygote (vLLM imported, engine.py zygote) and the CASES argvs. Expect: the
    zygote's parse reply has the same ok/args/explicit/error as a direct parse in a fresh
    process, and takes well under a second (direct: ~5 s of imports). Not: a fork of a server."""
    d = _short_dir()
    sock, cache = os.path.join(d, "z.sock"), os.path.join(d, "c")
    r = subprocess.run([sys.executable, ENGINE, "zygote", "--python", VLLM_PY, "--zygote", sock,
                        "--cache-root", cache, "--log", os.path.join(d, "z.log")],
                       capture_output=True, text=True, timeout=600)
    rep = json.loads(r.stdout)
    try:
        assert rep["ready"] is True, r.stdout + r.stderr + Path(d, "z.log").read_text()[-3000:]
        direct = subprocess.run([VLLM_PY, "-c", PARSE_ALL, str(HERE), json.dumps(CASES), "[]"],
                                capture_output=True, text=True, timeout=300)
        direct = json.loads(direct.stdout.strip().splitlines()[-1])["cases"]
        for k, argv in CASES.items():
            z = zygote.parse(sock, argv)
            keep = ("ok", "args", "explicit", "error")
            assert {x: z.get(x) for x in keep} == {x: direct[k].get(x) for x in keep}, k
            assert z["zygote_seconds"] < 2, z
    finally:
        _stop(rep["pid"])


def test_launcher_reads_the_argv_with_vllms_parser():
    """Input: `engine.py start` under the vLLM python, flags mixed: `--cache-root C Qwen/Qwen3-8B
    --log L -tp 2 --vllm FAKE --port P`. Expect: the report's vllm_args say model Qwen/Qwen3-8B
    (not vLLM's default), tp 2, port P, via "direct"; the fake server got exactly
    `serve Qwen/Qwen3-8B -tp 2 --port P --host 127.0.0.1` and answered /health on P."""
    with tempfile.TemporaryDirectory() as d:
        fake = _script(d, "vllm", FAKE_VLLM)
        out, port = os.path.join(d, "ran.json"), _free_port()
        r = subprocess.run([VLLM_PY, ENGINE, "start", "--cache-root", os.path.join(d, "c"),
                            "Qwen/Qwen3-8B", "--log", os.path.join(d, "s.log"), "-tp", "2",
                            "--vllm", fake, "--port", str(port), "--timeout", "60"],
                           capture_output=True, text=True, timeout=300,
                           env={**os.environ, "FAKE_OUT": out})
        rep = json.loads(r.stdout.strip().splitlines()[-1])
        if rep.get("pid"):
            _stop(rep["pid"])
        assert r.returncode == 0, r.stdout + r.stderr[-3000:]
        v = rep["vllm_args"]
        assert (v["via"], v["args"]["model"], v["args"]["tensor_parallel_size"],
                v["args"]["port"]) == ("direct", "Qwen/Qwen3-8B", 2, port)
        assert json.loads(Path(out).read_text())["argv"] == [
            "serve", "Qwen/Qwen3-8B", "-tp", "2", "--port", str(port), "--host", "127.0.0.1"]
