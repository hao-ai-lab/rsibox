"""Live lab dashboard server: start a real `vllm serve`, stream its startup as events, and let a
browser hot-patch it.

  browser  <-- /events (server-sent events) --  this server  --  vllm serve (dev mode + lab ext)

Actions (all fixed; the browser cannot send code or commands):
  POST /engine/start       start vllm serve with the cache root and lab extension
  POST /engine/stop        stop it
  POST /patch/install      {"word": "Paris"}: install a sampler patch that bans that word
  POST /patch/uninstall    restore the base code
  POST /request            {"prompt": "..."}: one greedy completion, compared with the base output
  GET  /events?since=N     event stream: engine.*, phase.end, log, cache.diff, patch.*, request.*, witness
  GET  /                   the dashboard page;  /site/ serves --site if given

The demo patch edits `apply_all_penalties` in vllm/v1/sample/ops/penalties.py, a plain function
that runs outside CUDA graphs on every sampling step that has a penalty, so a request with a tiny
repetition penalty runs it. Binds 127.0.0.1 only: dev mode lets any local caller run code in the
engine. Standard library only.

Usage:
  python3 server.py --model <path or HF id> --cache-root <dir> --log-dir <dir> \
      [--gpus 0] [--port 8766] [--engine-port 8010] [--site <dir>] [-- extra vllm serve args]
"""
import argparse, difflib, importlib.util, json, os, re, signal, subprocess, sys, threading, time
import urllib.request
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
LIVELAB = HERE.parent
sys.path.insert(0, str(LIVELAB / "startup"))
import caches, phases  # noqa: E402

PATCH_MODULE = "vllm.v1.sample.ops.penalties"
PATCH_FUNC = "apply_all_penalties"
REQ_PARAMS = {"max_tokens": 32, "temperature": 0.0, "repetition_penalty": 1.0001}


class Lab:
    def __init__(self, a, extra):
        self.a, self.extra = a, extra
        self.events, self.cv = [], threading.Condition()
        self.proc, self.t0, self.log_path, self.before = None, None, None, None
        self.ready, self.patched, self.base_outputs = False, None, {}
        self.phase_seconds = {}

    # ---------- events ----------
    def emit(self, type_, **kw):
        with self.cv:
            ev = {"seq": len(self.events), "t": round(time.time(), 3), "type": type_, **kw}
            self.events.append(ev)
            self.cv.notify_all()
        return ev

    # ---------- engine ----------
    def engine_url(self):
        return f"http://127.0.0.1:{self.a.engine_port}"

    def start(self):
        if self.proc and self.proc.poll() is None:
            raise RuntimeError("engine is already running")
        a = self.a
        env = os.environ.copy()
        env.update(caches.env(a.cache_root))
        env.update({"VLLM_SERVER_DEV_MODE": "1", "CUDA_VISIBLE_DEVICES": a.gpus,
                    "PYTHONPATH": os.pathsep.join([str(LIVELAB / "startup"), str(LIVELAB)] +
                                                  ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))})
        Path(a.log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = str(Path(a.log_dir) / f"engine-{time.strftime('%m%d-%H%M%S')}.log")
        cmd = [a.vllm, "serve", a.model, "--served-model-name", a.served_name,
               "--host", "127.0.0.1", "--port", str(a.engine_port),
               "--max-model-len", str(a.max_model_len),
               "--gpu-memory-utilization", str(a.gpu_mem),
               "--worker-extension-cls", "fastrsi_lab_ext.LabExtension", *self.extra]
        self.before = caches.snapshot(a.cache_root)
        self.ready, self.patched, self.phase_seconds = False, None, {}
        self.t0 = time.time()
        # cwd: the log folder. Started from inside a package folder, Python would import that
        # folder's modules first (e.g. vllm/tokenizers shadows the `tokenizers` package).
        self.proc = subprocess.Popen(cmd, stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT,
                                     env=env, cwd=a.log_dir, start_new_session=True)
        cold = sum(v["files"] for v in self.before.values()) == 0 if self.before else True
        self.emit("engine.start", model=a.served_name, gpus=a.gpus, pid=self.proc.pid,
                  cache_root_files=sum(v["files"] for v in self.before.values()), cold=cold)
        threading.Thread(target=self._tail, daemon=True).start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
            self.emit("engine.stop", pid=self.proc.pid)
        self.ready, self.patched = False, None

    def _tail(self):
        """Follow the engine log: phase ends (from startup/phases.py marks), log lines, ready."""
        seen, last_mark = set(), self.t0
        with open(self.log_path, errors="replace") as f:
            while True:
                line = f.readline()
                if not line:
                    if self.proc.poll() is not None:
                        self.emit("engine.exit", code=self.proc.returncode, log=self.log_path)
                        return
                    if self.ready:
                        time.sleep(0.5)
                        continue
                    time.sleep(0.1)
                    continue
                line = line.rstrip()
                if not line:
                    continue
                if not self.ready:
                    self.emit("log", line=line[-240:])
                for name, rx in phases.MARKS:
                    if (name not in seen or name in phases.LAST) and rx.search(line):
                        now = time.time()
                        secs = round(now - last_mark, 1)
                        # A repeated LAST mark (second graph capture) extends that phase.
                        self.phase_seconds[name] = round(self.phase_seconds.get(name, 0) + secs, 1)
                        seen.add(name)
                        last_mark = now
                        self.emit("phase.end", name=name, seconds=secs,
                                  total=self.phase_seconds[name], since_start=round(now - self.t0, 1))
                        if name == "ready":
                            self._on_ready()
                if "Traceback (most recent call last)" in line or "Engine core initialization failed" in line:
                    self.emit("engine.error", line=line[-240:])

    def _on_ready(self):
        self.ready = True
        diff = caches.diff(self.before, caches.snapshot(self.a.cache_root))
        new_files = sum(v.get("new_files", 0) for v in diff.values()) if isinstance(diff, dict) else None
        self.emit("cache.diff", diff=diff, new_files=new_files)
        self.emit("engine.ready", seconds=round(time.time() - self.t0, 1), phases=self.phase_seconds)

    # ---------- engine calls ----------
    def post(self, path, body, timeout=300):
        req = urllib.request.Request(self.engine_url() + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        return json.loads(raw) if raw else None

    def rpc(self, method, *args):
        return self.post("/collective_rpc", {"method": method, "args": list(args)})["results"]

    def need_ready(self):
        if not self.ready:
            raise RuntimeError("start the engine first and wait for ready")

    # ---------- patch ----------
    @staticmethod
    def base_source():
        spec = importlib.util.find_spec("vllm")  # top level only: does not import vllm
        path = Path(spec.origin).parent / "v1" / "sample" / "ops" / "penalties.py"
        return path.read_text(), str(path)

    def patched_source(self, word):
        src, _ = self.base_source()
        ids = set()
        for variant in {word, " " + word, word.lower(), " " + word.lower(), word.capitalize(),
                        " " + word.capitalize()}:
            toks = self.post("/tokenize", {"model": self.a.served_name, "prompt": variant,
                                          "add_special_tokens": False})["tokens"]
            ids.add(toks[0])  # ban the first token of each spelling
        ids = sorted(ids)
        start = src.index(f"def {PATCH_FUNC}(")
        at = src.index("    return apply_penalties(", start)
        line = f'    logits[:, {ids}] = float("-inf")  # fastrsi demo: ban {json.dumps(word)}\n'
        new = src[:at] + line + src[at:]
        diff = "".join(difflib.unified_diff(src.splitlines(True), new.splitlines(True),
                                            "penalties.py", "penalties.py (patched)", n=2))
        return new, ids, diff

    def install(self, word):
        self.need_ready()
        if self.patched:
            raise RuntimeError("a patch is installed; undo it first")
        if not re.fullmatch(r"[A-Za-z][A-Za-z\- ']{0,39}", word or ""):
            raise ValueError("use one word of letters, up to 40 characters")
        src, ids, diff = self.patched_source(word.strip())
        t = time.time()
        res = self.rpc("fastrsi_install", PATCH_MODULE, PATCH_FUNC, src)
        ms = round((time.time() - t) * 1000, 1)
        self.patched = {"word": word, "ids": ids}
        return self.emit("patch.install", module=PATCH_MODULE, func=PATCH_FUNC, word=word,
                         token_ids=ids, ms=ms, ranks=res, diff=diff)

    def uninstall(self):
        self.need_ready()
        t = time.time()
        res = self.rpc("fastrsi_uninstall", PATCH_MODULE, PATCH_FUNC)
        ms = round((time.time() - t) * 1000, 1)
        self.patched = None
        return self.emit("patch.uninstall", ms=ms, ranks=res)

    def request(self, prompt):
        self.need_ready()
        prompt = (prompt or "")[:500]
        t = time.time()
        out = self.post("/v1/completions", {"model": self.a.served_name, "prompt": prompt,
                                            **REQ_PARAMS})
        secs = round(time.time() - t, 3)
        text = out["choices"][0]["text"]
        arm = "patched" if self.patched else "base"
        if arm == "base" and prompt not in self.base_outputs:
            self.base_outputs[prompt] = text
        base = self.base_outputs.get(prompt)
        hits = self.rpc("fastrsi_witness", PATCH_MODULE)
        self.emit("witness", hits=[h.get("hits", {}) for h in hits])
        return self.emit("request.done", arm=arm, prompt=prompt, text=text, seconds=secs,
                         same_as_base=(None if base is None else text == base),
                         banned=self.patched["word"] if self.patched else None)


def make_handler(lab, site):
    page = (HERE / "index.html")

    class H(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                body = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/events":
                since = int(parse_qs(u.query).get("since", ["0"])[0])
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                i = since
                try:
                    while True:
                        with lab.cv:
                            lab.cv.wait_for(lambda: len(lab.events) > i, timeout=15)
                            batch = lab.events[i:]
                        if not batch:
                            self.wfile.write(b": ping\n\n")
                        for ev in batch:
                            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                        i += len(batch)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            elif u.path.startswith("/site") and site:
                self.path = u.path[len("/site"):] or "/"
                self.directory = site
                return super().do_GET()
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            routes = {"/engine/start": lambda: lab.start(), "/engine/stop": lambda: lab.stop(),
                      "/patch/install": lambda: lab.install(body.get("word", "")),
                      "/patch/uninstall": lambda: lab.uninstall(),
                      "/request": lambda: lab.request(body.get("prompt", ""))}
            fn = routes.get(urlparse(self.path).path)
            if not fn:
                return self._json(404, {"error": "not found"})
            try:
                self._json(200, {"ok": True, "result": fn()})
            except Exception as e:  # noqa: BLE001  shown to the user in the page
                lab.emit("error", where=self.path, message=f"{type(e).__name__}: {e}"[:300])
                self._json(400, {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})

    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--served-name", default="demo")
    p.add_argument("--cache-root", required=True); p.add_argument("--log-dir", required=True)
    p.add_argument("--gpus", default="0"); p.add_argument("--port", type=int, default=8766)
    p.add_argument("--engine-port", type=int, default=8010)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem", type=float, default=0.4)
    p.add_argument("--vllm", default="vllm"); p.add_argument("--site", default="")
    a, extra = p.parse_known_args()
    extra = [x for x in extra if x != "--"]
    lab = Lab(a, extra)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(lab, a.site))
    srv.daemon_threads = True
    print(f"dashboard on http://127.0.0.1:{a.port}/", flush=True)
    try:
        srv.serve_forever()
    finally:
        lab.stop()


if __name__ == "__main__":
    main()
