#!/usr/bin/env python3
"""The review service behind <review-service-url>.

Runs a reviewer CLI on a prompt and returns its raw reply. It does not check, post or apply
anything (review.py and code_review.py do that in CI), so it never needs GitHub access.
Asynchronous, because a review takes minutes and Cloudflare closes proxied requests after 100 s:

  POST /v1/review       {"kind", "engine", "repo", "pr", "prompt", "model"} -> 202 {"id"}
  GET  /v1/review/<id>  -> {"status": "queued|running|done|error", "reply", "error", "seconds"}
  GET  /v1/health       -> {"ok": true, "queued": n, "running": n, "engines": [...]} (no auth)

Kinds: "pr-description" (engine agy) and "code-review" (engine agy or codex). `engine`
defaults to agy, so description-review clients from before code review keep working.

Safety (the prompt carries untrusted PR text):
- Every call except /v1/health needs `Authorization: Bearer <token>`.
- Each reviewer runs in a fresh empty directory with a minimal environment and a fresh HOME
  that holds only copies of its credential files (never its real state directory with other
  conversations, settings or plugins), under confine.py: Landlock lets it read only system
  paths and its own binary, and write only the job directory. Tested: a prompt
  telling agy to read a file elsewhere gets "permission denied", even with all agy tools
  approved. agy also runs with `--sandbox` and slash commands off; codex with its read-only
  sandbox and its shell, exec, web, browser, app and plugin tools disabled.
- A reply or error that contains material from a secret file (the service token, the
  reviewers' credentials, SSH keys) is withheld: the job errors without returning it.
- Only allow-listed kinds, engines and models; prompts up to 120 KB; 2 reviews at a time.

Standard library only.

  FASTRSI_REVIEW_TOKEN_FILE=~/.config/fastrsi-review/token AGY=/path/to/agy CODEX=/path/to/codex \
      python3 server.py --port 8792
"""
import argparse, json, os, re, secrets, shutil, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
KINDS = {"pr-description": {"agy"}, "code-review": {"agy", "codex"}}
MODELS = {"agy": {"gemini-4-argon-high", "gemini-4-argon-medium", "gemini-3.8-flash-high"},
          "codex": {"default"}}
EXPECT = {"pr-description": "<<<TITLE", "code-review": '"findings"'}
MAX_PROMPT = 120_000          # one argv string is capped at 128 KiB on Linux (agy takes -p)
KEEP_SECONDS = 24 * 3600
SYSTEM_READ = ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/run", "/proc/self", "/sys"]
CODEX_DISABLE = ["shell_tool", "unified_exec", "apps", "plugins", "browser_use",
                 "browser_use_external", "computer_use", "in_app_browser", "web_search_request",
                 "web_search_cached", "standalone_web_search", "image_generation", "enable_mcp_apps"]
JOBS, LOCK = {}, threading.Lock()


def log(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), msg, flush=True)


def validate(req):
    """Return an error string for a bad request, or None. Fills in the default engine."""
    if not isinstance(req, dict):
        return "request must be a JSON object"
    kind = req.get("kind")
    if kind not in KINDS:
        return f"kind must be one of {sorted(KINDS)}"
    req.setdefault("engine", "agy")
    if req["engine"] not in KINDS[kind]:
        return f"engine for {kind} must be one of {sorted(KINDS[kind])}"
    if req.get("model") not in MODELS[req["engine"]]:
        return f"model must be one of {sorted(MODELS[req['engine']])}"
    p = req.get("prompt")
    if not isinstance(p, str) or not 0 < len(p.encode()) <= MAX_PROMPT:
        return f"prompt must be a non-empty string of at most {MAX_PROMPT} bytes"
    return None


def secret_strings(paths):
    """Long tokens from secret files: any of them in a reply means the reviewer read a secret."""
    out = set()
    for p in paths:
        try:
            text = Path(os.path.expanduser(p)).read_text(errors="ignore")
        except OSError:
            continue
        out.update(t for t in re.split(r"[^A-Za-z0-9_\-+/=.]", text) if len(t) >= 20)
    return out


def leaks(reply, secrets_):
    return any(s in reply for s in secrets_)


# Per engine: the state directory a reviewer needs and the only files copied into it. Each job
# gets a fresh home holding just these copies, so a reviewer never sees the real state
# directory (other conversations, settings, plugins); a refreshed credential is copied back.
STATE = {"agy": (".gemini/antigravity-cli", ["antigravity-oauth-token", "installation_id"]),
         "codex": (".codex", ["auth.json"])}


def real_state(engine):
    if engine == "codex" and os.environ.get("CODEX_HOME"):
        return Path(os.path.expanduser(os.environ["CODEX_HOME"]))
    return Path.home() / STATE[engine][0]


IDENTITY = {"agy": (("auth_method",), ("id_token_email",)),
            "codex": (("auth_mode",), ("tokens", "account_id"))}
COPY_BACK_LOCK = threading.Lock()


def job_home(engine, workdir):
    """A fresh HOME inside the job directory with copies of the engine's credential files.
    Returns (env, [(copy, original, snapshot bytes)])."""
    home = Path(workdir, "home")
    state = home / STATE[engine][0]
    state.mkdir(parents=True)
    pairs = []
    for name in STATE[engine][1]:
        src = real_state(engine) / name
        if src.exists():
            snap = src.read_bytes()
            (state / name).write_bytes(snap)
            os.chmod(state / name, 0o600)
            pairs.append((state / name, src, snap))
    if engine == "agy":
        (state / "settings.json").write_text("{}")
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "TMPDIR": workdir}
    if engine == "codex":
        env["CODEX_HOME"] = str(state)
    return env, pairs


def _shape(x):
    return {k: _shape(v) for k, v in x.items()} if isinstance(x, dict) else None


def _get(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def refresh_ok(engine, snap, new):
    """True only for a token refresh: JSON with the same keys and the same account. Anything
    else (garbage, another account's credentials planted by a prompt-injected reviewer) is
    refused."""
    try:
        a, b = json.loads(snap), json.loads(new)
    except ValueError:
        return False
    if not isinstance(a, dict) or _shape(a) != _shape(b):
        return False
    return all(_get(a, path) == _get(b, path) for path in IDENTITY[engine])


def copy_back(engine, pairs):
    """Keep a credential the reviewer refreshed (token rotation), but only a genuine refresh of
    the same account, and only if no other job changed the real file meanwhile."""
    with COPY_BACK_LOCK:
        for copy, src, snap in pairs:
            if not copy.exists():
                continue
            new = copy.read_bytes()
            if new == snap or src.read_bytes() != snap:
                continue                                  # unchanged, or a newer one is in place
            if not refresh_ok(engine, snap, new):
                log(f"refused to copy back {src.name}: not a refresh of the same account")
                continue
            fd, tmp = tempfile.mkstemp(dir=src.parent, prefix=f".{src.name}.")
            with os.fdopen(fd, "wb") as f:
                f.write(new)
            os.chmod(tmp, 0o600)
            os.replace(tmp, src)


def command(engine, prompt, model, workdir, bins):
    """argv (confined) and stdin for one reviewer run."""
    exe = os.path.realpath(bins[engine])
    if engine == "agy":
        argv = [exe, "-p", prompt, "--model", model, "--sandbox", "--disable-slash-commands",
                "--print-timeout", "540s"]
        stdin = None
    else:
        argv = [exe, "exec", "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
                "--ignore-rules", "--ignore-user-config", "--color", "never",
                "-o", os.path.join(workdir, "last-message.txt")]
        for f in CODEX_DISABLE:
            argv += ["--disable", f]
        if model != "default":
            argv += ["-m", model]
        argv.append("-")           # the prompt comes on stdin (no argv size limit)
        stdin = prompt
    confine = [sys.executable, "-I", str(HERE / "confine.py"),
               "--read", *SYSTEM_READ, os.path.dirname(exe),
               "--write", workdir, "/dev/null", "/dev/shm", "--"]
    return confine + argv, stdin


def run_review(job_id, req, bins, secret_files):
    with LOCK:
        JOBS[job_id].update(status="running", started=time.time())
    workdir = tempfile.mkdtemp(prefix="review-")
    pairs, secret = [], set()
    try:
        env, pairs = job_home(req["engine"], workdir)
        argv, stdin = command(req["engine"], req["prompt"], req["model"], workdir, bins)
        out = subprocess.run(argv, input=stdin, cwd=workdir, env=env, capture_output=True,
                             text=True, timeout=600)
        secret = secret_strings(list(secret_files) + [str(c) for c, _, _ in pairs])
        reply = out.stdout
        if req["engine"] == "codex":
            last = Path(workdir, "last-message.txt")
            reply = last.read_text() if last.exists() else ""
        if leaks(reply + out.stdout + out.stderr, secret):
            raise RuntimeError("reply withheld: it contains material from a secret file")
        if out.returncode != 0 or EXPECT[req["kind"]] not in reply:
            raise RuntimeError(f"{req['engine']} exit {out.returncode}: "
                               f"{(out.stderr or out.stdout)[-400:]}")
        update = dict(status="done", reply=reply)
    except Exception as e:  # report the failure to the caller instead of hanging the job
        msg = str(e)[:500]
        if leaks(msg, secret or secret_strings(secret_files)):
            msg = "error withheld: it contains material from a secret file"
        update = dict(status="error", error=msg)
    finally:
        try:
            copy_back(req["engine"], pairs)
        except OSError as e:
            log(f"job {job_id}: could not copy back a refreshed credential: {e}")
        shutil.rmtree(workdir, ignore_errors=True)
    with LOCK:
        job = JOBS[job_id]
        job.update(update, seconds=round(time.time() - job["started"], 1))
        log(f"job {job_id} {req['kind']}/{req['engine']} {job['status']} in {job['seconds']} s "
            f"({job['repo']}#{job['pr']})")


class Handler(BaseHTTPRequestHandler):
    server_version = "fastrsi-review/2"

    def send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        got = self.headers.get("Authorization", "")
        return secrets.compare_digest(got.encode(), f"Bearer {self.server.token}".encode())

    def do_GET(self):
        if self.path == "/v1/health":
            with LOCK:
                states = [j["status"] for j in JOBS.values()]
            return self.send(200, {"ok": True, "queued": states.count("queued"),
                                   "running": states.count("running"),
                                   "engines": sorted(self.server.bins)})
        if not self.authorized():
            return self.send(401, {"error": "unauthorized"})
        m = re.fullmatch(r"/v1/review/([0-9a-f]{16})", self.path)
        with LOCK:
            job = JOBS.get(m.group(1)) if m else None
            view = {k: job.get(k) for k in ("status", "reply", "error", "seconds")} if job else None
        return self.send(200, view) if view else self.send(404, {"error": "no such job"})

    def do_POST(self):
        if self.path != "/v1/review":
            return self.send(404, {"error": "not found"})
        if not self.authorized():
            return self.send(401, {"error": "unauthorized"})
        size = int(self.headers.get("Content-Length") or 0)
        if size > 2 * MAX_PROMPT:
            return self.send(413, {"error": "request too large"})
        try:
            req = json.loads(self.rfile.read(size))
        except ValueError as e:
            return self.send(400, {"error": f"bad JSON: {e}"})
        err = validate(req)
        if not err and req["engine"] not in self.server.bins:
            err = f"engine {req['engine']} is not installed on this service"
        if err:
            return self.send(400, {"error": err})
        job_id = secrets.token_hex(8)
        with LOCK:
            now = time.time()
            for k in [k for k, j in JOBS.items() if now - j["created"] > KEEP_SECONDS]:
                del JOBS[k]
            JOBS[job_id] = dict(status="queued", created=now, repo=str(req.get("repo"))[:100],
                                pr=str(req.get("pr"))[:20])
        self.server.pool.submit(run_review, job_id, req, self.server.bins, self.server.secret_files)
        log(f"job {job_id} {req['kind']}/{req['engine']} queued "
            f"({req.get('repo')}#{req.get('pr')}, {len(req['prompt'])} chars)")
        return self.send(202, {"id": job_id})

    def log_message(self, fmt, *args):  # one line per request, without query bodies
        log(f"{self.address_string()} {fmt % args}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8792)
    args = ap.parse_args()
    token_file = os.path.expanduser(os.environ.get("FASTRSI_REVIEW_TOKEN_FILE",
                                                   "~/.config/fastrsi-review/token"))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.token = open(token_file).read().strip()
    server.bins = {e: shutil.which(os.environ.get(e.upper(), e)) for e in ("agy", "codex")}
    server.bins = {e: b for e, b in server.bins.items() if b}
    server.secret_files = [token_file, "~/.gemini/antigravity-cli/antigravity-oauth-token",
                           str(real_state("codex") / "auth.json"),
                           *[str(p) for p in Path("~/.ssh").expanduser().glob("id_*")]]
    server.secret_files += [p for p in os.environ.get("FASTRSI_REVIEW_SECRET_FILES", "").split(":") if p]
    server.pool = ThreadPoolExecutor(max_workers=2)
    log(f"listening on {args.host}:{args.port}, engines={server.bins}")
    server.serve_forever()


if __name__ == "__main__":
    main()
