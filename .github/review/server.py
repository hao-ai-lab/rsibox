#!/usr/bin/env python3
"""The review service behind https://services.fastrsi.ai/v1/review.

Runs the Antigravity CLI on a PR-description prompt and returns its raw reply. It does not
check or apply anything; review.py does that in CI. Asynchronous, because a review takes
minutes and Cloudflare closes proxied requests after 100 s:

  POST /v1/review       {"kind": "pr-description", "repo", "pr", "prompt", "model"} -> 202 {"id"}
  GET  /v1/review/<id>  -> {"status": "queued|running|done|error", "reply", "error", "seconds"}
  GET  /v1/health       -> {"ok": true, "queued": n, "running": n}   (no auth)

Safety:
- Every call except /v1/health needs `Authorization: Bearer <token>`.
- agy runs headless in a fresh empty directory with no tool allow-rules, so tools that need
  the "command" permission are auto-denied (tested: a prompt asking it to `touch` a file did not
  create it). Slash commands are disabled.
- Only allow-listed models, prompts up to 200 KB, at most 2 reviews at a time.

Standard library only.

  FASTRSI_REVIEW_TOKEN_FILE=~/.config/fastrsi-review/token AGY=/path/to/agy python3 server.py --port 8792
"""
import argparse, json, os, re, secrets, shutil, subprocess, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = {"gemini-4-argon-high", "gemini-4-argon-medium", "gemini-3.8-flash-high"}
MAX_PROMPT = 200_000
KEEP_SECONDS = 24 * 3600
JOBS, LOCK = {}, threading.Lock()


def log(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), msg, flush=True)


def run_review(job_id, prompt, model, agy):
    with LOCK:
        JOBS[job_id].update(status="running", started=time.time())
    workdir = tempfile.mkdtemp(prefix="review-")
    try:
        out = subprocess.run(
            [agy, "-p", prompt, "--model", model, "--sandbox", "--disable-slash-commands",
             "--print-timeout", "540s"],
            cwd=workdir, capture_output=True, text=True, timeout=600)
        if out.returncode != 0 or "<<<TITLE" not in out.stdout:
            raise RuntimeError(f"agy exit {out.returncode}: {(out.stderr or out.stdout)[-400:]}")
        update = dict(status="done", reply=out.stdout)
    except Exception as e:  # report the failure to the caller instead of hanging the job
        update = dict(status="error", error=str(e)[:500])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    with LOCK:
        job = JOBS[job_id]
        job.update(update, seconds=round(time.time() - job["started"], 1))
        log(f"job {job_id} {job['status']} in {job['seconds']} s ({job['repo']}#{job['pr']})")


class Handler(BaseHTTPRequestHandler):
    server_version = "fastrsi-review/1"

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
                                   "running": states.count("running")})
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
        if size > MAX_PROMPT + 10_000:
            return self.send(413, {"error": "request too large"})
        try:
            req = json.loads(self.rfile.read(size))
            assert req.get("kind") == "pr-description", "kind must be pr-description"
            assert req.get("model") in MODELS, f"model must be one of {sorted(MODELS)}"
            assert isinstance(req.get("prompt"), str) and 0 < len(req["prompt"]) <= MAX_PROMPT, "bad prompt"
        except (ValueError, AssertionError) as e:
            return self.send(400, {"error": str(e)})
        job_id = secrets.token_hex(8)
        with LOCK:
            now = time.time()
            for k in [k for k, j in JOBS.items() if now - j["created"] > KEEP_SECONDS]:
                del JOBS[k]
            JOBS[job_id] = dict(status="queued", created=now, repo=str(req.get("repo"))[:100],
                                pr=str(req.get("pr"))[:20])
        self.server.pool.submit(run_review, job_id, req["prompt"], req["model"], self.server.agy)
        log(f"job {job_id} queued ({req.get('repo')}#{req.get('pr')}, {len(req['prompt'])} chars)")
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
    server.agy = os.environ.get("AGY", "agy")
    server.pool = ThreadPoolExecutor(max_workers=2)
    log(f"listening on {args.host}:{args.port}, agy={server.agy}")
    server.serve_forever()


if __name__ == "__main__":
    main()
