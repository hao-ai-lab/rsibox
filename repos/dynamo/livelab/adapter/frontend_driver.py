"""Lightweight HTTP frontend driver (compiled Rust HTTP proxy or Python HTTP server)."""
from __future__ import annotations
import contextlib, http.server, json, os, shutil, subprocess, tempfile, threading, time, urllib.request
from typing import Any, Callable
from rsibox import proc

_RUST_SRC = r"""use std::io::{copy, Read, Write}; use std::net::{TcpListener, TcpStream}; use std::{env, process, thread};
fn handle(mut c: TcpStream, be_addr: String) {
    let (mut buf, mut tmp) = (Vec::with_capacity(4096), [0u8; 4096]);
    let he = loop { match c.read(&mut tmp) { Ok(0) | Err(_) => return, Ok(n) => { buf.extend_from_slice(&tmp[..n]); if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") { break p + 4; } } } };
    let head = String::from_utf8_lossy(&buf[..he]);
    if head.lines().next().unwrap_or("").starts_with("GET /health") { let b = format!("{{\"ok\":true,\"frontend\":\"rust\",\"pid\":{}}}", process::id()); let _ = c.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len()).as_bytes()); return; }
    let clen: usize = head.lines().find_map(|l| { let mut sp = l.splitn(2, ':'); if sp.next()?.trim().eq_ignore_ascii_case("content-length") { sp.next()?.trim().parse().ok() } else { None } }).unwrap_or(0);
    while buf.len() < he + clen { match c.read(&mut tmp) { Ok(0) | Err(_) => break, Ok(n) => buf.extend_from_slice(&tmp[..n]) } }
    if let Ok(mut be) = TcpStream::connect(&be_addr) { let _ = be.write_all(&buf[..(he + clen).min(buf.len())]); let _ = copy(&mut be, &mut c); } else { let _ = c.write_all(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 12\r\nConnection: close\r\n\r\n{\"ok\":false}"); }
}
fn main() { let a: Vec<String> = env::args().collect(); for conn in TcpListener::bind(&a[1]).expect("bind").incoming().flatten() { let ba = a[2].clone(); thread::spawn(move || handle(conn, ba)); } }"""

class FrontendDriver:
    """Isolated HTTP frontend process (Rust proxy or Python HTTP) over a backend handler."""
    def __init__(self, handler: Callable[[dict[str, Any]], dict[str, Any]], *, mode: str = "auto", cache_dir: str | None = None) -> None:
        self._handler, self._mode, self._cache_dir = handler, str(mode), cache_dir
        self._be_http: Any = None
        self._be_thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._rust_bin = self._owned_temp_dir = None
        self._backend_port = self._frontend_port = self.restart_count = 0
        self.runtime = "python_http"
    def _ensure_rust_binary(self) -> str | None:
        if self._mode == "python": return None
        if self._rust_bin and os.path.isfile(self._rust_bin): return self._rust_bin
        rustc = shutil.which("rustc") or (p if os.path.isfile(p := os.path.expanduser("~/.cargo/bin/rustc")) else None)
        if not rustc:
            if self._mode == "rust": raise RuntimeError("rustc not found for frontend_mode='rust'")
            return None
        base_dir = self._cache_dir or tempfile.mkdtemp(prefix="rsidyn-fe-")
        self._owned_temp_dir = None if self._cache_dir else base_dir; os.makedirs(base_dir, exist_ok=True)
        bin_path, src_path = os.path.join(base_dir, "dynamo_frontend_rs"), os.path.join(base_dir, "dynamo_frontend_rs.rs")
        if not os.path.isfile(bin_path):
            with open(src_path, "w", encoding="utf-8") as fh: fh.write(_RUST_SRC)
            if (res := subprocess.run([rustc, "-O", src_path, "-o", bin_path], capture_output=True, text=True, timeout=60)).returncode != 0:
                if self._mode == "rust": raise RuntimeError(f"rustc build failed: {res.stderr.strip()}")
                return None
        self._rust_bin = bin_path; return bin_path
    def _stop_fe_proc(self) -> None:
        if (p := self._proc) is not None:
            self._proc = None
            with contextlib.suppress(Exception): proc.stop_session(p.pid, grace=2.0, reap=p.poll)
            with contextlib.suppress(Exception):
                if p.poll() is None: p.kill()
            with contextlib.suppress(Exception): p.wait(timeout=2.0)
    def start(self, timeout_s: float = 10.0) -> dict[str, Any]:
        """Start backend bridge and Rust/Python HTTP frontend listener."""
        try:
            if self._be_http is None:
                handler_fn = self._handler
                class _H(http.server.BaseHTTPRequestHandler):
                    def log_message(self, format: str, *args: Any) -> None: return None
                    def _send(self, p: dict[str, Any]) -> None:
                        b = json.dumps(p).encode("utf-8")
                        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
                    def do_GET(self) -> None: self._send({"ok": True, "frontend": "python", "pid": os.getpid()})
                    def do_POST(self) -> None:
                        try: self._send(handler_fn(json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))))
                        except Exception as exc: self._send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                class _S(http.server.ThreadingHTTPServer):
                    def handle_error(self, request: Any, client_address: Any) -> None: return None
                srv = _S(("127.0.0.1", 0), _H); self._backend_port, self._be_http = int(srv.server_address[1]), srv
                t = threading.Thread(target=srv.serve_forever, daemon=True); t.start(); self._be_thread = t
            rust_bin, t0 = self._ensure_rust_binary(), time.perf_counter()
            if not rust_bin:
                self.runtime, self._frontend_port = "python_http", self._backend_port
                return {"ok": True, "runtime": self.runtime, "pid": os.getpid(), "port": self._frontend_port, "start_s": round(time.perf_counter() - t0, 6)}
            self.runtime, self._frontend_port = "rust_http_proxy", proc.free_port()
            self._proc = subprocess.Popen([rust_bin, f"127.0.0.1:{self._frontend_port}", f"127.0.0.1:{self._backend_port}"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if self._proc.poll() is not None: raise RuntimeError(f"rust frontend exited (rc={self._proc.returncode})")
                with contextlib.suppress(Exception):
                    if self.health().get("ok"): return {"ok": True, "runtime": self.runtime, "pid": self._proc.pid, "port": self._frontend_port, "start_s": round(time.perf_counter() - t0, 6)}
                time.sleep(0.01)
            raise TimeoutError("rust frontend /health timed out")
        except BaseException:
            self.stop(); raise
    pid = property(lambda self: self._proc.pid if self._proc is not None else (os.getpid() if self._be_http is not None else None))
    def _http_call(self, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{self._frontend_port}{path}", data=data, method="GET" if data is None else "POST")
        if data is not None: req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp: return json.loads(resp.read().decode("utf-8"))  # noqa: S310
    def health(self, timeout: float = 5.0) -> dict[str, Any]: return self._http_call("/health", None, timeout=timeout)
    def complete(self, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]: return self._http_call("/v1/completions", payload, timeout=timeout)
    def restart_frontend(self, timeout_s: float = 10.0) -> dict[str, Any]:
        """Kill and respawn only the frontend HTTP process while preserving backend workers."""
        old_pid, t0 = self.pid, time.perf_counter()
        self._stop_fe_proc(); started = self.start(timeout_s=timeout_s); self.restart_count += 1
        return {**started, "old_pid": old_pid, "new_pid": self.pid, "restart_s": round(time.perf_counter() - t0, 6), "restart_count": self.restart_count}
    def stop(self) -> None:
        """Stop frontend proxy, internal HTTP bridge, and temporary cache directory."""
        self._stop_fe_proc()
        if (b := self._be_http) is not None:
            self._be_http = None
            with contextlib.suppress(Exception): b.shutdown(); b.server_close()
        if self._be_thread is not None: self._be_thread.join(timeout=2.0); self._be_thread = None
        if self._owned_temp_dir: shutil.rmtree(self._owned_temp_dir, ignore_errors=True); self._owned_temp_dir = None
