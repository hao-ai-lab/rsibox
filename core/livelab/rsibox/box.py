"""Box: one live lab behind one call gate.

A box owns four things and nothing else: its state (`booting`, `ready`, `failed`, `closed`),
the order of calls (one verb at a time), the event list, and the receipt file's path. The lab
it wraps still owns `base` / `edited` / `poisoned` (its `Session`) and writes the receipts.

`call` never raises: a lab call that raises is answered as `{"ok": False, "raised": True}`.
"""
import hashlib
import importlib
import json
import os
import threading
import time
import traceback

VERBS = ("install", "check", "reset", "uninstall", "status")
READS = ("info", "events", "receipts", "source")


def load_lab(spec):
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls)


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Box:
    def __init__(self, lab_factory, spec="", state_dir=None, busy_wait_s=5.0, clock=time.monotonic):
        """`lab_factory(receipts=path)` builds the lab; it runs on a boot thread (`start()`)."""
        self.spec, self.state_dir, self.busy_wait_s, self.clock = spec, state_dir, busy_wait_s, clock
        self.receipts = os.path.join(state_dir, "receipts.jsonl") if state_dir else None
        self._factory, self.lab = lab_factory, None
        self.state, self.error, self.boot_s = "booting", None, None
        self._gate = threading.Lock()            # held while a verb runs
        self._meta = threading.Lock()            # guards events and in_flight
        self.events, self.in_flight = [], None
        self.started_at = time.time()
        self._t0 = clock()

    # -- lifecycle -----------------------------------------------------------------------
    def start(self):
        threading.Thread(target=self.boot, name="rsibox-boot", daemon=True).start()
        return self

    def boot(self):
        t0 = self.clock()
        try:
            lab = self._factory(receipts=self.receipts)
        except BaseException as exc:  # noqa: BLE001 - a lab that cannot start must not kill the daemon
            self.error = f"{type(exc).__name__}: {exc}"
            self.boot_s = round(self.clock() - t0, 3)
            self.state = "failed"
            self._event("boot", {"ok": False, "error": self.error, "trace": traceback.format_exc()[-2000:]},
                        self.boot_s)
            return
        self.lab, self.boot_s = lab, round(self.clock() - t0, 3)
        self.state = "ready"
        self._event("boot", {"ok": True}, self.boot_s)

    def close(self):
        if self.state == "closed":
            return {"ok": True, "state": "closed"}
        with self._gate:
            lab, self.lab, self.state = self.lab, None, "closed"
            t0 = self.clock()
            try:
                if lab is not None:
                    lab.close()
                reply = {"ok": True, "state": "closed"}
            except BaseException as exc:  # noqa: BLE001
                reply = {"ok": False, "state": "closed", "error": f"{type(exc).__name__}: {exc}"}
            self._event("close", reply, round(self.clock() - t0, 3))
        return reply

    # -- calls ---------------------------------------------------------------------------
    def call(self, verb, body=None):
        """Run one interface verb. Returns `(http_status, reply)`."""
        body = body or {}
        if verb not in VERBS:
            return 404, {"ok": False, "error": f"unknown verb {verb!r}", "verbs": list(VERBS)}
        if self.state != "ready":
            return 409, {"ok": False, "error": f"box is {self.state}", "state": self.state,
                         "boot_error": self.error}
        if not self._gate.acquire(timeout=self.busy_wait_s):
            return 409, {"ok": False, "error": "box is busy", "state": "busy", "in_flight": self.in_flight}
        try:
            if self.state != "ready":            # closed while this call waited
                return 409, {"ok": False, "error": f"box is {self.state}", "state": self.state}
            t0 = self.clock()
            with self._meta:
                self.in_flight = {"verb": verb, "since": time.time()}
            try:
                reply = self._dispatch(verb, body)
                if not isinstance(reply, dict):
                    reply = {"ok": False, "error": f"lab returned {type(reply).__name__}, not a dict"}
            except BaseException as exc:  # noqa: BLE001 - the interface says a lab never raises
                reply = {"ok": False, "raised": True, "error": f"{type(exc).__name__}: {exc}",
                         "trace": traceback.format_exc()[-2000:]}
            wall = round(self.clock() - t0, 6)
            seq = self._event(verb, reply, wall)
            return 200, {**reply, "box": {"seq": seq, "wall_s": wall}}
        finally:
            with self._meta:
                self.in_flight = None
            self._gate.release()

    def _dispatch(self, verb, body):
        lab = self.lab
        if verb == "install":
            patch = body.get("patch", body)
            if not isinstance(patch, dict) or not (patch.get("files") or patch.get("diff")):
                return {"ok": False, "outcome": "refused", "codes": ["patch_invalid"],
                        "error": 'a patch is {"files": {module: full text}} or {"diff": unified diff}'}
            stale = self._stale_modules(patch.get("base"))
            if stale:
                return {"ok": False, "outcome": "refused", "codes": ["base_mismatch"], "stale": stale,
                        "error": "the live text of these modules is not the text the patch was made from"}
            return lab.install({k: v for k, v in patch.items() if k != "base"})
        if verb == "check":
            return lab.check(body.get("workload"))
        if verb == "reset":
            return lab.reset(body.get("kind", "state"))
        if verb == "uninstall":
            return lab.uninstall(body.get("id"))
        return lab.status()

    def _stale_modules(self, base):
        """Modules whose live text hash differs from `base` ({module: hash the caller edited from})."""
        stale = {}
        for module, want in (base or {}).items():
            _, got = self.source(module)
            if got.get("hash") != want:
                stale[module] = {"expected": want, "live": got.get("hash")}
        return stale

    # -- reads (never wait for the gate) -------------------------------------------------
    def info(self):
        with self._meta:
            n, flight = len(self.events), self.in_flight
        return {"ok": True, "state": self.state, "lab": self.spec, "boot_s": self.boot_s,
                "boot_error": self.error, "in_flight": flight, "events": n,
                "started_at": self.started_at, "uptime_s": round(self.clock() - self._t0, 3),
                "busy_wait_s": self.busy_wait_s, "pid": os.getpid()}

    def events_since(self, since=0):
        with self._meta:
            return {"ok": True, "events": [e for e in self.events if e["seq"] > since]}

    def read_receipts(self):
        records = []
        try:
            with open(self.receipts, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        records.append(json.loads(line))
        except (OSError, TypeError, ValueError) as exc:
            if not isinstance(exc, FileNotFoundError):
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "receipts": records}
        return {"ok": True, "receipts": records}

    def source(self, module):
        """The live text of `module`: base plus every committed edit."""
        lab = self.lab
        if self.state != "ready" or lab is None:
            return 409, {"ok": False, "error": f"box is {self.state}", "state": self.state}
        text = None
        try:
            # A lab's own `source(module)` wins. The rest are where today's labs keep the text:
            # vLLM's `_live_source`, the `live_text` stack of `BaseLab` and the toy lab, then base.
            stack = getattr(lab, "live_text", None)
            for name in ("source", "_live_source"):
                if text is None and callable(getattr(lab, name, None)):
                    text = getattr(lab, name)(module)
            if text is None and stack and module in stack[-1]:
                text = stack[-1][module]
            if text is None and callable(getattr(lab, "_base_source", None)):
                text = lab._base_source(module)
            if text is None:
                text = (getattr(lab, "base_text", None) or {}).get(module)
        except BaseException as exc:  # noqa: BLE001
            return 200, {"ok": False, "module": module, "error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(text, str):
            avail = sorted(stack[-1]) if stack else sorted(getattr(lab, "base_text", None) or ())
            reply = {"ok": False, "module": module, "error": "the lab does not serve this module's text"}
            if avail:
                reply["available"] = avail
            return 200, reply
        return 200, {"ok": True, "module": module, "text": text, "hash": text_hash(text)}

    # -- events --------------------------------------------------------------------------
    def _event(self, verb, reply, wall_s):
        ev = {"t": time.time(), "verb": verb, "ok": bool(reply.get("ok")), "wall_s": wall_s}
        for key in ("outcome", "id", "codes", "cost_s", "proven", "state", "error", "raised", "undone"):
            if reply.get(key) is not None:
                ev[key] = reply[key]
        with self._meta:
            ev["seq"] = len(self.events) + 1
            self.events.append(ev)
        if self.state_dir:
            try:
                with open(os.path.join(self.state_dir, "events.jsonl"), "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(ev, default=str) + "\n")
            except OSError:
                pass
        return ev["seq"]
