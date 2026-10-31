"""The session: one state machine, the same for every engine.

base --install--> edited --undo--> base
any doubt -----> poisoned --restart--> base
"""

import json
import os
import socket
import time
import uuid

from fastrsi.edits import Refused

BASE, EDITED, POISONED = "base", "edited", "poisoned"
CALL_TIMEOUT = 30.0


class AgentGone(Exception):
    pass


def call(path, op, timeout=CALL_TIMEOUT, **fields):
    """One request to one process's agent."""
    try:
        with socket.socket(socket.AF_UNIX) as sock:
            sock.settimeout(timeout)
            sock.connect(path)
            sock.sendall(json.dumps({"op": op, **fields}).encode() + b"\n")
            data = b""
            while not data.endswith(b"\n"):
                chunk = sock.recv(1 << 20)
                if not chunk:
                    raise AgentGone(f"{op}: connection closed")
                data += chunk
    except OSError as error:
        raise AgentGone(f"{op}: {error}") from error
    return json.loads(data)


class Session:
    def __init__(self, engine, directory, max_installs=200, max_age_s=7200.0):
        self.engine = engine
        self.directory = directory
        self.max_installs = max_installs
        self.max_age_s = max_age_s
        self.receipts = os.path.join(directory, "receipts")
        os.makedirs(self.receipts, exist_ok=True)
        self.path = os.path.join(directory, "session.json")
        self.data = {"state": BASE, "reason": None, "installs": 0, "started": time.time(),
                     "edit": None, "functions": {}, "targets": [], "sequence": 0}
        if os.path.exists(self.path):
            with open(self.path) as handle:
                self.data = json.load(handle)

    @property
    def state(self):
        return self.data["state"]

    def save(self):
        temporary = self.path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(self.data, handle, indent=1)
        os.replace(temporary, self.path)

    def receipt(self, op, **fields):
        self.data["sequence"] += 1
        record = {"op": op, "time": time.time(), "state": self.state, **fields}
        name = f"{self.data['sequence']:05d}-{op}.json"
        with open(os.path.join(self.receipts, name), "w") as handle:
            json.dump(record, handle, indent=1)
        self.save()
        return record

    def poison(self, reason):
        self.data.update(state=POISONED, reason=reason)
        self.save()

    def sockets(self):
        """Every agent that answers. Dead sockets are removed."""
        found = []
        directory = self.engine.agent_dir
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if name.endswith(".sock"):
                try:
                    call(path, "ping", timeout=5.0)
                    found.append(path)
                except AgentGone:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
        return found

    def _targets(self, paths):
        return [s for s in self.sockets() if call(s, "present", paths=paths).get("present")]

    def _prepare(self, targets, txn, files, fault, witness):
        functions = None
        for index, path in enumerate(targets):
            injected = fault if fault == "prepare" and index == len(targets) - 1 else None
            reply = call(path, "prepare", txn=txn, files=files, fault=injected, witness=witness)
            if not reply["ok"]:
                raise RuntimeError(reply["error"])
            if functions is None:
                functions = reply["functions"]
            elif functions != reply["functions"]:
                raise RuntimeError("processes disagree on code identity")
        return functions

    def install(self, edit, fault=None, witness="first"):
        """All processes, or none. fault is for the conformance suite only."""
        began = time.perf_counter()
        if self.state != BASE:
            return self.receipt("install", outcome="refused", code="state",
                                reason=f"session is {self.state}: undo or restart first")
        try:
            route = self.engine.classify(edit)
        except Refused as refused:
            return self.receipt("install", outcome="refused", code=refused.code,
                                reason=refused.reason, edit=edit.digest)
        files = {f.path: {"base": f.base, "new": f.new, "functions": route[f.path]}
                 for f in edit.files if f.path in route}
        txn = uuid.uuid4().hex
        targets, committed = [], []
        try:
            if not self.sockets():
                return self.receipt("install", outcome="failed", reason="no agent is running",
                                    edit=edit.digest)
            targets = self._targets(list(files))
            if not targets:
                return self.receipt("install", outcome="failed", reason="no process has this file",
                                    edit=edit.digest)
            try:
                functions = self._prepare(targets, txn, files, fault, witness)
            except RuntimeError as error:
                for path in targets:
                    call(path, "abort", txn=txn)
                return self.receipt("install", outcome="failed", reason=str(error),
                                    edit=edit.digest)
            for index, path in enumerate(targets):
                last = index == len(targets) - 1
                reply = call(path, "commit", txn=txn, fault=fault if last else None)
                if not reply["ok"]:
                    raise RuntimeError(reply["error"])
                committed.append(path)
            self.data.update(state=EDITED, edit=edit.digest, functions=functions, targets=targets)
            self.data["installs"] += 1
            self.save()
            self._verify("new")
        except AgentGone as error:
            self.poison(f"process lost during install: {error}")
            return self.receipt("install", outcome="poisoned", reason=self.data["reason"])
        except RuntimeError as error:
            for path in committed:
                try:
                    call(path, "undo")
                except AgentGone:
                    self.poison("rollback failed")
            if self.state == POISONED:
                return self.receipt("install", outcome="poisoned", reason=self.data["reason"])
            self.data.update(functions=functions, targets=targets)
            try:
                self._verify("base")
            except AgentGone:
                return self.receipt("install", outcome="poisoned", reason=self.data["reason"])
            self.data.update(state=BASE, edit=None, functions={}, targets=[])
            return self.receipt("install", outcome="failed", reason=str(error), rolled_back=True,
                                edit=edit.digest)
        self._budgets()
        return self.receipt("install", outcome="installed", edit=edit.digest,
                            processes=len(targets), functions=list(functions),
                            seconds=time.perf_counter() - began)

    def _verify(self, side):
        """Prove each target runs the expected code. A mismatch poisons the session."""
        pairs = [name.split(":", 1) for name in self.data["functions"]]
        want = {name: ids[side] for name, ids in self.data["functions"].items()}
        for path in self.data["targets"]:
            reply = call(path, "identity", functions=pairs)
            if not reply["ok"] or reply["functions"] != want:
                self.poison(f"identity mismatch after {side}: {path}")
                raise AgentGone(self.data["reason"])

    def undo(self):
        began = time.perf_counter()
        if self.state != EDITED:
            return self.receipt("undo", outcome="refused", code="state",
                                reason=f"session is {self.state}")
        try:
            for path in self.data["targets"]:
                reply = call(path, "undo")
                if not reply["ok"]:
                    raise AgentGone(reply["error"])
            self._verify("base")
        except AgentGone as error:
            self.poison(f"undo failed: {error}")
            return self.receipt("undo", outcome="poisoned", reason=self.data["reason"])
        self.data.update(state=BASE, edit=None, functions={}, targets=[])
        return self.receipt("undo", outcome="undone", seconds=time.perf_counter() - began)

    def witness(self):
        """Hits per function and process."""
        result = {}
        try:
            for path in self.data["targets"]:
                reply = call(path, "witness")
                for name, hits in reply.get("hits", {}).items():
                    result.setdefault(name, []).append(hits)
        except AgentGone as error:
            self.poison(f"process lost: {error}")
        return result

    def proof(self):
        """Did every edited function actually run? Installed is not the same as proven."""
        hits = self.witness()
        names = list(self.data["functions"])
        return {"proven": bool(names) and all(any(hits.get(n, ())) for n in names),
                "not_run": [n for n in names if not any(hits.get(n, ()))]}

    def exercise(self, timeout=120.0, **fields):
        """Run the engine's workload. A hang or a dead engine poisons the session."""
        try:
            return self.engine.exercise(timeout=timeout, **fields)
        except TimeoutError:
            self.poison(f"no result within {timeout} s")
        except (OSError, RuntimeError) as error:
            self.poison(f"engine failed: {error}")
        return None

    def health(self):
        """Check every target still answers, and that no target appeared or vanished."""
        if self.state == EDITED:
            try:
                live = self._targets(list({n.split(":", 1)[0] for n in self.data["functions"]}))
                if live != self.data["targets"]:
                    self.poison("process set changed since install")
                else:
                    self._verify("new")
            except AgentGone as error:
                if self.state != POISONED:
                    self.poison(f"process lost: {error}")
        if self.state != POISONED and not self.engine.alive():
            self.poison("engine process exited")
        self._budgets()
        return self.state

    def _budgets(self):
        if self.state == POISONED:
            return
        if self.data["installs"] >= self.max_installs:
            self.poison(f"install budget reached: {self.max_installs}")
        elif time.time() - self.data["started"] >= self.max_age_s:
            self.poison(f"age budget reached: {self.max_age_s} s")

    def restart(self):
        began = time.perf_counter()
        self.engine.stop()
        self.engine.start()
        self.data.update(state=BASE, reason=None, installs=0, started=time.time(), edit=None,
                         functions={}, targets=[])
        return self.receipt("restart", outcome="restarted", seconds=time.perf_counter() - began)
