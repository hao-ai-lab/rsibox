"""Zero-eager-import MPI worker hook for TensorRT-LLM `TP >= 2` ranks.

Installed via a generated `sitecustomize.py` when `RSIBOX_TRTLLM_MPI_SOCK_DIR`
is set. Never imports `tensorrt_llm` or `torch` at interpreter startup so
`mpi4py.futures.MPIPoolExecutor` completes `MPI_Init` cleanly before the hook
attaches a per-rank RSIBox `LocalParticipant` Unix JSON server inside
`PyExecutor.__init__`.
"""
from __future__ import annotations

import importlib.abc
import json
import os
import queue
import sys
import threading
from typing import Any

ENV_MPI_SOCK_DIR = "RSIBOX_TRTLLM_MPI_SOCK_DIR"
ENV_MPI_MODULES = "RSIBOX_TRTLLM_MPI_MODULES"
ENV_MPI_ENABLE_BLOCK_REUSE = "RSIBOX_TRTLLM_MPI_ENABLE_BLOCK_REUSE"
ENV_MPI_SEED = "RSIBOX_TRTLLM_MPI_SEED"
_TARGET_MODULE = "tensorrt_llm._torch.pyexecutor.py_executor"

SITECUSTOMIZE_SOURCE = """\
import os
if os.environ.get("RSIBOX_TRTLLM_MPI_SOCK_DIR"):
    try:
        from adapter import mpi_hook
        mpi_hook.install()
    except Exception:
        pass
"""

_SERVERS: list[Any] = []


def _start_rank_server(py_executor: Any) -> None:
    sock_dir = os.environ.get(ENV_MPI_SOCK_DIR)
    if not sock_dir:
        return
    from tensorrt_llm._utils import mpi_rank

    from adapter import agent
    from kernel import findings
    from rsibox import ipc

    rank_index = int(mpi_rank())
    sock_path = os.path.join(sock_dir, f"rank-{rank_index}.sock")
    raw_modules = os.environ.get(ENV_MPI_MODULES, "[]")
    try:
        tracked_modules = tuple(json.loads(raw_modules) or ())
    except Exception:
        tracked_modules = ()
    enable_block_reuse = os.environ.get(ENV_MPI_ENABLE_BLOCK_REUSE, "0") == "1"
    seed = int(os.environ.get(ENV_MPI_SEED, "0"))

    handle_request, should_stop_fn = agent.make_worker_handler(
        mode="real",
        engine=py_executor,
        rank_index=rank_index,
        seed=seed,
        enable_block_reuse=enable_block_reuse,
        tracked_modules=tracked_modules,
    )

    def _serve() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(rank_index % torch.cuda.device_count())
        except Exception:
            pass
        with ipc.UnixJsonServer(sock_path, handler=None) as server:
            while not should_stop_fn():
                try:
                    request, send_reply = server.next(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    reply = handle_request(request)
                except BaseException as error:
                    reply = findings.not_ok(f"{request.get('method')}: {type(error).__name__}: {error}")
                send_reply(reply)

    thread = threading.Thread(target=_serve, name=f"rsibox-trtllm-rank{rank_index}", daemon=True)
    thread.start()
    _SERVERS.append(thread)


def _patch_py_executor_module(module: Any) -> None:
    cls = getattr(module, "PyExecutor", None)
    if cls is None or getattr(cls, "_rsibox_mpi_hooked", False):
        return
    cls._rsibox_mpi_hooked = True
    orig_init = cls.__init__

    def _wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        _start_rank_server(self)

    cls.__init__ = _wrapped_init


class _AfterImportLoader(importlib.abc.Loader):
    def __init__(self, orig_loader: Any) -> None:
        self._orig = orig_loader

    def create_module(self, spec: Any) -> Any:
        if hasattr(self._orig, "create_module"):
            return self._orig.create_module(spec)
        return None

    def exec_module(self, module: Any) -> None:
        self._orig.exec_module(module)
        _patch_py_executor_module(module)


class _AfterImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._busy = False

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if self._busy or fullname != _TARGET_MODULE:
            return None
        self._busy = True
        try:
            for finder in list(sys.meta_path):
                if finder is self:
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _AfterImportLoader(spec.loader)
                    return spec
        finally:
            self._busy = False
        return None


def install() -> None:
    """Install the post-import hook if not already present."""
    if not os.environ.get(ENV_MPI_SOCK_DIR):
        return
    existing = sys.modules.get(_TARGET_MODULE)
    if existing is not None:
        _patch_py_executor_module(existing)
        return
    if not any(isinstance(f, _AfterImportFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AfterImportFinder())
