"""Engine-agnostic stdlib runtime kit built on `core/livelab/kernel/` and `forksafe`.

Provides shared, zero-dependency building blocks for rsibox live labs:
- `bootstrap_livelab`: locate `core/livelab` on `sys.path` without mutating `os.environ`.
- `ipc`: atomic JSON, `AF_UNIX` JSON-line RPC, file channels, and NFS-safe mailboxes.
- `proc`: cross-platform process-tree discovery, termination, reaping, and TCP port helpers.
- `hooks`: stack-safe `MetaPathFinder` import hooks and source overlays (`OverlayFinder`).
- `harness`: `edit_once`, module source lookup, `PackageOverlay`, fresh subprocess runner,
  and canonical `fail_apply` / `fail_commit` / `tamper` fault injection.
- `diffs`: pure-stdlib + `git apply` unified-diff resolver (`apply`, `paths`, `module_of`).
- `lab`: `BaseLab` (`LabUnderTest`), `RecaptureCoordinator`, and `RoleEndpoint`.
- `pool`: `SandboxPool` multi-agent FIFO scheduler with clean-slate isolation and failover.
- `cli`: unified engine-agnostic CLI (`rsibox list | conformance | survey | ctl | doctor`).
"""
import os
import sys

from . import diffs, doctor, op_guard, pool, promote, warm
from .diffs import apply as apply_diff, module_of as module_of_path, paths as diff_paths
from .doctor import check_rlimit_nofile, run_preflight
from .op_guard import op_redefine_guard
from .promote import run_promotion, verify_promotion_receipt
from .harness import (
    PackageOverlay,
    SourceStack,
    append_jsonl,
    build_package_overlay,
    check_failure,
    edit_once,
    inject_participant_fault,
    install_refusal,
    module_file,
    module_relpath,
    plain_finding,
    read_module,
    resolve_patch_files,
    run_fresh_subprocess,
)
from .hooks import (
    AfterImportHook,
    OverlayFinder,
    dependents,
    import_graph,
    install_import_hook,
    install_overlay,
    purge_modules,
    unbind_module,
)
from .ipc import (
    MAX_SOCK_PATH,
    FileChannelServer,
    MailboxServer,
    UnixJsonServer,
    atomic_write_json,
    file_channel_call,
    mailbox_call,
    read_json,
    short_sock_path,
    sock_path_error,
    unix_json_call,
    unix_json_stream,
)
from .lab import BaseLab, RecaptureCoordinator, RoleEndpoint
from .pool import (
    AgentQuarantinedError, PoolClosedError, PoolError, PoolSlot, PoolTimeoutError,
    SandboxPool, SlotLease, verify_trace,
)
from .proc import (
    descendants,
    direct_children,
    free_port,
    kill_tree,
    pg_alive,
    pid_alive,
    port_in_use,
    ppid_map,
    reap_all,
    stat_ppid,
    stat_state,
    stop_session,
    stop_tree,
)


def bootstrap_livelab(
    anchor_file: str | None = None,
    *,
    env_var: str = "RSIBOX_LIVELAB",
    fallback_env: str | None = None,
) -> str:
    """Resolve `core/livelab` and prepend it to `sys.path` without mutating `os.environ`."""
    override = os.environ.get(env_var) or (os.environ.get(fallback_env) if fallback_env else None)
    if override:
        livelab = os.path.abspath(override)
    elif anchor_file:
        cur = os.path.dirname(os.path.realpath(anchor_file))
        livelab = None
        for _ in range(8):
            cand = os.path.join(cur, "core", "livelab")
            if os.path.isdir(cand):
                livelab = os.path.abspath(cand)
                break
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        if livelab is None:
            livelab = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    else:
        livelab = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if livelab not in sys.path:
        sys.path.insert(0, livelab)
    return livelab


__all__ = [
    "AfterImportHook",
    "AgentQuarantinedError",
    "BaseLab",
    "FileChannelServer",
    "MAX_SOCK_PATH",
    "MailboxServer",
    "OverlayFinder",
    "PackageOverlay",
    "PoolClosedError",
    "PoolError",
    "PoolSlot",
    "PoolTimeoutError",
    "RecaptureCoordinator",
    "RoleEndpoint",
    "SandboxPool",
    "SlotLease",
    "SourceStack",
    "UnixJsonServer",
    "append_jsonl",
    "apply_diff",
    "atomic_write_json",
    "bootstrap_livelab",
    "build_package_overlay",
    "check_failure",
    "check_rlimit_nofile",
    "dependents",
    "descendants",
    "diff_paths",
    "diffs",
    "direct_children",
    "doctor",
    "edit_once",
    "file_channel_call",
    "free_port",
    "import_graph",
    "inject_participant_fault",
    "install_import_hook",
    "install_overlay",
    "install_refusal",
    "kill_tree",
    "mailbox_call",
    "module_file",
    "module_of_path",
    "module_relpath",
    "op_guard",
    "op_redefine_guard",
    "pg_alive",
    "pid_alive",
    "plain_finding",
    "pool",
    "port_in_use",
    "ppid_map",
    "promote",
    "purge_modules",
    "read_json",
    "read_module",
    "reap_all",
    "resolve_patch_files",
    "run_fresh_subprocess",
    "run_preflight",
    "run_promotion",
    "short_sock_path",
    "sock_path_error",
    "stat_ppid",
    "stat_state",
    "stop_session",
    "stop_tree",
    "unbind_module",
    "unix_json_call",
    "unix_json_stream",
    "verify_promotion_receipt",
    "verify_trace",
    "warm",
]
