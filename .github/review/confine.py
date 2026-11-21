#!/usr/bin/env python3
"""Run a command that can only read and write the paths it is given (Linux Landlock).

The review service runs reviewer CLIs (agy, codex) on untrusted text: a PR diff or description
can tell the reviewer to read a file and quote it. agy's file tools cannot be switched off
from its command line (tested: a headless `agy -p --sandbox` read a file outside its working
directory), so the service confines the whole process instead. Landlock is enforced by the
kernel, is inherited by every child and thread, and cannot be lifted by the confined process.

  confine.py --read DIR... --write DIR... -- CMD ARGS...

`--read` paths are readable and executable, `--write` paths are fully usable; everything else
on the filesystem is denied (network is not restricted). Fails closed: if Landlock is
unavailable it refuses to run the command. Standard library only.
"""
import argparse, ctypes, errno, os, sys

SYS_CREATE, SYS_ADD_RULE, SYS_RESTRICT = 444, 445, 446   # x86_64 and aarch64 alike
PR_SET_NO_NEW_PRIVS, RULE_PATH_BENEATH = 38, 1
EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
DIR_WRITE = sum(1 << i for i in range(4, 13))      # remove/make dir, file, sock, fifo, symlink...
REFER, TRUNCATE = 1 << 13, 1 << 14                  # ABI 2, ABI 3
FILE_ONLY = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE
READ = EXECUTE | READ_FILE | READ_DIR

CLONE_NEWNS, CLONE_NEWUSER = 0x00020000, 0x10000000
MS_NOSUID, MS_NODEV, MS_REC, MS_PRIVATE = 2, 4, 16384, 1 << 18

libc = ctypes.CDLL(None, use_errno=True)
if hasattr(libc, "unshare"):
    libc.unshare.argtypes = [ctypes.c_int]
    libc.unshare.restype = ctypes.c_int
if hasattr(libc, "mount"):
    libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                           ctypes.c_ulong, ctypes.c_void_p]
    libc.mount.restype = ctypes.c_int


class PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _check(rc, what):
    if rc < 0:
        raise OSError(ctypes.get_errno(), f"landlock {what}: {os.strerror(ctypes.get_errno())}")
    return rc


def handled_access():
    abi = libc.syscall(SYS_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    if abi < 1:
        raise OSError("Landlock is not available on this kernel; refusing to run unconfined")
    handled = READ | WRITE_FILE | DIR_WRITE
    if abi >= 2:
        handled |= REFER
    if abi >= 3:
        handled |= TRUNCATE
    return handled


def setup_private_shm(target="/dev/shm"):
    """Give this process its own `/dev/shm` mount namespace (a fresh private tmpfs) before
    Landlock runs, so concurrent jobs never share `/dev/shm`. Returns the resolved mount path,
    or None when unprivileged user/mount namespaces are disallowed by the host kernel/profile
    (caller must never fall back to granting the host `/dev/shm`)."""
    real_target = os.path.realpath(target)
    if not os.path.isdir(real_target) or not hasattr(libc, "unshare") or not hasattr(libc, "mount"):
        return None
    uid, gid = os.getuid(), os.getgid()
    if libc.unshare(CLONE_NEWUSER | CLONE_NEWNS) < 0:
        err = ctypes.get_errno()
        if err in (errno.EPERM, errno.EACCES, errno.ENOSYS, errno.EINVAL,
                   errno.ENOSPC, getattr(errno, "EUSERS", errno.ENOSPC)):
            return None
        raise OSError(err, f"unshare(CLONE_NEWUSER|CLONE_NEWNS): {os.strerror(err)}")
    # Map only our own uid/gid after denying setgroups (required since Linux 3.19).
    with open("/proc/self/setgroups", "w") as f:
        f.write("deny\n")
    with open("/proc/self/uid_map", "w") as f:
        f.write(f"{uid} {uid} 1\n")
    with open("/proc/self/gid_map", "w") as f:
        f.write(f"{gid} {gid} 1\n")
    if libc.mount(None, b"/", None, MS_REC | MS_PRIVATE, None) < 0:
        err = ctypes.get_errno()
        if err not in (errno.EINVAL, errno.EPERM):
            raise OSError(err, f"mount make-rprivate /: {os.strerror(err)}")
    if libc.mount(b"tmpfs", real_target.encode(), b"tmpfs",
                  MS_NOSUID | MS_NODEV, b"size=128m,mode=0700") < 0:
        err = ctypes.get_errno()
        if err in (errno.EPERM, errno.EACCES, errno.ENOSYS):
            return None
        raise OSError(err, f"mount tmpfs {real_target}: {os.strerror(err)}")
    return real_target


def confine(read, write):
    """Restrict this process (and everything it execs) to `read` and `write` paths."""
    handled = handled_access()
    attr = ctypes.c_uint64(handled)
    fd = _check(libc.syscall(SYS_CREATE, ctypes.byref(attr), ctypes.c_size_t(8),
                             ctypes.c_uint32(0)), "create_ruleset")
    try:
        for paths, access in ((read, READ), (write, handled)):
            for p in paths:
                if not os.path.exists(p):
                    continue
                pfd = os.open(p, os.O_PATH | os.O_CLOEXEC)
                try:
                    allowed = access & handled
                    if not os.path.isdir(p):
                        allowed &= FILE_ONLY
                    rule = PathBeneath(allowed, pfd)
                    _check(libc.syscall(SYS_ADD_RULE, ctypes.c_int(fd), ctypes.c_int(RULE_PATH_BENEATH),
                                        ctypes.byref(rule), ctypes.c_uint32(0)), f"add_rule {p}")
                finally:
                    os.close(pfd)
        _check(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "no_new_privs")
        _check(libc.syscall(SYS_RESTRICT, ctypes.c_int(fd), ctypes.c_uint32(0)), "restrict_self")
    finally:
        os.close(fd)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read", nargs="*", default=[])
    ap.add_argument("--write", nargs="*", default=[])
    ap.add_argument("--private-shm", action="store_true",
                    help="mount an isolated per-process /dev/shm tmpfs (denies /dev/shm if unavailable)")
    ap.add_argument("--require-private-shm", action="store_true",
                    help="fail closed if an isolated /dev/shm namespace cannot be mounted")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    write_paths = [os.path.expanduser(p) for p in a.write]
    if a.private_shm or a.require_private_shm:
        shm_path = setup_private_shm()
        if shm_path is None:
            if a.require_private_shm:
                raise OSError("private /dev/shm namespace unavailable; refusing to run")
        elif shm_path not in write_paths:
            write_paths.append(shm_path)
    confine([os.path.expanduser(p) for p in a.read], write_paths)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
