#!/usr/bin/env python3
"""Run a command that can only read and write the paths it is given (Linux Landlock).

The review service runs reviewer CLIs (agy, codex) on untrusted text: a PR diff or description
can tell the reviewer to read a file and quote it. agy's file tools cannot be switched off
from its command line (tested: a headless `agy -p --sandbox` read a file outside its working
directory), so the service confines the whole process instead. Landlock is enforced by the
kernel, is inherited by every child and thread, and cannot be lifted by the confined process.

  confine.py --read PATH... --write PATH... -- CMD ARGS...

`--read` paths are readable and executable, `--write` paths are fully usable; everything else
on the filesystem is denied (network is not restricted). Fails closed: if Landlock is
unavailable it refuses to run the command. Standard library only.
"""
import argparse, ctypes, os, sys

SYS_CREATE, SYS_ADD_RULE, SYS_RESTRICT = 444, 445, 446   # x86_64 and aarch64 alike
PR_SET_NO_NEW_PRIVS, RULE_PATH_BENEATH = 38, 1
EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
DIR_WRITE = sum(1 << i for i in range(4, 13))      # remove/make dir, file, sock, fifo, symlink...
REFER, TRUNCATE = 1 << 13, 1 << 14                  # ABI 2, ABI 3
FILE_ONLY = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE
READ = EXECUTE | READ_FILE | READ_DIR

libc = ctypes.CDLL(None, use_errno=True)


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
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    confine([os.path.expanduser(p) for p in a.read], [os.path.expanduser(p) for p in a.write])
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
