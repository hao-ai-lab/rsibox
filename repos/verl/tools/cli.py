"""Gate client, `verl` launcher, and CLI (`python3 -m tools.cli`: `train`, `request`, `stop`, `conformance`, `rescore`, `replay-adv`)."""
import argparse, importlib.util, json, os, sys, sysconfig
from pathlib import Path

VERL_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIVELAB_DIR = os.path.join(VERL_REPO, "livelab")
for p in (VERL_REPO, LIVELAB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapter.client import GateError, _job_died, _next_request, request, summary  # noqa: E402, F401


def launch_command(overrides, gate_dir=None, trace_path=None, roots=None):
    from adapter import LIVELAB
    env = dict(os.environ)
    py_paths = [LIVELAB_DIR, VERL_REPO, LIVELAB] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    argv = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    if trace_path:
        env["VERL_STARTUP_TRACE"] = str(Path(trace_path).resolve())
    if gate_dir:
        if roots is None:
            spec = importlib.util.find_spec("verl")
            vroot = str(Path(spec.origin).parent) if spec and spec.origin else None
            rdir = next((str(Path(it.partition("=")[2]).resolve().parent) for it in overrides
                         if it.lstrip("+").startswith("reward.custom_reward_function.path=")), None)
            roots = [r for r in (vroot, rdir) if r]
        env["RSIBOX_RL_GATE"], env["RSIBOX_RL_ROOTS"] = str(Path(gate_dir).resolve()), os.pathsep.join(roots)
        argv += ["trainer.v1.trainer_mode=rsibox", "+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=adapter.hook.setup"]
    site = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    libs = [str(site / lib / "lib") for lib in ("cudnn", "nccl") if (site / lib / "lib").is_dir()]
    if libs:
        env["LD_LIBRARY_PATH"] = ":".join(libs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    return argv, env


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m tools.cli")
    sub = ap.add_subparsers(dest="command", required=True)
    req = sub.add_parser("request")
    req.add_argument("--gate", required=True)
    req.add_argument("--verb", default="install", choices=("install", "check", "status", "uninstall", "reset", "inject"))
    req.add_argument("--steps", type=int, default=None, help="RL steps to run after the verb (default: 1 for install/check, else 0)")
    req.add_argument("--timeout", type=float, default=1800.0)
    req.add_argument("--full", action="store_true")
    req.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    req.add_argument("--id", default=None, help="uninstall: undo this transaction and every later one")
    req.add_argument("--fault", default=None, help="inject: fail_apply | fail_commit | crash | hang | lost | tamper")
    req.add_argument("--participant", default=None, help="inject: driver, reward<i>, rollout<i> or trainer<i>")
    req.add_argument("files", nargs="*")
    sub.add_parser("stop").add_argument("--gate", required=True)
    conf = sub.add_parser("conformance", help="run core/livelab conformance c0-c10 against a live gated job")
    conf.add_argument("--gate", required=True)
    conf.add_argument("--launch", default=None, help="shell command that starts the gated job on $GATE with reward $REWARD (needed for respawn and fresh checks)")
    conf.add_argument("--reward", required=True, help="the custom reward file the job loads")
    conf.add_argument("--only", default="")
    conf.add_argument("--receipts", default=None)
    conf.add_argument("--json", action="store_true")
    conf.add_argument("--check-timeout", type=float, default=None)
    conf.add_argument("--boot-timeout", type=float, default=None)
    res = sub.add_parser("rescore")
    res.add_argument("--rollouts", required=True)
    res.add_argument("--reward", required=True)
    res.add_argument("--function", default="compute_score")
    res.add_argument("--data-source", default=None)
    adv = sub.add_parser("replay-adv")
    adv.add_argument("--step", required=True)
    adv.add_argument("--atol", type=float, default=0.0)
    adv.add_argument("--fn", default=None)
    trn = sub.add_parser("train")
    trn.add_argument("--gate", default=None)
    trn.add_argument("--trace", default=None)
    trn.add_argument("--roots", default=None)
    trn.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    if a.command == "train":
        cmd, env = launch_command(a.overrides, a.gate, a.trace, a.roots.split(os.pathsep) if a.roots else None)
        if a.gate:
            Path(a.gate).mkdir(parents=True, exist_ok=True)
            for stale in Path(a.gate).glob("*.json"):
                stale.unlink()
        os.execvpe(cmd[0], cmd, env)
    if a.command == "stop":
        request(a.gate, stop=True)
        return 0
    if a.command == "conformance":
        import adapter  # noqa: F401  (puts core/livelab on sys.path)
        from adapter.conformance_lab import VerlConformanceLab
        launch_argv = ["--gate", a.gate, "--reward", a.reward]
        if a.launch is not None:
            launch_argv += ["--launch", a.launch]
        if a.check_timeout is not None:
            launch_argv += ["--check-timeout", str(a.check_timeout)]
        if a.boot_timeout is not None:
            launch_argv += ["--boot-timeout", str(a.boot_timeout)]
        os.environ.update(VerlConformanceLab.launch_environment(launch_argv))
        from conformance.__main__ import main as conformance_main
        argv = ["--lab", "adapter.conformance_lab:VerlConformanceLab"] + (["--only", a.only] if a.only else []) + (["--receipts", a.receipts] if a.receipts else []) + (["--json"] if a.json else [])
        return conformance_main(argv)
    if a.command == "rescore":
        from tools import rescore
        print(json.dumps(rescore.rescore(a.rollouts, a.reward, a.function, a.data_source)))
        return 0
    if a.command == "replay-adv":
        from tools import replay_adv
        print(json.dumps(replay_adv.replay_advantages(a.step, atol=a.atol, estimator_spec=a.fn)))
        return 0
    pairs = [it.partition("=") for it in a.set]
    if any(not sep for _, sep, _ in pairs):
        ap.error("--set expects KEY=VALUE")
    steps = a.steps if a.steps is not None else (1 if a.verb in ("install", "check") else 0)
    extra = {**({"id": a.id} if a.id else {}), **({"fault": a.fault} if a.fault else {}), **({"participant": a.participant} if a.participant else {})}
    if a.verb == "inject" and not a.fault:
        ap.error("--verb inject needs --fault")
    try:
        rcpt = request(a.gate, files=a.files, steps=steps, timeout=a.timeout, live_settings={k: v for k, _, v in pairs}, verb=a.verb, extra=extra)
    except GateError as err:
        print(json.dumps({"ok": False, "error": str(err)}))
        return 2
    print(json.dumps(rcpt if a.full else summary(rcpt), default=str))
    return 3 if rcpt.get("restart_required") else (0 if rcpt.get("ok") else 1)


if __name__ == "__main__":
    sys.exit(main())
