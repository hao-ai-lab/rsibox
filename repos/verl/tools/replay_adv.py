"""Offline advantage probe (`replay_adv`, ~12 ms for 640 trajectories on CPU)."""
import argparse, importlib, importlib.util, json, sys, time
from pathlib import Path


def _load_fn(spec_str):
    target, _, fn_name = spec_str.partition(":")
    fn_name = fn_name or "grpo_advantages"
    if Path(target).is_file():
        spec = importlib.util.spec_from_file_location("fastrsi_verl_adv_target", target)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return getattr(loaded, fn_name)
    return getattr(importlib.import_module(target), fn_name)


def replay_json(record_path, estimator_spec, atol=0.0):
    rec = json.loads(Path(record_path).read_text(encoding="utf-8"))
    fn = _load_fn(estimator_spec)
    t0 = time.perf_counter()
    after_groups = [list(fn(list(g))) for g in rec["reward_groups"]]
    ms = round((time.perf_counter() - t0) * 1000.0, 3)
    diffs, rows_changed, fb, fa = [], 0, [], []
    for b_row, a_row in zip(rec["advantage_groups"], after_groups):
        delta = [abs(float(a) - float(b)) for b, a in zip(b_row, a_row)]
        diffs.extend(delta)
        rows_changed += int(any(d > atol for d in delta))
        fb.extend(float(x) for x in b_row)
        fa.extend(float(x) for x in a_row)
    mx = max(diffs, default=0.0)
    return {"rows": len(rec["advantage_groups"]), "equal": bool(mx <= atol), "max_abs_diff": round(float(mx), 6),
            "rows_changed": rows_changed, "before": {"max": round(max(fb, default=0.0), 6), "min": round(min(fb, default=0.0), 6)},
            "after": {"max": round(max(fa, default=0.0), 6), "min": round(min(fa, default=0.0), 6)}, "compute_ms": ms}


def replay_advantages(step_dir, atol=0.0, estimator_spec=None):
    step_path = Path(step_dir)
    json_file = step_path / "adv.json" if step_path.is_dir() else step_path
    if json_file.suffix == ".json" and json_file.is_file():
        if not estimator_spec:
            raise ValueError("JSON step bundle requires --fn <module_or_file:function>")
        return replay_json(json_file, estimator_spec, atol=atol)
    import numpy as np, torch
    from omegaconf import OmegaConf
    from verl.protocol import DataProto
    from verl.trainer.ppo.core_algos import AdvantageEstimator
    from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories
    rec = torch.load(step_path / "adv.pt", weights_only=False)
    data = DataProto.from_dict(tensors={k: v.clone() for k, v in rec["inputs"].items()},
                               non_tensors={"uid": np.array(rec["uid"], dtype=object)})
    kw = dict(rec["kwargs"])
    for member in AdvantageEstimator:
        if kw["adv_estimator"] in (str(member), member.value, member.name):
            kw["adv_estimator"] = member
            break
    cfg = OmegaConf.create(rec["config"]) if rec.get("config") is not None else None
    t0 = time.perf_counter()
    out = compute_advantage_for_multi_trajectories(data, rec["batch_keys"], config=cfg, **kw)
    ms = round((time.perf_counter() - t0) * 1000.0, 3)
    before, after = rec["advantages"], out.batch["advantages"]
    diff, mask = (after - before).abs(), rec["inputs"]["response_mask"].bool()
    return {"rows": int(before.shape[0]), "equal": bool(diff.max() <= atol), "max_abs_diff": float(diff.max()),
            "rows_changed": int((diff > atol).any(dim=1).sum()),
            "before": {"max": float(before[mask].max()), "min": float(before[mask].min())},
            "after": {"max": float(after[mask].max()), "min": float(after[mask].min())}, "compute_ms": ms}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m tools.replay_adv", description=__doc__)
    ap.add_argument("--step", required=True)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--fn", default=None)
    a = ap.parse_args(argv)
    print(json.dumps(replay_advantages(a.step, atol=a.atol, estimator_spec=a.fn)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
