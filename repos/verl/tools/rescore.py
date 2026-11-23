"""Offline reward probe (`rescore`, ~0.5 ms for 640 recorded rollouts on CPU)."""
import argparse, importlib.util, json, sys, time
from pathlib import Path


def rescore(rollouts_path, reward_path, function="compute_score", data_source=None, show=5):
    rows = [json.loads(ln) for ln in Path(rollouts_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    spec = importlib.util.spec_from_file_location("rsibox_verl_rescore_target", reward_path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    fn = getattr(loaded, function)
    t0 = time.perf_counter()
    new_scores = []
    for row in rows:
        raw = fn(data_source=data_source if data_source is not None else row.get("data_source"),
                 solution_str=row["output"], ground_truth=row["gts"], extra_info=row.get("extra_info") or {})
        new_scores.append(float(raw["score"] if isinstance(raw, dict) else raw))
    ms = round((time.perf_counter() - t0) * 1000.0, 3)
    old_scores = [float(r["score"]) for r in rows]
    changed = [i for i, (a, b) in enumerate(zip(old_scores, new_scores)) if a != b]
    n = max(len(rows), 1)
    return {"responses": len(rows), "changed": len(changed),
            "mean_before": round(sum(old_scores) / n, 4), "mean_after": round(sum(new_scores) / n, 4),
            "score_ms": ms, "examples": [{"index": i, "before": old_scores[i], "after": new_scores[i],
                                          "gts": rows[i]["gts"], "tail": str(rows[i]["output"])[-120:]} for i in changed[:show]]}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m tools.rescore", description=__doc__)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--reward", required=True)
    ap.add_argument("--function", default="compute_score")
    ap.add_argument("--data-source", default=None)
    a = ap.parse_args(argv)
    print(json.dumps(rescore(a.rollouts, a.reward, a.function, a.data_source)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
