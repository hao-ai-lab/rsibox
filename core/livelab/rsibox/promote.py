"""Clean cold-start promotion gate (Lessons 13 & 14).

Runs paired, alternating fresh subprocesses (`base/candidate`, `candidate/base`, ...) across
>= 2 distinct input shapes, enforcing strict finite positive metrics, numerical parity,
per-shape recompile counts, process-group cleanup, and signed verdict receipts.
Integrity note: unkeyed SHA-256 receipts detect accidental edits or corruption; pass
`hmac_key` to `run_promotion` / `verify_promotion_receipt` for keyed HMAC-SHA256 verification.
"""
import argparse, hashlib, hmac, json, math, os, platform, re, shlex, statistics, subprocess, sys, threading, time
from typing import Any

from . import proc as proc_mod

SCHEMA, RESULT_PREFIX = "rsibox-promotion-receipt/v1", "RSIBOX_PROMOTE_RESULT:"
_TOKEN_RE = re.compile(r"\{(?:shapes_json|arm|pair)\}")
_REQUIRED_KEYS = ("schema", "created_at", "verdict", "reasons", "policy", "shapes", "pairs", "summary", "artifacts", "env_fingerprint", "receipt_sha256")


def _strict_loads(text: str) -> Any:
    def _reject(token: str) -> Any:
        raise ValueError(f"non-standard JSON constant {token!r} is forbidden")
    def _no_dups(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, val in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key!r}")
            out[key] = val
        return out
    return json.loads(text, parse_constant=_reject, object_pairs_hook=_no_dups)

def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

def _digest(obj: Any, hmac_key: str | bytes | None = None) -> str:
    payload = _canon(obj).encode("utf-8")
    if hmac_key is None:
        return hashlib.sha256(payload).hexdigest()
    return hmac.new(hmac_key.encode("utf-8") if isinstance(hmac_key, str) else bytes(hmac_key), payload, hashlib.sha256).hexdigest()

def _write_strict_json(path: str | os.PathLike, payload: dict[str, Any]) -> None:
    target = os.fspath(path)
    if parent := os.path.dirname(target):
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{target}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
    os.replace(tmp_path, target)

def _is_num(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)

def _shape_key(shape: Any, idx: int) -> str:
    return shape if isinstance(shape, str) else (shape["name"] if isinstance(shape, dict) and isinstance(shape.get("name"), str) and shape["name"] else f"shape_{idx}")


def _outputs_match(base_val: Any, cand_val: Any, *, atol: float | None = None, rtol: float = 0.0, path: str = "$") -> tuple[bool, str | None]:
    is_num_pair = isinstance(base_val, (int, float)) and not isinstance(base_val, bool) and isinstance(cand_val, (int, float)) and not isinstance(cand_val, bool)
    if is_num_pair:
        base_f, cand_f = float(base_val), float(cand_val)
        if not (math.isfinite(base_f) and math.isfinite(cand_f)):
            return False, f"{path}: non-finite number (base={base_f}, candidate={cand_f})"
        if atol is None and rtol == 0.0:
            return (True, None) if base_val == cand_val else (False, f"{path}: base={base_val!r} != candidate={cand_val!r}")
        diff, limit = abs(cand_f - base_f), (atol or 0.0) + rtol * abs(base_f)
        return (True, None) if diff <= limit else (False, f"{path}: diff={diff:.6g} > tol={limit:.6g}")
    if type(base_val) is not type(cand_val):
        return False, f"{path}: type mismatch ({type(base_val).__name__} vs {type(cand_val).__name__})"
    if isinstance(base_val, dict):
        if set(base_val) != set(cand_val):
            return False, f"{path}: keys differ ({sorted(base_val)} vs {sorted(cand_val)})"
        return next((res for key in sorted(base_val) if not (res := _outputs_match(base_val[key], cand_val[key], atol=atol, rtol=rtol, path=f"{path}.{key}"))[0]), (True, None))
    if isinstance(base_val, list):
        if len(base_val) != len(cand_val):
            return False, f"{path}: length mismatch ({len(base_val)} vs {len(cand_val)})"
        return next((res for idx, (bv, cv) in enumerate(zip(base_val, cand_val)) if not (res := _outputs_match(bv, cv, atol=atol, rtol=rtol, path=f"{path}[{idx}]"))[0]), (True, None))
    return (True, None) if base_val == cand_val else (False, f"{path}: base={base_val!r} != candidate={cand_val!r}")


def _env_fingerprint() -> dict[str, Any]:
    affinity_n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    try:
        import resource
        nofile_soft = int(resource.getrlimit(resource.RLIMIT_NOFILE)[0])
    except Exception:
        nofile_soft = None
    return {"python": sys.version.split()[0], "platform": sys.platform, "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "affinity_cpus": affinity_n, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "cuda_visible_devices_set": "CUDA_VISIBLE_DEVICES" in os.environ, "rlimit_nofile_soft": nofile_soft}


def _parse_payload(stdout: str) -> tuple[dict | None, str | None]:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for prefix_only in (True, False):
        for ln in reversed(lines):
            if (ln.startswith(RESULT_PREFIX) if prefix_only else (ln.startswith("{") and ln.endswith("}"))):
                try:
                    parsed = _strict_loads(ln[len(RESULT_PREFIX):].strip() if prefix_only else ln)
                    return (parsed, None) if isinstance(parsed, dict) else (None, "result is not a JSON object")
                except ValueError as exc:
                    return None, f"invalid JSON payload: {exc}"
    return None, "no JSON result object found on stdout"

def _normalize_shapes(payload: dict, shapes: list[Any]) -> tuple[list[dict] | None, str | None]:
    raw = payload.get("shapes")
    if isinstance(raw, dict):
        items = []
        for idx, shape in enumerate(shapes):
            key = _shape_key(shape, idx)
            if not isinstance(raw.get(key), dict):
                return None, f"missing or invalid result for shape {key!r}"
            items.append({**raw[key], "shape": raw[key].get("shape", key)})
        return items, None
    if isinstance(raw, list) and len(raw) == len(shapes) and all(isinstance(entry, dict) for entry in raw):
        return [{**entry, "shape": entry.get("shape", _shape_key(shapes[idx], idx))} for idx, entry in enumerate(raw)], None
    return None, f"expected {len(shapes)} shape result objects in 'shapes'"

def _arm_err(arm: str, returncode: int | None, wall_s: float, err: str) -> dict[str, Any]:
    return {"arm": arm, "ok": False, "returncode": returncode, "wall_s": wall_s, "metric": None, "_raw_metric": None, "error": err, "shapes": [], "_raw_outputs": []}

def _cleanup_proc(child: subprocess.Popen) -> None:
    proc_mod.stop_session(child.pid, grace=0.3, reap=child.poll, kill_grace=0.5)
    if child.poll() is None:
        child.kill()
        try:
            child.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
    for stream in (child.stdout, child.stderr):
        if stream is not None:
            stream.close()


def _run_arm(cmd: list[str], *, arm: str, pair_idx: int, shapes: list[Any], timeout_s: float, cwd: str | None, extra_env: dict[str, str] | None) -> dict[str, Any]:
    shapes_json = _canon(shapes)
    tokens = {"{shapes_json}": shapes_json, "{arm}": arm, "{pair}": str(pair_idx)}
    argv = [_TOKEN_RE.sub(lambda match: tokens[match.group(0)], tok) for tok in cmd]
    env = {**os.environ, **(extra_env or {}), "RSIBOX_PROMOTE_SHAPES": shapes_json, "RSIBOX_PROMOTE_ARM": arm, "RSIBOX_PROMOTE_PAIR": str(pair_idx)}
    t0 = time.monotonic()
    try:
        child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    except OSError as exc:
        return _arm_err(arm, None, round(time.monotonic() - t0, 6), f"{type(exc).__name__}: {exc}")
    try:
        stdout, stderr = child.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _cleanup_proc(child)
        return _arm_err(arm, None, round(time.monotonic() - t0, 6), f"TimeoutExpired: exceeded {timeout_s}s")
    except BaseException:
        _cleanup_proc(child)
        raise
    if proc_mod.pg_alive(child.pid):
        proc_mod.stop_session(child.pid, grace=0.1, reap=child.poll, kill_grace=0.5)
    wall_s = round(time.monotonic() - t0, 6)
    if child.returncode != 0:
        tail = (stderr or stdout).strip().splitlines()
        return _arm_err(arm, child.returncode, wall_s, f"process exited {child.returncode}: {tail[-1][:200] if tail else 'failed'}")
    payload, err = _parse_payload(stdout)
    if payload is None or not isinstance(payload.get("ok"), bool):
        return _arm_err(arm, child.returncode, wall_s, err or "top-level 'ok' must be a boolean")
    norm, err = _normalize_shapes(payload, shapes)
    if norm is None:
        return _arm_err(arm, child.returncode, wall_s, err or "shape error")
    try:
        top_ok, metrics, summaries, raw_outputs = payload["ok"] is True, [], [], []
        for idx, item in enumerate(norm):
            shape_name = str(item.get("shape", _shape_key(shapes[idx], idx)))
            recomp = item.get("recompiles")
            if not isinstance(item.get("ok"), bool):
                return _arm_err(arm, child.returncode, wall_s, f"shape {shape_name!r}: 'ok' must be a boolean")
            if "output" not in item or type(recomp) is not int or recomp < 0:
                msg = "missing 'output' field" if "output" not in item else "'recompiles' must be int >= 0"
                return _arm_err(arm, child.returncode, wall_s, f"shape {shape_name!r}: {msg}")
            shape_ok = item["ok"] is True
            metric_raw = item["metric"] if "metric" in item else item.get("wall_s", item.get("latency_s"))
            if shape_ok and (not _is_num(metric_raw) or float(metric_raw) <= 0.0):
                return _arm_err(arm, child.returncode, wall_s, f"shape {shape_name!r}: finite positive metric (> 0) required")
            metric_f = float(metric_raw) if _is_num(metric_raw) and float(metric_raw) > 0.0 else None
            if metric_f is not None:
                metrics.append(metric_f)
            raw_outputs.append(item["output"])
            summaries.append({"shape": shape_name, "ok": shape_ok, "metric": metric_f, "recompiles": recomp,
                              "output_sha256": _digest(item["output"]), **({"error": str(item["error"])[:200]} if item.get("error") else {})})
        arm_ok, raw_metric = top_ok and all(entry["ok"] for entry in summaries), None
        if "metric" in payload:
            if not _is_num(payload["metric"]) or float(payload["metric"]) <= 0.0:
                return _arm_err(arm, child.returncode, wall_s, "top-level 'metric' must be a finite positive number")
            raw_metric = float(payload["metric"])
        elif arm_ok and len(metrics) == len(norm) and metrics:
            raw_metric = sum(metrics)
    except (TypeError, ValueError, OverflowError) as exc:
        return _arm_err(arm, child.returncode, wall_s, f"invalid child payload: {exc}")
    return {"arm": arm, "ok": arm_ok, "returncode": child.returncode, "wall_s": wall_s,
            "metric": round(raw_metric, 6) if raw_metric is not None else None, "_raw_metric": raw_metric,
            "error": str(payload["error"])[:200] if payload.get("error") else None, "shapes": summaries, "_raw_outputs": raw_outputs}


def _sign_receipt(body: dict[str, Any], hmac_key: str | bytes | None = None) -> dict[str, Any]:
    clean = {k: v for k, v in body.items() if k != "receipt_sha256"}
    clean["signature_mode"] = "hmac-sha256" if hmac_key is not None else "sha256"
    return {**clean, "receipt_sha256": _digest(clean, hmac_key=hmac_key)}


def verify_promotion_receipt(receipt_or_path: dict[str, Any] | str | os.PathLike, *, hmac_key: str | bytes | None = None) -> dict[str, Any]:
    """Verify integrity and schema of a signed promotion receipt dict or JSON file."""
    if isinstance(receipt_or_path, (str, os.PathLike)):
        try:
            with open(receipt_or_path, encoding="utf-8") as handle:
                receipt = _strict_loads(handle.read())
        except (OSError, ValueError) as exc:
            return {"ok": False, "verdict": None, "receipt_sha256": None, "errors": [f"cannot read receipt: {exc}"]}
    else:
        receipt = receipt_or_path
    if not isinstance(receipt, dict):
        return {"ok": False, "verdict": None, "receipt_sha256": None, "errors": ["receipt must be a JSON object"]}
    errors = [f"missing required field: {key}" for key in _REQUIRED_KEYS if key not in receipt]
    if receipt.get("schema") != SCHEMA:
        errors.append(f"unexpected schema: {receipt.get('schema')!r}")
    verdict, reasons = receipt.get("verdict"), receipt.get("reasons")
    if verdict not in ("promote", "reject"):
        errors.append(f"invalid verdict: {verdict!r}")
    if not isinstance(reasons, list):
        errors.append("reasons must be a list")
    elif (verdict == "promote" and reasons) or (verdict == "reject" and not reasons):
        errors.append(f"verdict {verdict!r} is inconsistent with reasons")
    actual, expected = receipt.get("receipt_sha256"), None
    try:
        expected = _digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}, hmac_key=hmac_key)
    except (TypeError, ValueError) as exc:
        errors.append(f"receipt contains unserializable or non-finite values: {exc}")
    if not isinstance(actual, str) or len(actual) != 64:
        errors.append("receipt_sha256 must be a 64-char hex digest")
    elif expected is not None and not hmac.compare_digest(actual, expected):
        errors.append("receipt_sha256 mismatch (receipt was tampered with)")
    return {"ok": not errors, "verdict": verdict, "receipt_sha256": actual, "expected_sha256": expected, "errors": errors}


def run_promotion(
    base_cmd: list[str], candidate_cmd: list[str], shapes: list[Any], *,
    pairs: int = 2, allow_single_pair: bool = False, min_shapes: int = 2, allow_single_shape: bool = False,
    min_speedup: float = 1.0, atol: float | None = None, rtol: float = 0.0, tie_tol: float = 1e-6,
    max_extra_recompiles: int = 0, allow_extra_recompiles: bool = False, higher_is_better: bool = False,
    timeout_s: float = 120.0, cwd: str | None = None, env: dict[str, str] | None = None,
    candidate_id: str | None = None, hmac_key: str | bytes | None = None, receipt_path: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Run paired, alternating fresh-process evaluations and return a signed verdict receipt."""
    if not base_cmd or not candidate_cmd:
        raise ValueError("base_cmd and candidate_cmd must be non-empty argv lists")
    if type(pairs) is not int or pairs < 1 or (not allow_single_pair and (pairs < 2 or pairs % 2 != 0)):
        raise ValueError(f"pairs must be an even integer >= 2 for balanced alternation (got {pairs}); pass allow_single_pair=True / --allow-single-pair to override")
    if type(min_shapes) is not int or min_shapes < 1 or (not allow_single_shape and min_shapes < 2):
        raise ValueError(f"min_shapes must be >= 2 unless allow_single_shape=True (got {min_shapes})")
    if not (_is_num(min_speedup) and min_speedup > 0 and _is_num(timeout_s) and timeout_s > 0):
        raise ValueError("min_speedup and timeout_s must be finite numbers > 0")
    if (atol is not None and not (_is_num(atol) and atol >= 0)) or not (_is_num(rtol) and rtol >= 0 and _is_num(tie_tol) and tie_tol >= 0) or type(max_extra_recompiles) is not int or max_extra_recompiles < 0:
        raise ValueError("atol, rtol, tie_tol must be finite >= 0 and max_extra_recompiles an int >= 0")
    req_shapes = 1 if allow_single_shape else min_shapes
    if (n_distinct := len({_canon(shape) for shape in shapes})) < req_shapes:
        raise ValueError(f"promotion requires at least {req_shapes} distinct input shapes (got {n_distinct}); pass allow_single_shape=True / --allow-single-shape to override")

    reasons, pair_records, ratios, base_metrics, cand_metrics = [], [], [], [], []
    correctness_ok, recompile_ok, wins, ties, losses = True, True, 0, 0, 0
    for p_idx in range(pairs):
        order = ("base", "candidate") if p_idx % 2 == 0 else ("candidate", "base")
        runs = {arm: _run_arm(base_cmd if arm == "base" else candidate_cmd, arm=arm, pair_idx=p_idx, shapes=shapes, timeout_s=timeout_s, cwd=cwd, extra_env=env) for arm in order}
        base_r, cand_r, pair_reasons = runs["base"], runs["candidate"], []
        if not base_r["ok"]:
            correctness_ok = False
            pair_reasons.append(f"pair {p_idx}: base failed ({base_r['error'] or 'shape failure'})")
        if not cand_r["ok"]:
            correctness_ok = False
            bad = [entry["shape"] for entry in cand_r["shapes"] if not entry["ok"]]
            pair_reasons.append(f"pair {p_idx}: candidate failed ({f'failed shapes {bad}' if bad else (cand_r['error'] or 'failed')})")
        if base_r["shapes"] and len(base_r["shapes"]) == len(cand_r["shapes"]):
            for s_idx, (bs, cs) in enumerate(zip(base_r["shapes"], cand_r["shapes"])):
                if bs["ok"] and cs["ok"] and not (chk := _outputs_match(base_r["_raw_outputs"][s_idx], cand_r["_raw_outputs"][s_idx], atol=atol, rtol=rtol))[0]:
                    correctness_ok = False
                    pair_reasons.append(f"pair {p_idx} shape {cs['shape']!r}: numerical parity failed ({chk[1]})")
                if s_idx >= 1 and not allow_extra_recompiles and (cs["recompiles"] - bs["recompiles"]) > max_extra_recompiles:
                    recompile_ok = False
                    pair_reasons.append(f"pair {p_idx} shape {cs['shape']!r}: excessive recompiles (candidate={cs['recompiles']}, base={bs['recompiles']}, allowed_extra={max_extra_recompiles})")
        speedup, outcome, bm, cm = None, "invalid", base_r["_raw_metric"], cand_r["_raw_metric"]
        if base_r["ok"] and cand_r["ok"] and _is_num(bm) and _is_num(cm) and bm > 0 and cm > 0:
            raw_ratio = (cm / bm) if higher_is_better else (bm / cm)
            speedup = round(raw_ratio, 6)
            ratios.append(raw_ratio); base_metrics.append(bm); cand_metrics.append(cm)
            if raw_ratio > 1.0 + tie_tol:
                outcome, wins = "win", wins + 1
            elif raw_ratio < 1.0 - tie_tol:
                outcome, losses = "loss", losses + 1
            else:
                outcome, ties = "tie", ties + 1
        else:
            correctness_ok = False
            if not pair_reasons:
                pair_reasons.append(f"pair {p_idx}: invalid metric (base={bm!r}, candidate={cm!r})")
        reasons.extend(pair_reasons)
        pair_records.append({"pair": p_idx, "order": list(order), "speedup": speedup, "outcome": outcome, "reasons": pair_reasons,
                             "base": {k: v for k, v in base_r.items() if not k.startswith("_")},
                             "candidate": {k: v for k, v in cand_r.items() if not k.startswith("_")}})

    med_raw, mean_raw = (statistics.median(ratios), statistics.fmean(ratios)) if ratios else (None, None)
    med_sp, mean_sp = (round(med_raw, 6) if med_raw is not None else None), (round(mean_raw, 6) if mean_raw is not None else None)
    if len(ratios) != pairs:
        speedup_ok = False
        if not ratios:
            reasons.append("no valid timed pairs completed")
    else:
        speedup_ok = (med_raw >= min_speedup and wins > losses) if min_speedup > 1.0 else ((med_raw > 1.0 and wins > losses) if min_speedup == 1.0 else (med_raw >= min_speedup))
        if not speedup_ok:
            reasons.append(f"insufficient speedup (median={med_sp}x, mean={mean_sp}x, W/T/L={wins}/{ties}/{losses}, min={min_speedup}x)")

    receipt = _sign_receipt({
        "schema": SCHEMA, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": "promote" if (correctness_ok and recompile_ok and speedup_ok and not reasons) else "reject",
        "reasons": reasons, "shapes": list(shapes), "pairs": pair_records,
        "artifacts": {"base_cmd_sha256": _digest(list(base_cmd)), "candidate_cmd_sha256": _digest(list(candidate_cmd)), **({"candidate_id": candidate_id} if candidate_id else {})},
        "policy": {"min_shapes": req_shapes, "allow_single_shape": allow_single_shape, "pairs": pairs, "allow_single_pair": allow_single_pair,
                   "min_speedup": min_speedup, "atol": atol, "rtol": rtol, "tie_tol": tie_tol,
                   "max_extra_recompiles": max_extra_recompiles, "allow_extra_recompiles": allow_extra_recompiles, "higher_is_better": higher_is_better},
        "summary": {"pairs_total": len(pair_records), "wins": wins, "ties": ties, "losses": losses,
                    "base_median": round(statistics.median(base_metrics), 6) if base_metrics else None, "base_mean": round(statistics.fmean(base_metrics), 6) if base_metrics else None,
                    "candidate_median": round(statistics.median(cand_metrics), 6) if cand_metrics else None, "candidate_mean": round(statistics.fmean(cand_metrics), 6) if cand_metrics else None,
                    "median_speedup": med_sp, "mean_speedup": mean_sp, "correctness_ok": correctness_ok, "recompile_ok": recompile_ok, "speedup_ok": speedup_ok},
        "env_fingerprint": _env_fingerprint(),
    }, hmac_key=hmac_key)
    if receipt_path is not None:
        _write_strict_json(receipt_path, receipt)
    return receipt


def _parse_cmd(text: str) -> list[str]:
    stripped = text.strip()
    if stripped.startswith("[") and isinstance(parsed := _strict_loads(stripped), list) and all(isinstance(x, str) for x in parsed):
        return parsed
    return shlex.split(stripped)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rsibox promote", description="Paired, alternating cold-start promotion gate.")
    for opt, kw in (("--verify", {"default": None, "metavar": "RECEIPT_JSON"}), ("--base-cmd", {"default": None}), ("--candidate-cmd", {"default": None}),
                    ("--candidate-id", {"default": None}), ("--shape", {"action": "append", "default": []}), ("--pairs", {"type": int, "default": 2}),
                    ("--allow-single-pair", {"action": "store_true"}), ("--min-shapes", {"type": int, "default": 2}), ("--allow-single-shape", {"action": "store_true"}),
                    ("--min-speedup", {"type": float, "default": 1.0}), ("--atol", {"type": float, "default": None}), ("--rtol", {"type": float, "default": 0.0}),
                    ("--max-extra-recompiles", {"type": int, "default": 0}), ("--allow-extra-recompiles", {"action": "store_true"}), ("--higher-is-better", {"action": "store_true"}),
                    ("--timeout", {"type": float, "default": 120.0}), ("--receipt", {"default": None}), ("--json", {"action": "store_true"})):
        parser.add_argument(opt, **kw)
    args = parser.parse_args(argv)
    if args.verify:
        res = verify_promotion_receipt(args.verify)
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1
    if not args.base_cmd or not args.candidate_cmd:
        print("rsibox promote: --base-cmd and --candidate-cmd are required (or use --verify)", file=sys.stderr)
        return 2
    def _shape_parse(raw: str) -> Any:
        try:
            return _strict_loads(raw)
        except ValueError:
            return raw
    try:
        receipt = run_promotion(
            _parse_cmd(args.base_cmd), _parse_cmd(args.candidate_cmd), [_shape_parse(raw) for raw in args.shape],
            pairs=args.pairs, allow_single_pair=args.allow_single_pair, min_shapes=args.min_shapes, allow_single_shape=args.allow_single_shape,
            min_speedup=args.min_speedup, atol=args.atol, rtol=args.rtol, max_extra_recompiles=args.max_extra_recompiles,
            allow_extra_recompiles=args.allow_extra_recompiles, higher_is_better=args.higher_is_better,
            timeout_s=args.timeout, candidate_id=args.candidate_id, receipt_path=args.receipt,
        )
    except ValueError as exc:
        print(f"rsibox promote: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(receipt, indent=1))
    else:
        summary = receipt["summary"]
        print(f"verdict: {receipt['verdict']} (median={summary['median_speedup']}x, mean={summary['mean_speedup']}x, W/T/L={summary['wins']}/{summary['ties']}/{summary['losses']})")
        print("\n".join([*(f"  reason: {reason}" for reason in receipt["reasons"]), f"receipt_sha256: {receipt['receipt_sha256']}"]))
    return 0 if receipt["verdict"] == "promote" else 1
