#!/usr/bin/env python3
"""Read-only schema and cross-file consistency validator for `site/facts/*.json` and `README.md`.

Pure Python stdlib (zero third-party dependencies). Validates:
1. Valid JSON and required schema across:
   - `site/facts/engines.json`
   - `site/facts/coverage.json`
   - `site/facts/prs.json`
   - `site/facts/case-copy.json`
   - `site/facts/case-timing.json`
   - `site/facts/conversion.json`
2. Internal counts and timing arithmetic (`prs.json` `meta.counts`, `case-timing.json`
   phase sums vs arm `total`, `effect` ratios vs `edit_step_s / base_step_s`).
3. Cross-file joins (`repos/*` <-> `engines.json` <-> `conversion.json`;
   `case-copy.json` <-> `case-timing.json` <-> `prs.json`).
4. Split-brain drift across `README.md` engine tables / repo layout and `coverage.json`
   (missing engines, stale conformance adapter claims when a `*Lab` class exists on disk).

Exit codes:
- 0: no schema/math/join errors (and, with `--strict`, no split-brain drifts)
- 1: one or more schema/math/join errors
- 2: `--strict` set and one or more split-brain drifts detected
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys

REQUIRED_FACT_FILES = (
    "engines.json",
    "coverage.json",
    "prs.json",
    "case-copy.json",
    "case-timing.json",
    "conversion.json",
)


def _norm(name: str) -> str:
    """Normalize an engine display or folder name (`Megatron-LM` -> `megatron`, `vLLM` -> `vllm`)."""
    token = re.sub(r"[^a-z0-9]+", "", name.lower())
    if token == "megatronlm":
        return "megatron"
    return token


def _has_real_conformance_lab(engine_dir: Path) -> bool:
    """AST-check whether `<engine_dir>/livelab/*/{conformance_lab,lab}.py` defines a `*Lab` class."""
    livelab = engine_dir / "livelab"
    if not livelab.is_dir():
        return False
    for entry in sorted(livelab.iterdir()):
        if not entry.is_dir() or entry.name in ("services", "__pycache__"):
            continue
        if not (entry / "__init__.py").is_file():
            continue
        for stem in ("conformance_lab.py", "lab.py"):
            cand = entry / stem
            if not cand.is_file():
                continue
            try:
                tree = ast.parse(cand.read_text(encoding="utf-8"), filename=str(cand))
            except (OSError, SyntaxError):
                continue
            if any(isinstance(n, ast.ClassDef) and n.name.endswith("Lab") for n in tree.body):
                return True
    return False


def _load_json(path: Path, findings: list[dict]):
    if not path.is_file():
        findings.append({
            "severity": "error",
            "code": "missing_file",
            "file": path.name,
            "message": f"required file missing: {path}",
        })
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        findings.append({
            "severity": "error",
            "code": "invalid_json",
            "file": path.name,
            "message": f"failed to parse JSON: {exc}",
        })
        return None


def _add(findings: list[dict], severity: str, code: str, file: str, message: str) -> None:
    findings.append({
        "severity": severity,
        "code": code,
        "file": file,
        "message": message,
    })


def _check_engines_and_conversion(
    root: Path,
    repo_folders: list[str],
    engines_data,
    conversion_data,
    findings: list[dict],
) -> dict[str, str]:
    """Validate `engines.json` and `conversion.json`, returning `{display_name: folder_basename}`."""
    known_engines: dict[str, str] = {}
    if isinstance(engines_data, dict):
        for req_key, expected_type in (("shared_kernel", dict), ("engines", list), ("caveats", list)):
            if not isinstance(engines_data.get(req_key), expected_type):
                _add(
                    findings, "error", "schema_error", "site/facts/engines.json",
                    f"top-level {req_key!r} must be {expected_type.__name__}",
                )
        seen_names: set[str] = set()
        seen_folders: set[str] = set()
        for idx, entry in enumerate(engines_data.get("engines") or ()):
            if not isinstance(entry, dict):
                _add(findings, "error", "schema_error", "site/facts/engines.json", f"engines[{idx}] must be object")
                continue
            name = entry.get("engine")
            folder = entry.get("folder")
            mechs = entry.get("mechanisms")
            if not isinstance(name, str) or not name.strip():
                _add(findings, "error", "schema_error", "site/facts/engines.json", f"engines[{idx}].engine missing")
                continue
            if not isinstance(folder, str) or not folder.strip():
                _add(findings, "error", "schema_error", "site/facts/engines.json", f"engines[{idx}] ({name}) missing folder")
                continue
            if not isinstance(mechs, list):
                _add(findings, "error", "schema_error", "site/facts/engines.json", f"engines[{idx}] ({name}) mechanisms must be list")
            if name in seen_names:
                _add(findings, "error", "duplicate_engine", "site/facts/engines.json", f"duplicate engine {name!r}")
            seen_names.add(name)
            base = Path(folder).name
            if base in seen_folders:
                _add(findings, "error", "duplicate_folder", "site/facts/engines.json", f"duplicate folder {folder!r}")
            seen_folders.add(base)
            known_engines[name] = base
            if not (root / folder).is_dir():
                _add(findings, "error", "missing_repo_folder", "site/facts/engines.json", f"folder {folder!r} does not exist")
        for folder in repo_folders:
            if folder not in seen_folders:
                _add(
                    findings, "error", "missing_engine_in_facts", "site/facts/engines.json",
                    f"repo engine 'repos/{folder}' missing from engines.json",
                )
    elif engines_data is not None:
        _add(findings, "error", "schema_error", "site/facts/engines.json", "root must be JSON object")

    if isinstance(conversion_data, dict):
        if not isinstance(conversion_data.get("meta"), dict):
            _add(findings, "error", "schema_error", "site/facts/conversion.json", "top-level 'meta' must be dict")
        if not isinstance(conversion_data.get("engines"), list):
            _add(findings, "error", "schema_error", "site/facts/conversion.json", "top-level 'engines' must be list")
        conv_folders: set[str] = set()
        for idx, entry in enumerate(conversion_data.get("engines") or ()):
            if not isinstance(entry, dict):
                _add(findings, "error", "schema_error", "site/facts/conversion.json", f"engines[{idx}] must be object")
                continue
            name = entry.get("engine")
            folder = entry.get("folder")
            if not isinstance(name, str) or not isinstance(folder, str):
                _add(findings, "error", "schema_error", "site/facts/conversion.json", f"engines[{idx}] missing engine/folder")
                continue
            if not isinstance(entry.get("recorded_facts"), list):
                _add(findings, "error", "schema_error", "site/facts/conversion.json", f"engines[{idx}] ({name}) recorded_facts must be list")
            base = Path(folder).name
            if base in conv_folders:
                _add(findings, "error", "duplicate_folder", "site/facts/conversion.json", f"duplicate folder {folder!r}")
            conv_folders.add(base)
            if not (root / folder).is_dir():
                _add(findings, "error", "missing_repo_folder", "site/facts/conversion.json", f"folder {folder!r} does not exist")
        for folder in repo_folders:
            if folder not in conv_folders:
                _add(
                    findings, "error", "missing_engine_in_conversion", "site/facts/conversion.json",
                    f"repo engine 'repos/{folder}' missing from conversion.json",
                )
    elif conversion_data is not None:
        _add(findings, "error", "schema_error", "site/facts/conversion.json", "root must be JSON object")

    return known_engines


def _check_prs(prs_data, known_engines: dict[str, str], findings: list[dict]) -> set[str]:
    """Validate `prs.json` schema and internal `meta.counts`, returning all PR ID strings."""
    pr_ids: set[str] = set()
    if not isinstance(prs_data, dict):
        if prs_data is not None:
            _add(findings, "error", "schema_error", "site/facts/prs.json", "root must be JSON object")
        return pr_ids
    meta = prs_data.get("meta")
    counts = meta.get("counts") if isinstance(meta, dict) else None
    if not isinstance(counts, dict):
        _add(findings, "error", "schema_error", "site/facts/prs.json", "meta.counts must be object")
        counts = {}
    engines_list = prs_data.get("engines")
    if not isinstance(engines_list, list):
        _add(findings, "error", "schema_error", "site/facts/prs.json", "engines must be list")
        return pr_ids

    by_norm: dict[str, dict] = {}
    for idx, entry in enumerate(engines_list):
        if not isinstance(entry, dict):
            _add(findings, "error", "schema_error", "site/facts/prs.json", f"engines[{idx}] must be object")
            continue
        eng = entry.get("engine")
        if not isinstance(eng, str) or not eng.strip():
            _add(findings, "error", "schema_error", "site/facts/prs.json", f"engines[{idx}].engine missing")
            continue
        if known_engines and eng not in known_engines:
            _add(findings, "error", "unknown_engine_name", "site/facts/prs.json", f"engine {eng!r} not in engines.json")
        by_norm[_norm(eng)] = entry
        prs = entry.get("prs")
        if not isinstance(prs, list):
            _add(findings, "error", "schema_error", "site/facts/prs.json", f"engines[{idx}] ({eng}) prs must be list")
            continue
        seen_eng_prs: set[str] = set()
        for p_idx, item in enumerate(prs):
            if not isinstance(item, dict) or item.get("pr") is None:
                _add(findings, "error", "schema_error", "site/facts/prs.json", f"{eng}.prs[{p_idx}] missing pr id")
                continue
            pid = str(item["pr"])
            if pid in seen_eng_prs:
                _add(findings, "error", "duplicate_pr_id", "site/facts/prs.json", f"duplicate PR {pid!r} in {eng}")
            seen_eng_prs.add(pid)
            pr_ids.add(pid)
            if not isinstance(item.get("installed_live"), bool):
                _add(findings, "error", "schema_error", "site/facts/prs.json", f"{eng} PR {pid}: installed_live must be bool")

    count_checks = (
        ("vllm_named_with_result", "vllm", "prs"),
        ("megatron_replayed_with_result", "megatron", "prs"),
        ("fastvideo_replayed_with_result", "fastvideo", "prs"),
        ("vllm_tagged_not_replayed", "vllm", "tagged_not_replayed"),
    )
    for count_key, eng_key, field in count_checks:
        if count_key in counts and eng_key in by_norm:
            actual = len(by_norm[eng_key].get(field) or ())
            if counts[count_key] != actual:
                _add(
                    findings, "error", "prs_count_mismatch", "site/facts/prs.json",
                    f"meta.counts[{count_key!r}]={counts[count_key]} != len({eng_key}.{field})={actual}",
                )
    return pr_ids


def _check_case_copy_and_timing(
    copy_data,
    timing_data,
    known_engines: dict[str, str],
    prs_ids: set[str],
    findings: list[dict],
) -> None:
    """Validate `case-copy.json`, `case-timing.json`, arithmetic invariants, and ID joins."""
    copy_ids: set[str] = set()
    if isinstance(copy_data, dict):
        blocks = copy_data.get("blocks")
        cases = copy_data.get("cases")
        block_keys: list[str] = []
        if not isinstance(blocks, list) or not blocks:
            _add(findings, "error", "schema_error", "site/facts/case-copy.json", "blocks must be non-empty list")
        else:
            for b in blocks:
                if isinstance(b, dict) and isinstance(b.get("key"), str):
                    block_keys.append(b["key"])
                else:
                    _add(findings, "error", "schema_error", "site/facts/case-copy.json", f"invalid block entry: {b!r}")
        if not isinstance(cases, list) or not cases:
            _add(findings, "error", "schema_error", "site/facts/case-copy.json", "cases must be non-empty list")
        else:
            for idx, item in enumerate(cases):
                if not isinstance(item, dict) or item.get("pr") is None:
                    _add(findings, "error", "schema_error", "site/facts/case-copy.json", f"cases[{idx}] missing 'pr'")
                    continue
                cid = str(item["pr"])
                if cid in copy_ids:
                    _add(findings, "error", "duplicate_case_copy_id", "site/facts/case-copy.json", f"duplicate case id {cid!r}")
                copy_ids.add(cid)
                for key in block_keys:
                    val = item.get(key)
                    if not isinstance(val, str) or not val.strip():
                        _add(
                            findings, "error", "case_copy_missing_block", "site/facts/case-copy.json",
                            f"case {cid!r} missing non-empty block {key!r}",
                        )
    elif copy_data is not None:
        _add(findings, "error", "schema_error", "site/facts/case-copy.json", "root must be JSON object")

    timing_ids: set[str] = set()
    timing_upstream_pr_ids: set[str] = set()
    if isinstance(timing_data, dict):
        cases = timing_data.get("cases")
        if not isinstance(cases, list) or not cases:
            _add(findings, "error", "schema_error", "site/facts/case-timing.json", "cases must be non-empty list")
        else:
            seen_keys: set[tuple[str, str, str]] = set()
            for idx, item in enumerate(cases):
                if not isinstance(item, dict):
                    _add(findings, "error", "schema_error", "site/facts/case-timing.json", f"cases[{idx}] must be object")
                    continue
                eng = item.get("engine")
                cid_raw = item.get("id")
                var = item.get("variant", "")
                src = item.get("source")
                if not isinstance(eng, str) or cid_raw is None or not isinstance(var, str):
                    _add(findings, "error", "schema_error", "site/facts/case-timing.json", f"cases[{idx}] missing engine/id/variant")
                    continue
                cid = str(cid_raw)
                if known_engines and eng not in known_engines:
                    _add(findings, "error", "unknown_engine_name", "site/facts/case-timing.json", f"case {cid!r}: unknown engine {eng!r}")
                key = (eng, cid, var)
                if key in seen_keys:
                    _add(findings, "error", "duplicate_case_timing_key", "site/facts/case-timing.json", f"duplicate key {key!r}")
                seen_keys.add(key)
                timing_ids.add(cid)
                if src == "upstream_pr":
                    timing_upstream_pr_ids.add(cid)
                for arm_name in ("restart", "rsibox"):
                    arm = item.get(arm_name)
                    if not isinstance(arm, dict):
                        _add(findings, "error", "schema_error", "site/facts/case-timing.json", f"{eng} {cid}: missing {arm_name} arm")
                        continue
                    total = arm.get("total")
                    phases = arm.get("phases")
                    if not isinstance(total, (int, float)) or total < 0:
                        _add(findings, "error", "invalid_timing", "site/facts/case-timing.json", f"{eng} {cid} {arm_name}: invalid total {total!r}")
                        continue
                    if not isinstance(phases, list) or not phases:
                        _add(findings, "error", "schema_error", "site/facts/case-timing.json", f"{eng} {cid} {arm_name}: phases must be non-empty list")
                        continue
                    phase_sum = 0.0
                    bad_phase = False
                    for pair in phases:
                        if (
                            not isinstance(pair, list)
                            or len(pair) != 2
                            or not isinstance(pair[0], str)
                            or not isinstance(pair[1], (int, float))
                            or pair[1] < 0
                        ):
                            _add(findings, "error", "invalid_timing", "site/facts/case-timing.json", f"{eng} {cid} {arm_name}: bad phase {pair!r}")
                            bad_phase = True
                            break
                        phase_sum += float(pair[1])
                    if not bad_phase and abs(phase_sum - float(total)) > 0.05:
                        _add(
                            findings, "error", "timing_phase_sum_mismatch", "site/facts/case-timing.json",
                            f"{eng} {cid} ({var or 'default'}) {arm_name}: sum(phases)={phase_sum:.2f} != total={total}",
                        )
                improved = item.get("improved")
                if improved is not None:
                    if not isinstance(improved, dict) or not isinstance(improved.get("total"), (int, float)) or improved["total"] < 0:
                        _add(findings, "error", "invalid_timing", "site/facts/case-timing.json", f"{eng} {cid}: invalid improved arm")
                for ef_idx, ef in enumerate(item.get("effect") or ()):
                    if not isinstance(ef, dict):
                        continue
                    b, e, r = ef.get("base_step_s"), ef.get("edit_step_s"), ef.get("ratio")
                    if isinstance(b, (int, float)) and isinstance(e, (int, float)) and isinstance(r, (int, float)) and b > 0:
                        expected = e / b
                        if abs(expected - r) > 0.005:
                            _add(
                                findings, "error", "effect_ratio_mismatch", "site/facts/case-timing.json",
                                f"{eng} {cid} effect[{ef_idx}]: edit/base={expected:.4f} != ratio={r}",
                            )
    elif timing_data is not None:
        _add(findings, "error", "schema_error", "site/facts/case-timing.json", "root must be JSON object")

    if copy_ids and timing_ids:
        for missing in sorted(copy_ids - timing_ids):
            _add(
                findings, "error", "case_copy_timing_id_mismatch", "site/facts/case-copy.json",
                f"case id {missing!r} present in case-copy.json but missing from case-timing.json",
            )
        for missing in sorted(timing_ids - copy_ids):
            _add(
                findings, "error", "case_copy_timing_id_mismatch", "site/facts/case-timing.json",
                f"case id {missing!r} present in case-timing.json but missing from case-copy.json",
            )

    if prs_ids and timing_upstream_pr_ids:
        for missing in sorted(prs_ids - timing_upstream_pr_ids):
            _add(
                findings, "error", "prs_timing_id_mismatch", "site/facts/prs.json",
                f"PR {missing!r} present in prs.json but missing upstream_pr row in case-timing.json",
            )
        for missing in sorted(timing_upstream_pr_ids - prs_ids):
            _add(
                findings, "error", "prs_timing_id_mismatch", "site/facts/case-timing.json",
                f"upstream_pr {missing!r} in case-timing.json missing from prs.json",
            )


def _check_coverage(
    root: Path,
    repo_folders: list[str],
    known_engines: dict[str, str],
    coverage_data,
    findings: list[dict],
) -> None:
    """Validate `coverage.json` schema and flag split-brain engine/conformance drift."""
    if not isinstance(coverage_data, dict):
        if coverage_data is not None:
            _add(findings, "error", "schema_error", "site/facts/coverage.json", "root must be JSON object")
        return
    measured = coverage_data.get("measured")
    not_covered = coverage_data.get("not_covered_stated_by_repo")
    if not isinstance(measured, list) or not measured:
        _add(findings, "error", "schema_error", "site/facts/coverage.json", "measured must be non-empty list")
        measured = []
    if not isinstance(not_covered, dict):
        _add(findings, "error", "schema_error", "site/facts/coverage.json", "not_covered_stated_by_repo must be object")
        not_covered = {}

    folder_to_name = {folder: name for name, folder in known_engines.items()}
    seen_ids: set[str] = set()
    conformance_row = None
    for idx, row in enumerate(measured):
        if not isinstance(row, dict):
            _add(findings, "error", "schema_error", "site/facts/coverage.json", f"measured[{idx}] must be object")
            continue
        rid = row.get("id")
        eng = row.get("engine")
        if not isinstance(rid, str) or not isinstance(eng, str):
            _add(findings, "error", "schema_error", "site/facts/coverage.json", f"measured[{idx}] missing id/engine")
            continue
        if rid in seen_ids:
            _add(findings, "error", "duplicate_coverage_id", "site/facts/coverage.json", f"duplicate measured id {rid!r}")
        seen_ids.add(rid)
        if eng != "all" and known_engines and eng not in known_engines:
            _add(findings, "error", "unknown_engine_name", "site/facts/coverage.json", f"measured[{rid!r}]: unknown engine {eng!r}")
        if rid == "conformance":
            conformance_row = row

    nc_norms: set[str] = set()
    for key in not_covered:
        if key == "source":
            continue
        if known_engines and key not in known_engines:
            _add(findings, "error", "unknown_engine_name", "site/facts/coverage.json", f"not_covered_stated_by_repo: unknown engine {key!r}")
        nc_norms.add(_norm(key))
    for folder in repo_folders:
        if _norm(folder) not in nc_norms:
            label = folder_to_name.get(folder, folder)
            _add(
                findings, "drift", "coverage_not_covered_missing_engine", "site/facts/coverage.json",
                f"engine {label!r} (repos/{folder}) missing from not_covered_stated_by_repo",
            )

    if isinstance(conformance_row, dict) and isinstance(conformance_row.get("results"), dict):
        results = conformance_row["results"]
        res_by_norm: dict[str, list[tuple[str, object]]] = {}
        for k, v in results.items():
            prefix = k.split("(", 1)[0].strip()
            res_by_norm.setdefault(_norm(prefix), []).append((k, v))
        for folder in repo_folders:
            label = folder_to_name.get(folder, folder)
            entries = res_by_norm.get(_norm(folder))
            has_real = _has_real_conformance_lab(root / "repos" / folder)
            if not entries:
                _add(
                    findings, "drift", "coverage_conformance_missing_engine", "site/facts/coverage.json",
                    f"engine {label!r} (repos/{folder}) omitted from coverage.json conformance results",
                )
            elif has_real and all(val is None for _, val in entries):
                _add(
                    findings, "drift", "coverage_conformance_null_with_adapter", "site/facts/coverage.json",
                    f"coverage.json conformance lists {label!r} as null, but a real conformance Lab class exists under repos/{folder}/livelab",
                )


def _check_readme(
    root: Path,
    repo_folders: list[str],
    known_engines: dict[str, str],
    findings: list[dict],
) -> None:
    """Parse engine table & repo layout in `docs/engines.md` (or `README.md`) and flag split-brain drift."""
    engines_doc = root / "docs" / "engines.md"
    if engines_doc.is_file():
        doc_path, doc_label = engines_doc, "docs/engines.md"
    else:
        doc_path, doc_label = root / "README.md", "README.md"
    if not doc_path.is_file():
        _add(findings, "error", "missing_file", doc_label, f"required file missing: {doc_path}")
        return
    text = doc_path.read_text(encoding="utf-8")
    folder_to_name = {folder: name for name, folder in known_engines.items()}

    # 1. "What is in the repo today" engine table
    m_table = re.search(r"## What is in the repo today\s*\n+((?:\|.*\n)+)", text)
    if not m_table:
        _add(findings, "error", "readme_missing_engine_table", doc_label, f"could not locate '## What is in the repo today' table in {doc_label}")
    else:
        rows = [
            ln for ln in m_table.group(1).splitlines()
            if ln.startswith("|") and not re.match(r"^\|[\s:|-]+\|$", ln)
        ]
        table_engines: dict[str, str] = {}
        for row in rows[1:]:
            cols = [c.strip() for c in row.strip().strip("|").split("|")]
            if not cols:
                continue
            cell = cols[0]
            norm = _norm(cell)
            if norm in ("toylab", "toy"):
                continue
            table_engines[norm] = row
        for folder in repo_folders:
            label = folder_to_name.get(folder, folder)
            row = table_engines.get(_norm(folder))
            if row is None:
                _add(
                    findings, "drift", "readme_table_missing_engine", doc_label,
                    f"engine {label!r} (repos/{folder}) missing from 'What is in the repo today' table",
                )
            elif _has_real_conformance_lab(root / "repos" / folder) and re.search(
                r"no conformance adapter yet", row, re.IGNORECASE
            ):
                _add(
                    findings, "drift", "readme_stale_conformance_claim", doc_label,
                    f"{doc_label} claims {label!r} has 'no conformance adapter yet', but a real conformance Lab class exists under repos/{folder}/livelab",
                )

    # 2. "Repository layout" repos/<engine>/ listing
    m_layout = re.search(r"repos/<engine>/\s+([^\n]+)", text)
    if m_layout:
        listed = {_norm(tok) for tok in re.split(r"[,\s]+", m_layout.group(1).strip()) if tok.strip()}
        for folder in repo_folders:
            if _norm(folder) not in listed:
                _add(
                    findings, "drift", "readme_layout_missing_engine", doc_label,
                    f"engine folder {folder!r} omitted from 'Repository layout' repos/<engine>/ list",
                )


def verify_repo(repo_root: str | os.PathLike) -> dict:
    """Run all schema, arithmetic, join, and drift checks against `repo_root` (read-only)."""
    root = Path(repo_root).resolve()
    facts_dir = root / "site" / "facts"
    repos_dir = root / "repos"
    repo_folders = sorted(
        p.name for p in repos_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"
    ) if repos_dir.is_dir() else []

    findings: list[dict] = []
    loaded = {name: _load_json(facts_dir / name, findings) for name in REQUIRED_FACT_FILES}

    known_engines = _check_engines_and_conversion(
        root, repo_folders, loaded["engines.json"], loaded["conversion.json"], findings
    )
    prs_ids = _check_prs(loaded["prs.json"], known_engines, findings)
    _check_case_copy_and_timing(
        loaded["case-copy.json"], loaded["case-timing.json"], known_engines, prs_ids, findings
    )
    _check_coverage(root, repo_folders, known_engines, loaded["coverage.json"], findings)
    _check_readme(root, repo_folders, known_engines, findings)

    errors = [f for f in findings if f["severity"] == "error"]
    drifts = [f for f in findings if f["severity"] == "drift"]
    return {
        "ok": not errors and not drifts,
        "schema_ok": not errors,
        "repo_root": str(root),
        "engines": repo_folders,
        "summary": {
            "files_checked": len(REQUIRED_FACT_FILES) + 1,
            "errors": len(errors),
            "drifts": len(drifts),
        },
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate site/facts/*.json and README.md engine tables (read-only).",
    )
    parser.add_argument("--repo-root", default=str(default_root), help="path to repository root")
    parser.add_argument("--json", action="store_true", help="print structured JSON report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 when split-brain doc/coverage drifts are detected (in addition to exit 1 on schema/math errors)",
    )
    args = parser.parse_args(argv)
    report = verify_repo(args.repo_root)
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        s = report["summary"]
        print(
            f"verify_facts: checked {s['files_checked']} files across {len(report['engines'])} repo engines — "
            f"{s['errors']} error(s), {s['drifts']} split-brain drift(s)"
        )
        for item in report["findings"]:
            print(f"  [{item['severity'].upper():<5}] {item['file']} ({item['code']}): {item['message']}")
    if report["summary"]["errors"] > 0:
        return 1
    if args.strict and report["summary"]["drifts"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
