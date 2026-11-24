#!/usr/bin/env python3
"""Strict, fail-closed parser for `/run-gpu-ci` PR comments and workflow_dispatch inputs.

Stdlib only (does not import `modal`). Reads the authoritative suite allowlist
(`ALL_SUITE_ORDER`) directly from `.github/modal-ci/modal_app.py` via AST parsing
and validates all flags against strict character allowlists with zero shell
interpolation.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
MODAL_APP_PATH = HERE / "modal_app.py"

COMMAND_PREFIX = "/run-gpu-ci"
DEFAULT_GPU = "L4"

ALLOWED_LINE_PATTERN = re.compile(r"^[A-Za-z0-9:/._,= \t-]+$")
GPU_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:_-]{0,31}$")
VLLM_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
PR_NUMBER_PATTERN = re.compile(r"^[1-9][0-9]{0,9}$")

SUPPORTED_FLAGS = frozenset({"--gpu", "--suites", "--vllm-model"})


@dataclass(frozen=True)
class ParsedCommand:
    """Validated Modal GPU CI trigger parameters."""

    gpu: str = DEFAULT_GPU
    suites: str = ""
    vllm_model: str = ""
    pr: str = ""


def load_allowed_suites(modal_app_path: Path = MODAL_APP_PATH) -> frozenset[str]:
    """Read `ALL_SUITE_ORDER` from `modal_app.py` without importing `modal`."""
    source_text = modal_app_path.read_text(encoding="utf-8")
    syntax_tree = ast.parse(source_text, filename=str(modal_app_path))
    for statement in syntax_tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if isinstance(target, ast.Name) and target.id == "ALL_SUITE_ORDER":
                suite_names = ast.literal_eval(statement.value)
                if (
                    not isinstance(suite_names, (tuple, list))
                    or not suite_names
                    or not all(isinstance(name, str) and name for name in suite_names)
                ):
                    raise ValueError("ALL_SUITE_ORDER in modal_app.py must be a non-empty sequence of strings")
                return frozenset(suite_names)
    raise ValueError(f"ALL_SUITE_ORDER not found in {modal_app_path}")


def validate_options(
    *,
    gpu: str = DEFAULT_GPU,
    suites: str = "",
    vllm_model: str = "",
    pr: str = "",
    allowed_suites: frozenset[str] | None = None,
) -> ParsedCommand:
    """Validate GPU, suite CSV, vLLM model ID, and optional PR number fail-closed."""
    gpu_value = gpu.strip() if gpu else DEFAULT_GPU
    if not GPU_PATTERN.fullmatch(gpu_value):
        raise ValueError(
            f"invalid --gpu value {gpu_value!r}: must match ^[A-Za-z0-9][A-Za-z0-9:_-]{{0,31}}$"
        )

    suites_value = suites.strip() if suites else ""
    normalized_suites = ""
    if suites_value:
        suite_allowlist = load_allowed_suites() if allowed_suites is None else allowed_suites
        raw_entries = suites_value.split(",")
        deduped_suites: list[str] = []
        unknown_suites: list[str] = []
        for raw_entry in raw_entries:
            suite_name = raw_entry.strip()
            if not suite_name:
                raise ValueError(f"invalid --suites value {suites_value!r}: empty suite entry")
            if suite_name not in suite_allowlist:
                unknown_suites.append(suite_name)
            elif suite_name not in deduped_suites:
                deduped_suites.append(suite_name)
        if unknown_suites:
            unknown_sorted = ", ".join(sorted(set(unknown_suites)))
            raise ValueError(f"unknown suite(s) in --suites: {unknown_sorted}")
        normalized_suites = ",".join(deduped_suites)

    model_value = vllm_model.strip() if vllm_model else ""
    if model_value:
        if not VLLM_MODEL_PATTERN.fullmatch(model_value) or ".." in model_value:
            raise ValueError(
                f"invalid --vllm-model value {model_value!r}: must match ^[A-Za-z0-9][A-Za-z0-9._/-]{{0,127}}$ without '..'"
            )

    pr_value = pr.strip() if pr else ""
    if pr_value and not PR_NUMBER_PATTERN.fullmatch(pr_value):
        raise ValueError(f"invalid pr value {pr_value!r}: must be a positive integer")

    return ParsedCommand(
        gpu=gpu_value,
        suites=normalized_suites,
        vllm_model=model_value,
        pr=pr_value,
    )


def parse_comment_command(
    comment_body: str,
    allowed_suites: frozenset[str] | None = None,
) -> ParsedCommand:
    """Parse a `/run-gpu-ci` PR comment body strictly; only the first line is evaluated."""
    if not isinstance(comment_body, str) or not comment_body.strip():
        raise ValueError("empty comment body")

    lines = comment_body.splitlines()
    command_line = lines[0].strip() if lines else ""
    if not command_line:
        raise ValueError("comment does not start with /run-gpu-ci")

    if not ALLOWED_LINE_PATTERN.fullmatch(command_line):
        raise ValueError("disallowed characters in /run-gpu-ci command line")

    tokens = command_line.split()
    if not tokens or tokens[0] != COMMAND_PREFIX:
        raise ValueError(f"comment must start with exact command {COMMAND_PREFIX!r}")

    parsed_flags: dict[str, str] = {}
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if "=" in token:
            flag_name, flag_value = token.split("=", 1)
            if flag_name not in SUPPORTED_FLAGS:
                raise ValueError(f"unsupported flag {flag_name!r}")
            if not flag_value:
                raise ValueError(f"missing value for {flag_name}")
            index += 1
        elif token in SUPPORTED_FLAGS:
            flag_name = token
            if index + 1 >= len(tokens):
                raise ValueError(f"missing value for {flag_name}")
            flag_value = tokens[index + 1]
            index += 2
        else:
            raise ValueError(f"unexpected argument {token!r}")

        if flag_name in parsed_flags:
            raise ValueError(f"duplicate flag {flag_name}")
        parsed_flags[flag_name] = flag_value

    return validate_options(
        gpu=parsed_flags.get("--gpu", DEFAULT_GPU),
        suites=parsed_flags.get("--suites", ""),
        vllm_model=parsed_flags.get("--vllm-model", ""),
        allowed_suites=allowed_suites,
    )


def write_github_output(output_path: Path, parsed_command: ParsedCommand) -> None:
    """Append validated single-line key=value pairs to a GitHub Actions output file."""
    lines = [
        f"gpu={parsed_command.gpu}",
        f"suites={parsed_command.suites}",
        f"vllm_model={parsed_command.vllm_model}",
        f"pr={parsed_command.pr}",
    ]
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--github-output",
        default="",
        help="path to GITHUB_OUTPUT file to append validated outputs",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print validated parameters as JSON to stdout",
    )
    args = parser.parse_args(argv)

    event_name = os.environ.get("EVENT_NAME", "issue_comment").strip()
    try:
        if event_name == "issue_comment":
            parsed_command = parse_comment_command(os.environ.get("COMMENT_BODY", ""))
        elif event_name == "workflow_dispatch":
            parsed_command = validate_options(
                gpu=os.environ.get("INPUT_GPU", "") or DEFAULT_GPU,
                suites=os.environ.get("INPUT_SUITES", ""),
                vllm_model=os.environ.get("INPUT_VLLM_MODEL", ""),
                pr=os.environ.get("INPUT_PR", ""),
            )
        else:
            raise ValueError(f"unsupported EVENT_NAME {event_name!r}")
    except ValueError as error:
        print(f"::error::Invalid Modal GPU CI trigger arguments: {error}", file=sys.stderr)
        return 1

    if args.github_output:
        write_github_output(Path(args.github_output), parsed_command)
    if args.json:
        print(
            json.dumps(
                {
                    "gpu": parsed_command.gpu,
                    "suites": parsed_command.suites,
                    "vllm_model": parsed_command.vllm_model,
                    "pr": parsed_command.pr,
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
