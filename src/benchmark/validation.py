"""Validate a recorded protocol snapshot; does not certify runner behavior."""

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

from benchmark import SCHEMA_VERSION
from benchmark.feedback import EXPLANATION_PROMPT, EXPLANATION_VERSION, FILTER_VERSION
from benchmark.llm import _write_json
from benchmark.task_dataset import read_selection

PROFILE = "benchmark-snapshot-v1"


def validate_snapshot(root: Path) -> dict:
    """Check archived settings and files, without Java, API, or Gold access.

    This is an additive manifest profile for review with the runner owner.
    A valid snapshot alone is not evidence of isolation or a successful pilot.
    """
    root = Path(root).resolve(strict=True)
    errors, hashes = [], {}

    def fail(message):
        errors.append(message)

    def text(value):
        return isinstance(value, str) and bool(value.strip())

    def number(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0

    def file_hash(relative, expected):
        if not isinstance(relative, str) or not relative:
            fail("Missing snapshot artifact path")
            return None
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or any(part.lower() in {"reference", "gold-check"} for part in path.parts)
        ):
            fail(f"Forbidden snapshot path: {relative}")
            return None
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            fail(f"Missing or external snapshot file: {relative}")
            return None
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if relative in hashes and hashes[relative] != digest:
            fail(f"Snapshot changed during validation: {relative}")
        hashes[relative] = digest
        if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            fail(f"Missing or invalid SHA-256: {relative}")
        elif digest != expected:
            fail(f"SHA-256 mismatch: {relative}")
        return resolved

    def mapping(value, name):
        if not isinstance(value, dict):
            fail(f"Missing or invalid {name}")
            return {}
        return value

    try:
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("Manifest must be a regular snapshot file")
        data = manifest_path.read_bytes()
        hashes["manifest.json"] = hashlib.sha256(data).hexdigest()
        manifest = json.loads(data)
    except (OSError, ValueError) as exc:
        return {
            "profile": PROFILE,
            "snapshot_valid": False,
            "main_readiness": "not_assessed",
            "errors": [str(exc)],
            "source_sha256": hashes,
        }
    manifest = mapping(manifest, "manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        fail("Unsupported schema_version")
    if manifest.get("validation_profile") != PROFILE:
        fail("Missing or unsupported validation_profile")
    if manifest.get("phase") not in {"pilot", "main"}:
        fail("phase must be pilot or main")
    if not text(manifest.get("experiment_id")):
        fail("Missing experiment_id")
    try:
        created = datetime.fromisoformat(manifest.get("created_at_utc", ""))
        if created.utcoffset() != timedelta(0):
            raise ValueError("Expected UTC timestamp")
    except (ValueError, TypeError):
        fail("Missing or invalid created_at_utc")
    code = mapping(manifest.get("code"), "code")
    if not isinstance(code.get("revision"), str) or not re.fullmatch(
        r"[a-f0-9]{40}", code["revision"]
    ):
        fail("Code revision must be a full commit SHA")
    if code.get("dirty") is not False:
        fail("Freeze a clean code revision before experiments")
    sources = mapping(code.get("source_files"), "code.source_files")
    if not sources:
        fail("Code source snapshot is empty")
    if not isinstance(code.get("runner_path"), str) or code["runner_path"] not in sources:
        fail("Runner implementation must be included in code.source_files")
    for path, digest in sources.items():
        file_hash(path, digest)

    model = mapping(manifest.get("model"), "model")
    for field in ("provider", "requested_id", "resolved_id"):
        if not text(model.get(field)):
            fail(f"Missing model.{field}")
    parameters = mapping(model.get("parameters"), "model.parameters")
    if not parameters:
        fail("Explicit generation parameters are required")
    try:
        json.dumps(parameters, allow_nan=False)
    except (ValueError, TypeError):
        fail("Model parameters must be finite JSON values")
    environment = mapping(manifest.get("environment"), "environment")
    for field in ("python_version", "java_version", "tools_version"):
        if not text(environment.get(field)):
            fail(f"Missing environment.{field}")
    verifier = mapping(manifest.get("verifier"), "verifier")
    file_hash(verifier.get("jar_path"), verifier.get("jar_sha256"))
    if not text(verifier.get("java_executable")):
        fail("Missing verifier.java_executable")
    for field in ("sany_timeout_seconds", "tlc_timeout_seconds"):
        if not number(verifier.get(field)):
            fail(f"Invalid verifier.{field}")
    arguments = verifier.get("tlc_arguments")
    if (
        not isinstance(arguments, list)
        or len(arguments) % 2
        or any(not isinstance(arg, str) for arg in arguments)
    ):
        fail("Explicit TLC arguments must be a list of pairs")
    else:
        for option, value in zip(arguments[::2], arguments[1::2]):
            if (
                option not in {"-workers", "-seed"}
                or not value.isdigit()
                or (option == "-workers" and int(value) < 1)
            ):
                fail("Unsupported TLC option in frozen protocol")

    protocol = mapping(manifest.get("protocol"), "protocol")
    fixed = {
        "methods": ["A", "B", "C"],
        "max_attempts": 10,
        "initial_attempt_number": 1,
        "shared_initial": True,
        "isolated_histories": True,
        "gold_in_prompts": False,
        "explanation_context": "fresh_specification_and_log_only",
        "explanation_model": "same_as_main",
        "explanation_counts_as_main_attempt": False,
        "code_response_policy": "forward_verbatim_and_flag",
        "empty_response_policy": "forward_verbatim_and_flag",
    }
    for key, expected in fixed.items():
        actual = protocol.get(key)
        if type(actual) is not type(expected) or actual != expected:
            fail(f"Protocol field differs from agreed design: {key}")
    if type(protocol.get("repetitions")) is not int or protocol["repetitions"] not in {1, 3}:
        fail("repetitions must explicitly be 1 (required) or 3 (extension)")
    for key in ("api_failure_policy", "verification_failure_policy", "context_overflow_policy"):
        if not text(protocol.get(key)):
            fail(f"Missing protocol.{key}")
    if protocol.get("filter_version") != FILTER_VERSION:
        fail("Filter version differs from implementation")
    if protocol.get("explanation_version") != EXPLANATION_VERSION:
        fail("Explanation version differs from implementation")

    prompts = mapping(manifest.get("prompts"), "prompts")
    for name in ("system", "initial", "repair", "explanation"):
        entry = mapping(prompts.get(name), f"prompts.{name}")
        path = file_hash(entry.get("path"), entry.get("sha256"))
        if path is not None and name == "explanation":
            if path.read_bytes() != EXPLANATION_PROMPT.encode():
                fail("Explanation prompt differs from implementation")
    tasks = mapping(manifest.get("tasks"), "tasks")
    selection_path = file_hash(tasks.get("selection_path"), tasks.get("selection_sha256"))
    selection = None
    if selection_path is not None:
        try:
            selection = read_selection(selection_path)
        except (ValueError, KeyError, TypeError) as exc:
            fail(f"Invalid task selection: {exc}")
    inputs = mapping(tasks.get("inputs"), "tasks.inputs")
    if selection is not None:
        if verifier.get("jar_sha256") != selection["jar_sha256"]:
            fail("JAR differs from pinned task selection")
        if set(inputs) != {task["task_id"] for task in selection["tasks"]}:
            fail("Task input IDs differ from the four selected tasks")
        for task in selection["tasks"]:
            entry = mapping(inputs.get(task["task_id"]), f"tasks.inputs.{task['task_id']}")
            for kind in ("description", "cfg"):
                file_hash(entry.get(kind + "_path"), task[kind + "_sha256"])

    # A snapshot is evidence, not a lock on a concurrently running producer.
    for relative, digest in list(hashes.items()):
        file_hash(relative, digest)
    return {
        "profile": PROFILE,
        "snapshot_valid": not errors,
        "main_readiness": "not_assessed",
        "errors": errors,
        "source_sha256": dict(sorted(hashes.items())),
        "validator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = validate_snapshot(args.snapshot)
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "snapshot_valid": report["snapshot_valid"],
                "main_readiness": report["main_readiness"],
                "errors": report["errors"],
            }
        )
    )
    return 0 if report["snapshot_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
