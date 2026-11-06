"""Archive Gold-free runs and connect the real GigaChat and Java components."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from benchmark import SCHEMA_VERSION
from benchmark.contracts import ModelSettings, VerifierSettings
from benchmark.feedback import EXPLANATION_PROMPT, EXPLANATION_VERSION, FILTER_VERSION
from benchmark.gigachat import GigaChatClient
from benchmark.runner import (
    EXTRACTION_VERSION,
    INITIAL_PROMPT,
    REPAIR_PROMPT,
    SYSTEM_PROMPT,
    Runner,
)
from benchmark.task_dataset import DEFAULT_SELECTION, ROOT, load_inputs, read_selection
from benchmark.validation import PROFILE
from benchmark.verifier import JavaVerifier


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _version(command):
    result = subprocess.run(command, capture_output=True, timeout=30, check=True)
    return (result.stdout + result.stderr).decode(errors="replace").strip()


def snapshot(config, selection_path, tasks, jar, phase, executed_ids, max_attempts):
    """Archive only allowlisted inputs and code; never traverse the dataset tree."""
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
    if dirty:
        raise ValueError("Commit the tested code before running a recorded pilot or main series")
    selection = read_selection(selection_path)
    jar_data = jar.read_bytes()
    if digest(jar_data) != selection["jar_sha256"]:
        raise ValueError("JAR differs from the pinned task set")
    files = {"tools/tla2tools.jar": jar_data, "tasks/selection.json": selection_path.read_bytes()}
    sources = {}
    for path in sorted((ROOT / "src/benchmark").glob("*.py")):
        name = "code/" + path.name
        files[name] = path.read_bytes()
        sources[name] = digest(files[name])
    prompts = {}
    for name, text in (
        ("system", SYSTEM_PROMPT),
        ("initial", INITIAL_PROMPT),
        ("repair", REPAIR_PROMPT),
        ("explanation", EXPLANATION_PROMPT),
    ):
        relative = f"prompts/{name}.txt"
        files[relative] = text.encode()
        prompts[name] = {"path": relative, "sha256": digest(files[relative])}
    inputs = {}
    for task in tasks:
        inputs[task.task_id] = {}
        for name, text, suffix in (
            ("description", task.description, "description.md"),
            ("cfg", task.cfg_text, "Spec.cfg"),
        ):
            relative = f"tasks/{task.task_id}/{suffix}"
            files[relative] = text.encode()
            inputs[task.task_id][name + "_path"] = relative
    stamp = datetime.now(UTC)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "validation_profile": PROFILE if phase == "main" else "runner-pilot-v1",
        "experiment_id": f"{phase}-{stamp.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}",
        "phase": phase,
        "created_at_utc": stamp.isoformat(),
        "code": {
            "revision": revision,
            "dirty": False,
            "runner_path": "code/runner.py",
            "source_files": sources,
        },
        "model": config["model"],
        "environment": {
            "python_version": sys.version,
            "java_version": _version(["java", "-version"]),
            "tools_version": _version(["java", "-cp", str(jar), "tlc2.TLC", "-help"]),
        },
        "verifier": {
            "jar_path": "tools/tla2tools.jar",
            "jar_sha256": selection["jar_sha256"],
            "java_executable": "java",
            **config["verifier"],
        },
        "protocol": {
            "methods": ["A", "B", "C"],
            "max_attempts": max_attempts,
            "initial_attempt_number": 1,
            "shared_initial": True,
            "isolated_histories": True,
            "gold_in_prompts": False,
            "repetitions": config["repetitions"],
            "explanation_context": "fresh_specification_and_log_only",
            "explanation_model": "same_as_main",
            "explanation_counts_as_main_attempt": False,
            "code_response_policy": "forward_verbatim_and_flag",
            "empty_response_policy": "forward_verbatim_and_flag",
            "source_extraction_policy": EXTRACTION_VERSION,
            "api_failure_policy": "abort_run_no_retries_preserve_completed_and_failed_calls",
            "verification_failure_policy": "stop_affected_branch_on_tool_error_or_timeout",
            "context_overflow_policy": "abort_on_provider_error_never_truncate_or_reset_history",
            "filter_version": FILTER_VERSION,
            "explanation_version": EXPLANATION_VERSION,
        },
        "prompts": prompts,
        "tasks": {
            "selection_path": "tasks/selection.json",
            "selection_sha256": digest(files["tasks/selection.json"]),
            "inputs": inputs,
        },
        "executed_task_ids": executed_ids,
        "api_timeout_seconds": config["api_timeout_seconds"],
    }
    return manifest, files


def _check_pilot(root, manifest):
    previous = json.loads((root / "manifest.json").read_text())
    completed = json.loads((root / "complete.json").read_text())
    if previous["phase"] != "pilot" or not completed["branches"]:
        raise ValueError("A completed real pilot is required before the main series")
    for key in ("model", "prompts", "verifier", "tasks", "environment"):
        if previous[key] != manifest[key]:
            raise ValueError(f"Pilot/main settings differ: {key}; repeat the pilot after changes")
    if previous["code"]["source_files"] != manifest["code"]["source_files"]:
        raise ValueError("Code differs from the completed pilot; repeat the pilot")
    if any(b["status"] == "verification_interruption" for b in completed["branches"]):
        raise ValueError("Resolve technical verifier interruptions before the main series")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/tasks")
    parser.add_argument("--jar", type=Path, default=ROOT / "data/lora-pilot/tla2tools.jar")
    parser.add_argument("--phase", choices=["pilot", "main"], default="pilot")
    parser.add_argument("--task", action="append", help="Pilot subset; repeat for multiple IDs")
    parser.add_argument(
        "--max-attempts", type=int, help="Pilot limit, including initial generation"
    )
    parser.add_argument("--pilot", type=Path, help="Completed, compatible pilot required for main")
    parser.add_argument("--output", type=Path, help="New directory; never overwritten")
    parser.add_argument(
        "--credentials-file",
        type=Path,
        default=Path.home() / ".config/tla-repair/gigachat-authorization.key",
    )
    parser.add_argument(
        "--ca-bundle", type=Path, default=Path.home() / ".config/tla-repair/gigachat-ca.pem"
    )
    args = parser.parse_args(argv)
    output = None
    try:
        if args.phase == "main" and (args.task or args.pilot is None):
            raise ValueError("Main requires all four tasks and --pilot")
        config = json.loads(args.config.read_text())
        selection = read_selection(args.selection)
        tasks = load_inputs(selection, args.dataset)
        ids = args.task or [t.task_id for t in tasks]
        if len(set(ids)) != len(ids) or set(ids) - {t.task_id for t in tasks}:
            raise ValueError("Unknown or repeated pilot task IDs")
        limit = args.max_attempts if args.max_attempts is not None else config["max_attempts"]
        if args.phase == "main" and (args.task or limit != 10 or args.pilot is None):
            raise ValueError("Main requires all four tasks, ten attempts and --pilot")
        manifest, files = snapshot(
            config, args.selection, tasks, args.jar.resolve(), args.phase, ids, limit
        )
        if args.phase == "main":
            _check_pilot(args.pilot.resolve(), manifest)
            manifest["pilot_evidence"] = {
                "path": str(args.pilot.resolve()),
                "complete_sha256": digest((args.pilot / "complete.json").read_bytes()),
            }
        output = (
            args.output.resolve() if args.output else ROOT / "results" / manifest["experiment_id"]
        )
        credentials = (
            os.environ.get("GIGACHAT_CREDENTIALS") or args.credentials_file.read_text().strip()
        )
        client = GigaChatClient(
            output,
            credentials=credentials,
            ca_bundle_file=args.ca_bundle,
            timeout_seconds=config["api_timeout_seconds"],
            expected_model=config["model"]["resolved_id"],
        )
        settings = ModelSettings(
            config["model"]["provider"],
            config["model"]["requested_id"],
            config["model"]["parameters"],
        )
        checker = VerifierSettings(
            args.jar.resolve(),
            selection["jar_sha256"],
            "java",
            config["verifier"]["sany_timeout_seconds"],
            config["verifier"]["tlc_timeout_seconds"],
            tuple(config["verifier"]["tlc_arguments"]),
        )
        runner = Runner(
            output,
            client,
            JavaVerifier(),
            settings,
            checker,
            max_attempts=limit,
            repetitions=config["repetitions"],
        )
        print(f"Experiment directory: {output}", flush=True)
        metrics = runner.run([t for t in tasks if t.task_id in ids], manifest, snapshot_files=files)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "branches": len(metrics["branches"]),
                    "calls": metrics["completed_calls_total"]["calls"],
                }
            ),
            flush=True,
        )
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Experiment stopped ({type(exc).__name__}). Evidence: {output}\n{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
