"""Reconstruct observed metrics from shared records without reading Gold."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

from benchmark import SCHEMA_VERSION
from benchmark.contracts import (
    AttemptRecord,
    CallRecord,
    CheckStatus,
    Method,
    ToolResult,
    Usage,
    VerificationResult,
)
from benchmark.llm import _write_json


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite nonnegative number")
    return value


def _usage(value):
    for count in value.values():
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("Token counts must be nonnegative integers or null")
    return Usage(**value)


def _totals(records):
    records = list(records)
    result = {"calls": len(records)}
    for field in ("input_tokens", "output_tokens", "duration_seconds"):
        values = [r[field] for r in records]
        result[field] = sum(values) if all(v is not None for v in values) else None
        result[field + "_known_sum"] = sum(v for v in values if v is not None)
        result[field + "_unknown_count"] = sum(v is None for v in values)
    return result


def _call_values(call):
    return {**asdict(call.usage), "duration_seconds": call.duration_seconds}


class _Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.hashes = {}

    def path(self, relative):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Artifact paths must stay within the experiment")
        resolved = (self.root / path).resolve(strict=True)
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise ValueError("Artifact paths must stay within the experiment")
        return resolved

    def read(self, relative, *, json_value=True):
        path = self.path(relative)
        data = path.read_bytes()
        relative = str(path.relative_to(self.root))
        digest = hashlib.sha256(data).hexdigest()
        if relative in self.hashes and self.hashes[relative] != digest:
            raise ValueError("Experiment artifacts changed during aggregation")
        self.hashes[relative] = digest
        return json.loads(data) if json_value else data


def _identity(record):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", record.task_id):
        raise ValueError("Unsafe task identifier")
    if type(record.repetition) is not int or record.repetition < 1:
        raise ValueError("Repetition must be a positive integer")
    if record.schema_version != SCHEMA_VERSION:
        raise ValueError("Unsupported record schema version")


def _verification(value):
    results = []
    for name in ("sany", "tlc"):
        result = ToolResult(
            **{
                **value[name],
                "status": CheckStatus(value[name]["status"]),
                "command": tuple(value[name]["command"]),
            }
        )
        _number(result.duration_seconds)
        if result.status == CheckStatus.NOT_RUN and (
            result.command
            or result.stdout
            or result.stderr
            or result.exit_code is not None
            or result.duration_seconds != 0
        ):
            raise ValueError("Invalid not_run verification")
        results.append(result)
    return VerificationResult(*results)


def summarize(root: Path) -> dict:
    evidence = _Evidence(root)
    manifest = evidence.read("manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("phase") not in {
        "pilot",
        "main",
    }:
        raise ValueError("A versioned pilot/main manifest is required")
    # Client artifacts without context must not silently become zero expense.
    for directory in evidence.root.glob("calls/*"):
        if directory.is_dir() and not (
            (directory / "record.json").is_file() or (directory / "error.json").is_file()
        ):
            raise ValueError("Missing CallRecord or technical error event for a call")
    calls = {}
    for path in sorted(evidence.root.glob("calls/*/record.json")):
        value = evidence.read(str(path.relative_to(evidence.root)))
        call = CallRecord(
            **{
                **value,
                "usage": _usage(value["usage"]),
                "method": Method(value["method"]) if value["method"] else None,
            }
        )
        _identity(call)
        _number(call.duration_seconds)
        if call.purpose not in {"initial", "repair", "explanation"}:
            raise ValueError("Unknown call purpose")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", call.call_id) or call.call_id != path.parent.name:
            raise ValueError("Call ID must match its artifact directory")
        if call.call_id in calls:
            raise ValueError("Duplicate call ID")
        evidence.read(call.request_path)
        evidence.read(call.response_path)
        calls[call.call_id] = call

    attempts = defaultdict(list)
    used_explanations = set()
    checks = {}
    for path in sorted(evidence.root.glob("tasks/*/repeat-*/*/attempt-*/record.json")):
        value = evidence.read(str(path.relative_to(evidence.root)))
        attempt = AttemptRecord(**{**value, "method": Method(value["method"])})
        _identity(attempt)
        if type(attempt.iteration) is not int:
            raise ValueError("Iteration must be an integer")
        expected = (
            Path("tasks")
            / attempt.task_id
            / f"repeat-{attempt.repetition:02d}"
            / (attempt.method.value)
            / f"attempt-{attempt.iteration:02d}"
            / "record.json"
        )
        if path != evidence.root / expected:
            raise ValueError("Attempt identity differs from its directory")
        call = calls.get(attempt.generation_call_id)
        if call is None or (call.task_id, call.repetition) != (attempt.task_id, attempt.repetition):
            raise ValueError("Missing or mismatched generation call")
        if (attempt.iteration == 1 and call.purpose != "initial") or (
            attempt.iteration > 1 and (call.purpose != "repair" or call.method != attempt.method)
        ):
            raise ValueError("Invalid generation call purpose/method")
        evidence.read(attempt.specification_path, json_value=False)
        verification_path = str(evidence.path(attempt.verification_path).relative_to(evidence.root))
        check = _verification(evidence.read(verification_path))
        checks[verification_path] = check
        if attempt.iteration == 1 and attempt.feedback_path is not None:
            raise ValueError("Initial attempt must not have repair feedback")
        if attempt.iteration > 1:
            if attempt.feedback_path is None:
                raise ValueError("Repair attempt requires saved feedback")
            feedback = evidence.read(attempt.feedback_path)
            if feedback["method"] != attempt.method.value or not isinstance(feedback["text"], str):
                raise ValueError("Feedback method/text differs from attempt")
            explanation_id = feedback.get("explanation_call_id")
            if attempt.method == Method.LLM_EXPLANATION:
                explanation = calls.get(explanation_id)
                if explanation is None or (
                    explanation.purpose,
                    explanation.task_id,
                    explanation.repetition,
                    explanation.method,
                ) != ("explanation", attempt.task_id, attempt.repetition, attempt.method):
                    raise ValueError("Missing or mismatched explanation call")
                if explanation_id in used_explanations:
                    raise ValueError("Explanation calls must be fresh for each repair")
                used_explanations.add(explanation_id)
            elif explanation_id is not None:
                raise ValueError("Only method C may use an explanation call")
        attempts[(attempt.task_id, attempt.repetition, attempt.method)].append((attempt, check))

    branches, initial_by_group, used_generations = [], {}, set()
    for (task, repetition, method), rows in sorted(attempts.items()):
        rows.sort(key=lambda row: row[0].iteration)
        if [a.iteration for a, _ in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("Attempt iterations must be contiguous from 1")
        if any(check.success for _, check in rows[:-1]):
            raise ValueError("Attempts continued after successful verification")
        first = rows[0][0]
        shared = (first.generation_call_id, first.specification_path, first.verification_path)
        key = (task, repetition)
        if key in initial_by_group and initial_by_group[key] != shared:
            raise ValueError("Methods must share initial call, specification and verification")
        initial_by_group[key] = shared
        for attempt, _ in rows:
            if attempt.iteration > 1 and attempt.generation_call_id in used_generations:
                raise ValueError("Repair calls must be unique across attempts")
            used_generations.add(attempt.generation_call_id)
        relevant = [
            c
            for c in calls.values()
            if (c.task_id, c.repetition, c.method) == (task, repetition, method)
        ]
        last = rows[-1][1]
        technical = any(
            r.status in {CheckStatus.TOOL_ERROR, CheckStatus.TIMEOUT} for r in (last.sany, last.tlc)
        )
        outcome = (
            "success"
            if last.success
            else "technical_interruption"
            if technical
            else "exhausted"
            if len(rows) == 10
            else "unresolved"
        )
        errors = Counter(
            r.error_type or "unknown"
            for _, v in rows
            for r in (v.sany, v.tlc)
            if r.status == CheckStatus.SPEC_ERROR
        )
        statuses = Counter(r.status.value for _, v in rows for r in (v.sany, v.tlc))
        branches.append(
            {
                "task_id": task,
                "repetition": repetition,
                "method": method.value,
                "sany_first_pass": next(
                    (a.iteration for a, v in rows if v.sany.status == CheckStatus.PASSED), None
                ),
                "tlc_first_pass": next(
                    (a.iteration for a, v in rows if v.tlc.status == CheckStatus.PASSED), None
                ),
                "iterations_total": len(rows),
                "success": last.success,
                "outcome": outcome,
                "initial_success": rows[0][1].success,
                "shared_initial_call_id": shared[0],
                "error_types": dict(errors),
                "tool_status_counts": dict(statuses),
                "repair": _totals(_call_values(c) for c in relevant if c.purpose == "repair"),
                "explanation": _totals(
                    _call_values(c) for c in relevant if c.purpose == "explanation"
                ),
                "verification_seconds_observed": sum(
                    v.sany.duration_seconds + v.tlc.duration_seconds for _, v in rows
                ),
            }
        )

    failed = []
    for path in sorted(evidence.root.glob("calls/*/error.json")):
        if path.parent.name in calls:
            raise ValueError("Call cannot be both completed and failed")
        event = evidence.read(str(path.relative_to(evidence.root)))
        usage = _usage(event.get("usage", {}))
        duration = event.get("duration_seconds")
        if duration is not None:
            _number(duration)
        failed.append({"call_id": path.parent.name, **asdict(usage), "duration_seconds": duration})
    methods = {}
    for method in Method:
        observed = [b for b in branches if b["method"] == method.value]
        successes = [b for b in observed if b["success"]]
        repair_eligible = [b for b in observed if not b["initial_success"]]
        method_calls = [c for c in calls.values() if c.method == method]
        methods[method.value] = {
            "repair": _totals(_call_values(c) for c in method_calls if c.purpose == "repair"),
            "explanation": _totals(
                _call_values(c) for c in method_calls if c.purpose == "explanation"
            ),
            "branches_observed": len(observed),
            "successes_observed": len(successes),
            "success_rate_observed": len(successes) / len(observed) if observed else None,
            "outcomes": dict(Counter(b["outcome"] for b in observed)),
            "initial_successes": sum(b["initial_success"] for b in observed),
            "repair_eligible_observed": len(repair_eligible),
            "repair_successes_observed": sum(b["success"] for b in repair_eligible),
            "iterations_mean_all_observed": mean(b["iterations_total"] for b in observed)
            if observed
            else None,
            "iterations_mean_successful_only": mean(b["iterations_total"] for b in successes)
            if successes
            else None,
            "iterations_median_successful_only": median(b["iterations_total"] for b in successes)
            if successes
            else None,
        }
    # Reject a snapshot that changed while it was being aggregated.
    for relative, expected_hash in evidence.hashes.items():
        if hashlib.sha256(evidence.path(relative).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Experiment artifacts changed during aggregation")
    return {
        "metrics_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "schema_version": SCHEMA_VERSION,
        "metrics_version": "observed-v1",
        "experiment_id": manifest.get("experiment_id"),
        "phase": manifest["phase"],
        "branches": branches,
        "methods": methods,
        "observed_task_repetitions": len(initial_by_group),
        "missing_method_branches": len(initial_by_group) * 3 - len(branches),
        "completed_calls": {
            purpose: _totals(_call_values(c) for c in calls.values() if c.purpose == purpose)
            for purpose in ("initial", "repair", "explanation")
        },
        "completed_calls_total": _totals(_call_values(c) for c in calls.values()),
        "technical_failed_calls": _totals(failed),
        "completed_and_failed_events_total": _totals(
            [_call_values(c) for c in calls.values()] + failed
        ),
        "technical_failed_call_ids": [e["call_id"] for e in failed],
        "unconsumed_explanation_call_ids": sorted(
            c.call_id
            for c in calls.values()
            if c.purpose == "explanation" and c.call_id not in used_explanations
        ),
        "unreferenced_generation_call_ids": sorted(
            c.call_id
            for c in calls.values()
            if c.purpose != "explanation" and c.call_id not in used_generations
        ),
        "verification_seconds_unique": sum(
            v.sany.duration_seconds + v.tlc.duration_seconds for v in checks.values()
        ),
        "sany_seconds_unique": sum(v.sany.duration_seconds for v in checks.values()),
        "tlc_seconds_unique": sum(v.tlc.duration_seconds for v in checks.values()),
        "cost": None,
        "cost_note": "No verified pricing supplied; tokens are reported separately",
        "source_sha256": dict(sorted(evidence.hashes.items())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.experiment)
    # Reserve a new output directory only after successful validation.
    args.output.mkdir(parents=True, exist_ok=False)
    _write_json(args.output / "summary.json", result)
    fields = [
        "task_id",
        "repetition",
        "method",
        "iterations_total",
        "sany_first_pass",
        "tlc_first_pass",
        "success",
        "initial_success",
        "outcome",
        "shared_initial_call_id",
    ]
    with (args.output / "branches.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["branches"])
    print(json.dumps({"output": str(args.output), "branches": len(result["branches"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
