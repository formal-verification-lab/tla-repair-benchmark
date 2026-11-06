"""Post-hoc diagnostic labels; never rewrite verification results or LLM inputs."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from benchmark.metrics import _Evidence, summarize

VERSION = "diagnostic-audit-v1"


def classify(tool, status, source, log):
    """Return one observed category and rule, not a semantic correctness verdict."""
    if status in {"tool_error", "timeout"}:
        return "technical_" + status, "recorded_status"
    if status != "spec_error":
        return status, "recorded_status"
    if tool == "sany":
        # A bare MODULE line is direct source evidence. A Java NPE by itself
        # is insufficient to attribute a tool crash to a specification defect.
        if re.search(r"^\s*MODULE\s+\w+\s*$", source, re.M) and not re.search(
            r"^-{4,}\s+MODULE\s+\w+\s+-{4,}", source, re.M
        ):
            return "module_header", "bare_module_without_delimiters"
        if "***Parse Error***" in log:
            return "parse_syntax", "sany_parse_error"
        if "Could not find declaration or definition" in log:
            return "unresolved_symbol", "sany_missing_declaration"
        if "NullPointerException" in log:
            return "parser_exception_unattributed", "sany_npe_requires_review"
    if tool == "tlc":
        if re.search(r"Error: Invariant .+ is violated", log):
            return "invariant_violation", "tlc_invariant_counterexample"
        if "is either undefined or not an operator" in log:
            return "evaluation_unbound_identifier", "tlc_unbound_identifier"
    return "unclassified_spec_error", "no_matching_rule"


def analyze(root):
    """Classify distinct checked versions; retain all A/B/C references to initial."""
    metrics = summarize(root)
    evidence = _Evidence(root)
    rows = {}
    for path in sorted(evidence.root.glob("tasks/*/repeat-*/*/attempt-*/record.json")):
        record = evidence.read(str(path.relative_to(evidence.root)))
        verification = evidence.read(record["verification_path"])
        source = evidence.read(record["specification_path"], json_value=False).decode("utf-8")
        for tool in ("sany", "tlc"):
            result = verification[tool]
            if result["status"] in {"passed", "not_run"}:
                continue
            key = (record["verification_path"], tool)
            if key not in rows:
                category, rule = classify(
                    tool, result["status"], source, result["stdout"] + "\n" + result["stderr"]
                )
                rows[key] = {
                    "task_id": record["task_id"],
                    "repetition": record["repetition"],
                    "iteration": record["iteration"],
                    "generation_call_id": record["generation_call_id"],
                    "specification_path": record["specification_path"],
                    "verification_path": record["verification_path"],
                    "tool": tool,
                    "recorded_status": result["status"],
                    "recorded_error_type": result["error_type"],
                    "category": category,
                    "rule": rule,
                    "parser_exception": "NullPointerException"
                    in (result["stdout"] + result["stderr"]),
                    "branch_references": [],
                }
            rows[key]["branch_references"].append(record["method"])
    entries = list(rows.values())
    for relative, expected in evidence.hashes.items():
        if hashlib.sha256(evidence.path(relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Experiment artifacts changed during analysis")
    return {
        "analysis_version": VERSION,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "experiment_id": metrics["experiment_id"],
        "unique_error_counts": dict(Counter(r["category"] for r in entries)),
        "branch_error_counts": {
            m: dict(Counter(r["category"] for r in entries if m in r["branch_references"]))
            for m in ("A", "B", "C")
        },
        "errors": entries,
        "source_sha256": dict(sorted({**metrics["source_sha256"], **evidence.hashes}.items())),
        "metrics": metrics,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.experiment.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Analysis must be outside the original experiment")
    report = analyze(root)
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = [
        "task_id",
        "repetition",
        "iteration",
        "tool",
        "category",
        "rule",
        "recorded_status",
        "recorded_error_type",
        "parser_exception",
        "branch_references",
        "generation_call_id",
        "specification_path",
        "verification_path",
    ]
    with (output / "errors.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {**r, "branch_references": ",".join(r["branch_references"])} for r in report["errors"]
        )
    print(json.dumps({"output": str(output), "unique_error_counts": report["unique_error_counts"]}))


if __name__ == "__main__":
    main()
