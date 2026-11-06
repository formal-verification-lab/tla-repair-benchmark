import json

import pytest
from test_metrics import experiment as experiment

from benchmark.error_analysis import analyze, classify, main


@pytest.mark.parametrize(
    "tool,status,source,log,expected",
    [
        ("sany", "spec_error", "MODULE Example\n", "NullPointerException", "module_header"),
        (
            "sany",
            "spec_error",
            "---- MODULE Example ----",
            "NullPointerException",
            "parser_exception_unattributed",
        ),
        ("sany", "tool_error", "MODULE Example", "NullPointerException", "technical_tool_error"),
        ("sany", "spec_error", "", "***Parse Error***", "parse_syntax"),
        (
            "sany",
            "spec_error",
            "",
            "Could not find declaration or definition of symbol '<='.",
            "unresolved_symbol",
        ),
        ("tlc", "spec_error", "", "Error: Invariant AtMostOne is violated.", "invariant_violation"),
        (
            "tlc",
            "spec_error",
            "",
            "ack is either undefined or not an operator",
            "evaluation_unbound_identifier",
        ),
        ("tlc", "timeout", "", "", "technical_timeout"),
        ("tlc", "spec_error", "", "unfamiliar diagnostic", "unclassified_spec_error"),
        ("tlc", "spec_error", "", "Error: Deadlock reached.", "deadlock"),
        (
            "tlc",
            "spec_error",
            "",
            "Error: Temporal properties were violated.",
            "temporal_violation",
        ),
        (
            "sany",
            "spec_error",
            "",
            "[] followed by action not of form [A]_v.",
            "temporal_action_form",
        ),
        (
            "tlc",
            "tool_error",
            "",
            "Error: The name Spec specified in the configuration file\n"
            "is not defined in the specification.",
            "configuration_reference",
        ),
    ],
)
def test_diagnostic_categories(tool, status, source, log, expected):
    assert classify(tool, status, source, log)[0] == expected


def test_unique_initial_errors_and_read_only_output(experiment, tmp_path):
    before = {str(p): p.read_bytes() for p in experiment.rglob("*") if p.is_file()}
    report = analyze(experiment)
    assert len(report["errors"]) == 2  # shared initial plus C's technical timeout
    assert report["errors"][0]["branch_references"] == ["A", "B", "C"]
    assert report["unique_error_counts"] == {
        "unclassified_spec_error": 1,
        "technical_timeout": 1,
    }
    assert report["branch_error_counts"]["A"] == {"unclassified_spec_error": 1}
    assert report["status_disagreements"] == []
    output = tmp_path / "analysis"
    main(["--experiment", str(experiment), "--output", str(output)])
    assert json.loads((output / "analysis.json").read_text()) == report
    assert {str(p): p.read_bytes() for p in experiment.rglob("*") if p.is_file()} == before
    with pytest.raises(FileExistsError):
        main(["--experiment", str(experiment), "--output", str(output)])
    with pytest.raises(ValueError, match="outside"):
        main(["--experiment", str(experiment), "--output", str(experiment / "audit")])


def test_audit_reports_status_disagreement_without_rewriting_recorded_metrics(experiment):
    path = experiment / "C-timeout.json"
    value = json.loads(path.read_text())
    value["tlc"].update(
        status="tool_error",
        exit_code=151,
        error_type="unrecognized_completion",
        stdout="Error: The name Spec specified in the configuration file\n"
        "is not defined in the specification.",
    )
    path.write_text(json.dumps(value))
    original = path.read_bytes()
    report = analyze(experiment)
    assert len(report["status_disagreements"]) == 1
    discrepancy = report["status_disagreements"][0]
    assert discrepancy["recorded_status"] == "tool_error"
    assert discrepancy["audited_status"] == "spec_error"
    branch = next(b for b in report["metrics"]["branches"] if b["method"] == "C")
    assert branch["outcome"] == "technical_interruption"
    assert path.read_bytes() == original
