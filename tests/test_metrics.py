import json
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmark import SCHEMA_VERSION
from benchmark.contracts import (
    AttemptRecord,
    CallRecord,
    CheckStatus,
    FeedbackResult,
    Method,
    ToolResult,
    Usage,
    VerificationResult,
)
from benchmark.metrics import main, summarize


def write(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value))
    return path


def call(root, name, purpose, method=None, usage=Usage(100, 10)):
    request = write(root, f"calls/{name}/request.json", {"messages": []})
    response = write(root, f"calls/{name}/response.json", {"text": "synthetic reply"})
    record = CallRecord(name, purpose, "easy_02", 1, method, request, response, usage, 2)
    write(root, f"calls/{name}/record.json", asdict(record))
    return name


def check(root, path, status=CheckStatus.SPEC_ERROR, sany=CheckStatus.PASSED):
    result = VerificationResult(
        ToolResult("sany", sany, ("java",), "synthetic log", "", 0, 0.25),
        ToolResult(
            "tlc",
            status,
            ("java",),
            "synthetic log",
            "",
            0,
            0.75,
            "invariant" if status == CheckStatus.SPEC_ERROR else None,
        ),
    )
    return write(root, path, asdict(result))


def attempt(root, method, iteration, generation, verification, feedback=None, spec="initial.tla"):
    record = AttemptRecord(
        "easy_02", 1, method, iteration, generation, spec, verification, feedback
    )
    path = f"tasks/easy_02/repeat-01/{method}/attempt-{iteration:02d}/record.json"
    write(root, path, asdict(record))


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "experiment"
    write(
        root,
        "manifest.json",
        {"schema_version": SCHEMA_VERSION, "phase": "pilot", "experiment_id": "synthetic-fixture"},
    )
    (root / "initial.tla").write_text("synthetic specification")
    initial = call(root, "shared", "initial")
    initial_check = check(root, "initial-verification.json")
    for method in Method:
        attempt(root, method, 1, initial, initial_check)
    repair = call(root, "repair-A", "repair", Method.FULL_LOG, Usage(20, 5))
    passed = check(root, "A-passed.json", CheckStatus.PASSED)
    feedback = write(root, "A-feedback.json", asdict(FeedbackResult(Method.FULL_LOG, "log")))
    attempt(root, Method.FULL_LOG, 2, repair, passed, feedback)
    call(root, "explanation-C", "explanation", Method.LLM_EXPLANATION, Usage(7, 3))
    feedback = write(
        root,
        "C-feedback.json",
        asdict(FeedbackResult(Method.LLM_EXPLANATION, "explanation", "explanation-C")),
    )
    repair = call(root, "repair-C", "repair", Method.LLM_EXPLANATION, Usage(30, 8))
    timed_out = check(root, "C-timeout.json", CheckStatus.TIMEOUT)
    attempt(root, Method.LLM_EXPLANATION, 2, repair, timed_out, feedback)
    return root


def test_shared_initial_and_explainer_accounted_once(experiment):
    report = summarize(experiment)
    assert report["completed_calls_total"]["calls"] == 4
    assert report["completed_calls_total"]["input_tokens"] == 157
    assert report["completed_calls"]["initial"]["input_tokens"] == 100
    assert report["completed_calls"]["repair"]["input_tokens"] == 50
    assert report["completed_calls"]["explanation"]["input_tokens"] == 7
    assert report["verification_seconds_unique"] == 3
    branches = {b["method"]: b for b in report["branches"]}
    assert branches["A"]["sany_first_pass"] == 1
    assert branches["A"]["tlc_first_pass"] == 2
    assert branches["A"]["iterations_total"] == 2
    assert branches["A"]["outcome"] == "success"
    assert branches["B"]["outcome"] == "unresolved"
    assert branches["B"]["tlc_first_pass"] is None
    assert branches["C"]["outcome"] == "technical_interruption"
    assert branches["C"]["explanation"]["calls"] == 1
    assert branches["C"]["iterations_total"] == 2
    assert branches["C"]["error_types"] == {"invariant": 1}
    assert branches["C"]["tool_status_counts"]["timeout"] == 1
    assert report["missing_method_branches"] == 0
    assert report["methods"]["A"]["iterations_mean_all_observed"] == 2
    assert report["methods"]["B"]["iterations_mean_successful_only"] is None
    assert report["cost"] is None


def test_unknown_usage_is_not_zero_and_failed_events_are_separate(experiment):
    path = experiment / "calls/repair-A/record.json"
    value = json.loads(path.read_text())
    value["usage"]["input_tokens"] = None
    path.write_text(json.dumps(value))
    write(experiment, "calls/failed/error.json", {"duration_seconds": 0.5, "usage": {}})
    report = summarize(experiment)
    total = report["completed_calls_total"]
    assert total["input_tokens"] is None
    assert total["input_tokens_known_sum"] == 137
    assert total["input_tokens_unknown_count"] == 1
    failed = report["technical_failed_calls"]
    assert failed["calls"] == 1 and failed["input_tokens"] is None
    assert report["completed_and_failed_events_total"]["calls"] == 5
    assert report["completed_and_failed_events_total"]["input_tokens_unknown_count"] == 2
    assert report["branches"][0]["iterations_total"] == 2


def test_initial_success_not_counted_as_successful_repair(experiment):
    for method in (Method.FULL_LOG, Method.LLM_EXPLANATION):
        path = experiment / f"tasks/easy_02/repeat-01/{method}/attempt-02/record.json"
        path.unlink()
    check(experiment, "initial-verification.json", CheckStatus.PASSED)
    report = summarize(experiment)
    for method in report["methods"].values():
        assert method["initial_successes"] == 1
        assert method["repair_eligible_observed"] == 0
        assert method["repair_successes_observed"] == 0
    # API usage remains real even when records aren't linked to attempts.
    assert report["completed_calls_total"]["calls"] == 4
    assert report["unreferenced_generation_call_ids"] == ["repair-A", "repair-C"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_call", "Missing or mismatched generation"),
        ("wrong_feedback", "Feedback method"),
        ("wrong_explanation", "Missing or mismatched explanation"),
        ("traversal", "within the experiment"),
        ("unsupported_schema", "Unsupported record schema"),
        ("fractional_tokens", "Token counts"),
    ],
)
def test_invalid_evidence_is_rejected(experiment, mutation, match):
    target = experiment / "tasks/easy_02/repeat-01/C/attempt-02/record.json"
    value = json.loads(target.read_text())
    if mutation == "missing_call":
        value["generation_call_id"] = "absent"
    elif mutation == "traversal":
        value["specification_path"] = "../outside.tla"
    elif mutation == "unsupported_schema":
        value["schema_version"] = "999"
    elif mutation == "wrong_feedback":
        write(experiment, "C-feedback.json", {"method": "A", "text": "log"})
    elif mutation == "wrong_explanation":
        write(
            experiment,
            "C-feedback.json",
            {"method": "C", "text": "log", "explanation_call_id": "shared"},
        )
    elif mutation == "fractional_tokens":
        path = experiment / "calls/shared/record.json"
        record = json.loads(path.read_text())
        record["usage"]["input_tokens"] = 0.5
        path.write_text(json.dumps(record))
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=match):
        summarize(experiment)


def test_methods_must_share_same_initial_verification(experiment):
    target = experiment / "tasks/easy_02/repeat-01/B/attempt-01/record.json"
    value = json.loads(target.read_text())
    value["verification_path"] = "A-passed.json"
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="share initial"):
        summarize(experiment)


def test_missing_method_is_visible(experiment):
    (experiment / "tasks/easy_02/repeat-01/B/attempt-01/record.json").unlink()
    report = summarize(experiment)
    assert report["missing_method_branches"] == 1
    assert report["methods"]["B"]["success_rate_observed"] is None


def test_no_gold_reads_and_output_does_not_overwrite(experiment, tmp_path, monkeypatch):
    (experiment / "reference").mkdir()
    (experiment / "reference/Gold.tla").write_text("Must never enter aggregation")
    original = Path.read_bytes

    def guard(path):
        if "reference" in path.parts:
            raise AssertionError("Gold was read by metrics")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    output = tmp_path / "summary"
    assert main(["--experiment", str(experiment), "--output", str(output)]) == 0
    assert (output / "summary.json").exists() and (output / "branches.csv").exists()
    report = json.loads((output / "summary.json").read_text())
    assert "manifest.json" in report["source_sha256"]
    with pytest.raises(FileExistsError):
        main(["--experiment", str(experiment), "--output", str(output)])


def test_mean_over_all_branches_does_not_hide_unresolved_run(experiment):
    generation = call(experiment, "repeat-two", "initial")
    path = experiment / "calls/repeat-two/record.json"
    value = json.loads(path.read_text())
    value["repetition"] = 2
    path.write_text(json.dumps(value))
    verification = check(experiment, "repeat-two-check.json")
    for method in Method:
        record = AttemptRecord("easy_02", 2, method, 1, generation, "initial.tla", verification)
        write(
            experiment, f"tasks/easy_02/repeat-02/{method}/attempt-01/record.json", asdict(record)
        )
    stats = summarize(experiment)["methods"]["A"]
    assert stats["iterations_mean_all_observed"] == 1.5
    assert stats["iterations_mean_successful_only"] == 2
    assert stats["success_rate_observed"] == 0.5
    assert stats["outcomes"] == {"success": 1, "unresolved": 1}


def test_exhausted_run_has_ten_main_attempts(experiment):
    feedback = write(
        experiment, "B-feedback.json", asdict(FeedbackResult(Method.FILTERED_LOG, "log"))
    )
    for iteration in range(2, 11):
        generation = call(experiment, f"B-{iteration}", "repair", Method.FILTERED_LOG)
        verification = check(experiment, f"B-{iteration}-check.json")
        attempt(experiment, Method.FILTERED_LOG, iteration, generation, verification, feedback)
    branch = next(b for b in summarize(experiment)["branches"] if b["method"] == "B")
    assert branch["iterations_total"] == 10 and branch["outcome"] == "exhausted"
    assert branch["tlc_first_pass"] is None and not branch["success"]


def test_symlink_escape_is_rejected(experiment, tmp_path):
    outside = tmp_path / "outside.tla"
    outside.write_text("outside experiment")
    spec = experiment / "initial.tla"
    spec.unlink()
    spec.symlink_to(outside)
    with pytest.raises(ValueError, match="within the experiment"):
        summarize(experiment)


def test_changed_artifact_changes_evidence_fingerprint(experiment):
    first = summarize(experiment)
    (experiment / "initial.tla").write_text("changed specification")
    second = summarize(experiment)
    assert first["source_sha256"]["initial.tla"] != second["source_sha256"]["initial.tla"]


def test_attempt_gap_and_continuing_after_success_are_rejected(experiment):
    path = experiment / "tasks/easy_02/repeat-01/A/attempt-01/record.json"
    contents = path.read_text()
    path.unlink()
    with pytest.raises(ValueError, match="contiguous"):
        summarize(experiment)
    path.write_text(contents)
    check(experiment, "initial-verification.json", CheckStatus.PASSED)
    with pytest.raises(ValueError, match="continued after successful"):
        summarize(experiment)


def test_reused_explanation_is_rejected(experiment):
    generation = call(experiment, "repair-C-third", "repair", Method.LLM_EXPLANATION)
    verification = check(experiment, "C-third-check.json")
    attempt(experiment, Method.LLM_EXPLANATION, 3, generation, verification, "C-feedback.json")
    with pytest.raises(ValueError, match="fresh for each repair"):
        summarize(experiment)


def test_sany_error_has_no_first_pass_and_tlc_is_not_run(experiment):
    value = VerificationResult(
        ToolResult("sany", CheckStatus.SPEC_ERROR, ("java",), "unknown symbol", "", 0, 0.2),
        ToolResult("tlc", CheckStatus.NOT_RUN, (), "", "", None, 0),
    )
    write(experiment, "initial-verification.json", asdict(value))
    branch = next(b for b in summarize(experiment)["branches"] if b["method"] == "B")
    assert branch["sany_first_pass"] is None and branch["tlc_first_pass"] is None
    assert branch["error_types"] == {"unknown": 1}
    assert branch["tool_status_counts"]["not_run"] == 1


def test_changes_during_aggregation_are_rejected(experiment, monkeypatch):
    from benchmark.metrics import _Evidence

    original = _Evidence.read
    changed = False

    def modify_after_first_read(self, relative, **kwargs):
        nonlocal changed
        result = original(self, relative, **kwargs)
        if relative == "initial.tla" and not changed:
            changed = True
            (experiment / "initial.tla").write_text("changed while reading")
        return result

    monkeypatch.setattr(_Evidence, "read", modify_after_first_read)
    with pytest.raises(ValueError, match="changed during aggregation"):
        summarize(experiment)


def test_provider_artifacts_without_context_cannot_be_reported_as_zero_expense(experiment):
    (experiment / "calls/shared/record.json").unlink()
    with pytest.raises(ValueError, match="Missing CallRecord"):
        summarize(experiment)
