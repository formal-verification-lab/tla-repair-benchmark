"""Module integration with real Java and synthetic LLM records, not a main runner."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmark import SCHEMA_VERSION
from benchmark.contracts import (
    AttemptRecord,
    CallRecord,
    LLMResponse,
    Method,
    ModelSettings,
    Usage,
    VerifierSettings,
)
from benchmark.feedback import ExplanationStrategy, FilteredLogStrategy, FullLogStrategy
from benchmark.metrics import summarize
from benchmark.verifier import JavaVerifier

SOURCE = """---- MODULE Validation ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x' = 1 - x
Safe == x = 0
====
"""
CFG = "INIT Init\nNEXT Next\nINVARIANT Safe\n"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def record_call(root, call_id, purpose, method, usage, request, response):
    request_path = f"calls/{call_id}/request.json"
    response_path = f"calls/{call_id}/response.json"
    save(root / request_path, request)
    save(root / response_path, response)
    record = CallRecord(
        call_id, purpose, "synthetic_validation", 1, method, request_path, response_path, usage, 0.1
    )
    save(root / f"calls/{call_id}/record.json", asdict(record))


@pytest.mark.skipif(not os.environ.get("TLA_TOOLS_JAR"), reason="Set TLA_TOOLS_JAR for Java checks")
def test_real_counterexample_survives_feedback_and_metrics_accounting(tmp_path, monkeypatch):
    root = tmp_path / "synthetic-integration"
    root.mkdir()
    (root / "Validation.tla").write_text(SOURCE)
    (root / "Spec.cfg").write_text(CFG)
    # Make Gold inaccessible for the entire integration path.
    (root / "reference").mkdir()
    (root / "reference/Gold.tla").write_text("FORBIDDEN_GOLD_MARKER")
    original = Path.open

    def guard(path, *args, **kwargs):
        assert "reference" not in path.parts, "Gold accessed during module integration"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    settings = VerifierSettings(
        Path(os.environ["TLA_TOOLS_JAR"]),
        "936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88",
        "java",
        10,
        10,
    )
    failed = JavaVerifier().verify(
        root / "Validation.tla", root / "Spec.cfg", settings, root / "initial-check"
    )
    assert not failed.success and "State 2:" in failed.tlc.stdout
    save(
        root / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "phase": "pilot",
            "experiment_id": "synthetic-integration-not-a-real-pilot",
        },
    )
    record_call(
        root,
        "initial",
        "initial",
        None,
        Usage(100, 10),
        {"fixture": "synthetic initial request"},
        {"text": SOURCE},
    )

    class StubExplainer:
        def generate(self, messages, model_settings):
            assert [m.role for m in messages] == ["system", "user"]
            assert "FORBIDDEN_GOLD_MARKER" not in json.dumps([asdict(m) for m in messages])
            response = LLMResponse(
                "explanation",
                "The transition violates Safe.",
                model_settings.model,
                Usage(7, 3),
                0.1,
                {"fixture": True},
            )
            record_call(
                root,
                response.call_id,
                "explanation",
                Method.LLM_EXPLANATION,
                response.usage,
                {"messages": [asdict(m) for m in messages]},
                asdict(response),
            )
            return response

    strategies = {
        Method.FULL_LOG: FullLogStrategy(root / "A"),
        Method.FILTERED_LOG: FilteredLogStrategy(root / "B"),
        Method.LLM_EXPLANATION: ExplanationStrategy(
            root / "C", StubExplainer(), ModelSettings("fixture", "synthetic-model", {"test": True})
        ),
    }
    for method, strategy in strategies.items():
        feedback = strategy.build(SOURCE, failed)
        feedback_dir = next((root / method / "feedback").iterdir())
        if method != Method.LLM_EXPLANATION:
            diagnostic = failed.tlc.stdout[failed.tlc.stdout.index("Error:") :]
            assert diagnostic in feedback.text
            assert "State 1:" in feedback.text and "State 2:" in feedback.text
        corrected = root / method / "Validation.tla"
        corrected.write_text(SOURCE.replace("Safe == x = 0", "Safe == x \\in {0, 1}"))
        # The correction is synthetic test input, not an output attributed to an LLM.
        passed = JavaVerifier().verify(
            corrected, root / "Spec.cfg", settings, root / method / "corrected-check"
        )
        assert passed.success
        repair_id = f"repair-{method}"
        record_call(
            root,
            repair_id,
            "repair",
            method,
            Usage(20, 5),
            {"fixture": "synthetic repair request", "feedback": feedback.text},
            {"text": corrected.read_text()},
        )
        for iteration, call_id, spec_path, verification, feedback_path in [
            (1, "initial", "Validation.tla", "initial-check/verification.json", None),
            (
                2,
                repair_id,
                f"{method}/Validation.tla",
                f"{method}/corrected-check/verification.json",
                str((feedback_dir / "feedback.json").relative_to(root)),
            ),
        ]:
            attempt = AttemptRecord(
                "synthetic_validation",
                1,
                method,
                iteration,
                call_id,
                spec_path,
                verification,
                feedback_path,
            )
            save(
                root
                / f"tasks/synthetic_validation/repeat-01/{method}/attempt-{iteration:02d}"
                / "record.json",
                asdict(attempt),
            )

    report = summarize(root)
    assert report["completed_calls"]["initial"]["calls"] == 1
    assert report["completed_calls"]["repair"]["calls"] == 3
    assert report["completed_calls"]["explanation"]["calls"] == 1
    assert report["completed_calls_total"]["input_tokens"] == 167
    assert report["unconsumed_explanation_call_ids"] == []
    assert all(branch["tlc_first_pass"] == 2 for branch in report["branches"])
    assert all(branch["iterations_total"] == 2 for branch in report["branches"])
    assert len(report["branches"]) == 3
    assert report["missing_method_branches"] == 0
