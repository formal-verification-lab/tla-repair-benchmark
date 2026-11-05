import json
from dataclasses import replace

import pytest

from benchmark.contracts import (
    CheckStatus,
    LLMCallError,
    LLMResponse,
    ModelSettings,
    ToolResult,
    Usage,
    VerificationResult,
)
from benchmark.feedback import (
    ExplanationStrategy,
    FilteredLogStrategy,
    FullLogStrategy,
    filter_log,
    verification_log,
)


@pytest.fixture
def verification():
    return VerificationResult(
        ToolResult(
            "sany",
            CheckStatus.PASSED,
            ("java",),
            "Semantic processing of module Spec\n",
            "",
            0,
            0.1,
        ),
        ToolResult(
            "tlc",
            CheckStatus.SPEC_ERROR,
            ("java",),
            "TLC2 Version 2.19\r\nParsing file Spec.tla\r\n"
            "Error: Invariant Safe is violated.\r\n"
            "State 1: <Initial predicate>\r\n/\\ x = 0\r\n"
            "State 2: <Next line 8, col 1>\r\n/\\ x = 1\r\n"
            "Back to state 1: <Next>\r\nFinished in 01s\r\n",
            "stderr context without known markers\n",
            12,
            0.2,
            "invariant",
        ),
    )


def records(root):
    return sorted((root / "feedback").iterdir())


def test_full_log_preserves_both_tools_and_streams(tmp_path, verification):
    result = FullLogStrategy(tmp_path).build("unused specification", verification)
    assert result.method == "A" and result.explanation_call_id is None
    for tool in (verification.sany, verification.tlc):
        assert tool.stdout in result.text
        assert tool.stderr in result.text
    saved = records(tmp_path)[0]
    assert json.loads((saved / "feedback.json").read_text())["text"] == result.text
    assert (saved / "tlc.stdout.log").read_bytes() == verification.tlc.stdout.encode()


def test_filter_removes_only_known_prefix_and_keeps_entire_trace(tmp_path, verification):
    result = FilteredLogStrategy(tmp_path).build("unused", verification)
    assert result.method == "B"
    assert "TLC2 Version" not in result.text
    suffix = verification.tlc.stdout.split("Error:", 1)[1]
    assert "Error:" + suffix in result.text
    assert "Parsing file Spec.tla" in result.text
    assert verification.tlc.stderr in result.text
    assert (
        records(tmp_path)[0] / "tlc.stdout.log"
    ).read_bytes() == verification.tlc.stdout.encode()


@pytest.mark.parametrize(
    "log", ["", "unknown output\n", "Progress(1) x\nunknown\n", "diagnostic without keywords\r\n"]
)
def test_unknown_formats_are_not_discarded(log):
    assert filter_log(log) == log


def test_no_truncation_of_long_counterexample():
    trace = (
        "Error: temporal property violated\n"
        + "".join(f"State {i}: <Next>\n/\\ x = {i}\n" for i in range(1, 5001))
        + "Back to state 1: <Next>\n"
    )
    assert filter_log("TLC2 Version 2.19\n" + trace) == trace


def test_sany_error_and_skipped_tlc_are_visible(tmp_path, verification):
    failed = VerificationResult(
        replace(
            verification.sany,
            status=CheckStatus.SPEC_ERROR,
            stdout="*** Semantic errors: 1\nline 8, col 3 of module Spec\nUnknown X\n",
        ),
        ToolResult("tlc", CheckStatus.NOT_RUN, (), "", "", None, 0),
    )
    text = FilteredLogStrategy(tmp_path).build("unused", failed).text
    assert "line 8, col 3" in text and "Unknown X" in text
    assert "[tlc status=not_run]" in text


class Client:
    def __init__(self, text="Explain the violated invariant."):
        self.calls = []
        self.text = text

    def generate(self, messages, settings):
        self.calls.append((messages, settings))
        return LLMResponse(
            str(len(self.calls)), self.text, settings.model, Usage(10, 5), 0.2, {"raw": self.text}
        )


@pytest.mark.parametrize(
    "reply,empty,code",
    [
        ("Describe the correction.", False, False),
        ("Explanation\n```tla\n---- MODULE Spec ----\n====\n```", False, True),
        ("", True, False),
        ("VARIABLE x\nInit == x = 0", False, True),
    ],
)
def test_explanation_is_verbatim_and_usage_is_saved(tmp_path, verification, reply, empty, code):
    client = Client(reply)
    settings = ModelSettings("gigachat", "same-main-model", {"temperature": 0.1})
    strategy = ExplanationStrategy(tmp_path, client, settings)
    result = strategy.build("---- MODULE Spec ----\n====", verification)
    assert result.method == "C" and result.explanation_call_id == "1"
    assert result.text == reply
    assert len(client.calls) == 1  # No retry for empty or code-containing answers.
    directory = records(tmp_path)[0]
    assert json.loads((directory / "format.json").read_text()) == {
        "empty": empty,
        "code_detected": code,
    }
    saved = json.loads((directory / "response.json").read_text())
    assert saved["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert saved["raw_response"] == {"raw": reply}
    request = json.loads((directory / "request.json").read_text())
    assert request["settings"] == {
        "provider": "gigachat",
        "model": "same-main-model",
        "parameters": {"temperature": 0.1},
    }


def test_every_explanation_gets_fresh_context_and_settings(tmp_path, verification):
    client = Client()
    parameters = {"temperature": 0.1}
    strategy = ExplanationStrategy(tmp_path, client, ModelSettings("test", "model", parameters))
    parameters["temperature"] = 0.9
    strategy.build("FIRST_SPEC", verification)
    client.calls[0][1].parameters["temperature"] = 0.8
    strategy.build("SECOND_SPEC", verification)
    first, second = client.calls
    assert first[0] is not second[0]
    assert [m.role for m in second[0]] == ["system", "user"]
    assert "FIRST_SPEC" not in second[0][1].content
    assert "SECOND_SPEC" in second[0][1].content
    assert verification_log(verification) in second[0][1].content
    assert second[1].parameters["temperature"] == 0.1
    assert len(records(tmp_path)) == 2


def test_api_failure_is_recorded_without_feedback_or_retry(tmp_path, verification):
    class BrokenClient(Client):
        def generate(self, messages, settings):
            self.calls.append((messages, settings))
            raise LLMCallError("Technical failure")

    client = BrokenClient()
    strategy = ExplanationStrategy(tmp_path, client, ModelSettings("test", "model", {}))
    with pytest.raises(LLMCallError):
        strategy.build("spec", verification)
    directory = records(tmp_path)[0]
    assert len(client.calls) == 1
    assert (directory / "request.json").exists() and (directory / "error.json").exists()
    assert not (directory / "feedback.json").exists()


def test_storage_failure_prevents_api_call(tmp_path, verification):
    client = Client()
    root = tmp_path / "not-a-directory"
    root.write_text("occupied")
    strategy = ExplanationStrategy(root, client, ModelSettings("test", "model", {}))
    with pytest.raises(OSError):
        strategy.build("spec", verification)
    assert not client.calls


def test_known_performance_warning_does_not_hide_later_diagnostic():
    prefix = (
        "TLC2 Version 2.19\n"
        "Warning: Please run the Java VM with a throughput optimized garbage collector.\n"
        "(Use the -nowarning option to disable this warning.)\n"
        "Running breadth-first search Model-Checking with 1 worker\n"
        "Parsing file Spec.tla\n"
    )
    diagnostic = "Error: Deadlock reached.\nState 1: <Initial predicate>\n/\\ x = 0\n"
    assert filter_log(prefix + diagnostic) == "Parsing file Spec.tla\n" + diagnostic
    # An unfamiliar warning is never removed.
    assert filter_log("Warning: unknown condition\n" + diagnostic).startswith(
        "Warning: unknown condition\n"
    )
