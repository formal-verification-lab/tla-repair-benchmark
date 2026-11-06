import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmark import SCHEMA_VERSION
from benchmark.contracts import (
    CheckStatus,
    LLMCallError,
    LLMResponse,
    Message,
    ModelSettings,
    TaskInput,
    ToolResult,
    Usage,
    VerificationResult,
    VerifierSettings,
)
from benchmark.feedback import EXPLANATION_PROMPT
from benchmark.metrics import summarize
from benchmark.runner import Runner, initial_messages


class Client:
    def __init__(self, *, initial="broken", repair="repaired", fail_at=None):
        self.initial, self.repair, self.fail_at = initial, repair, fail_at
        self.calls = []

    def generate(self, messages, settings):
        self.calls.append((messages, settings))
        if len(self.calls) == self.fail_at:
            raise LLMCallError("synthetic API failure")
        text = (
            "Explanation only."
            if messages[0].content == EXPLANATION_PROMPT
            else self.initial
            if len(messages) == 2
            else self.repair
        )
        return LLMResponse(
            f"call-{len(self.calls)}",
            text,
            "version-1",
            Usage(10, 5),
            0.1,
            {"text": text, "model": "version-1"},
        )


class Checker:
    def __init__(self, *, technical=False):
        self.technical = technical
        self.sources = []

    def verify(self, source, cfg, settings, work_dir):
        text = source.read_text()
        self.sources.append(text)
        status = (
            CheckStatus.TIMEOUT
            if self.technical
            else CheckStatus.PASSED
            if text == "repaired"
            else CheckStatus.SPEC_ERROR
        )
        return VerificationResult(
            ToolResult("sany", CheckStatus.PASSED, ("fixture",), "", "", 0, 0.01),
            ToolResult(
                "tlc",
                status,
                ("fixture",),
                "Error: Invariant Safe is violated\nState 1:\nx=0\nState 2:\nx=1\n"
                if status != CheckStatus.PASSED
                else "",
                "",
                0 if status == CheckStatus.PASSED else 1,
                0.02,
                "invariant" if status == CheckStatus.SPEC_ERROR else None,
            ),
        )


def setup(tmp_path, *, client=None, checker=None, max_attempts=10, repetitions=1):
    client, checker = client or Client(), checker or Checker()
    settings = ModelSettings("fixture", "requested-model", {"max_tokens": 32})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "fixture",
        "phase": "pilot",
        "model": {
            "provider": settings.provider,
            "requested_id": settings.model,
            "resolved_id": "version-1",
            "parameters": dict(settings.parameters),
        },
        "protocol": {"max_attempts": max_attempts, "repetitions": repetitions},
    }
    runner = Runner(
        tmp_path / "experiment",
        client,
        checker,
        settings,
        VerifierSettings(Path("unused.jar"), "unused", "java", 30, 60),
        max_attempts=max_attempts,
        repetitions=repetitions,
    )
    task = TaskInput("fixture", "Sample", "A synthetic task", "INIT Init\nNEXT Next\n")
    return runner, client, checker, task, manifest


def test_actual_runner_shares_initial_and_isolates_branches_and_explainer(tmp_path, monkeypatch):
    runner, client, checker, task, manifest = setup(tmp_path)
    ref = tmp_path / "reference/Gold.tla"
    ref.parent.mkdir()
    ref.write_text("SECRET_REFERENCE")
    original = Path.open

    def guard(path, *args, **kwargs):
        if "reference" in path.parts:
            raise AssertionError("Runner opened Gold")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    result = runner.run([task], manifest)
    assert len(client.calls) == 5 and len(checker.sources) == 4
    assert result["completed_calls"]["initial"]["calls"] == 1
    assert result["completed_calls"]["repair"]["calls"] == 3
    assert result["completed_calls"]["explanation"]["calls"] == 1
    assert result["completed_calls_total"]["input_tokens"] == 50
    assert result["missing_method_branches"] == 0
    assert all(b["tlc_first_pass"] == 2 for b in result["branches"])
    initials = [
        json.loads(p.read_text())
        for p in runner.root.glob("tasks/*/repeat-01/*/attempt-01/record.json")
    ]
    assert (
        len(
            {
                (i["generation_call_id"], i["specification_path"], i["verification_path"])
                for i in initials
            }
        )
        == 1
    )
    repairs = [m for m, _ in client.calls if len(m) > 2]
    assert all(
        m[:2] == initial_messages(task) and m[2] == Message("assistant", "broken") for m in repairs
    )
    explanation = [m for m, _ in client.calls if m[0].content == EXPLANATION_PROMPT]
    assert len(explanation) == 1 and len(explanation[0]) == 2
    assert task.description not in explanation[0][1].content
    assert task.cfg_text not in explanation[0][1].content
    assert "Explanation only." in repairs[-1][-1].content
    assert "Explanation only." not in repairs[0][-1].content
    assert "SECRET_REFERENCE" not in json.dumps(
        [[asdict(m) for m in messages] for messages, _ in client.calls]
    )
    assert not result["unconsumed_explanation_call_ids"]


def test_ten_attempt_budget_counts_initial_but_not_explanations(tmp_path):
    runner, client, _, task, manifest = setup(tmp_path, client=Client(repair="still broken"))
    result = runner.run([task], manifest)
    assert len(client.calls) == 37  # one initial, 27 repairs, nine explanations
    assert all(
        b["iterations_total"] == 10 and b["outcome"] == "exhausted" for b in result["branches"]
    )
    assert result["completed_calls"]["explanation"]["calls"] == 9
    explanations = [m for m, _ in client.calls if m[0].content == EXPLANATION_PROMPT]
    assert all(len(m) == 2 for m in explanations)


def test_initial_success_needs_one_call_per_repetition_without_feedback(tmp_path):
    runner, client, _, task, manifest = setup(
        tmp_path, client=Client(initial="repaired"), repetitions=3
    )
    result = runner.run([task], manifest)
    assert len(client.calls) == 3
    assert len(result["branches"]) == 9
    assert all(b["initial_success"] and b["iterations_total"] == 1 for b in result["branches"])
    assert not list(runner.root.glob("feedback/*"))


def test_technical_verifier_status_is_not_sent_to_the_model_for_repair(tmp_path):
    runner, client, _, task, manifest = setup(tmp_path, checker=Checker(technical=True))
    result = runner.run([task], manifest)
    assert len(client.calls) == 1
    assert all(b["outcome"] == "technical_interruption" for b in result["branches"])


def test_api_failure_preserves_successful_explanation_without_false_attempt(tmp_path):
    runner, client, _, task, manifest = setup(tmp_path, client=Client(fail_at=5))
    with pytest.raises(LLMCallError):
        runner.run([task], manifest)
    assert len(client.calls) == 5
    assert (runner.root / "failure.json").exists() and not (runner.root / "complete.json").exists()
    result = summarize(runner.root)
    assert result["completed_calls_total"]["calls"] == 4
    assert result["technical_failed_calls"]["calls"] == 1
    assert result["unconsumed_explanation_call_ids"] == ["call-4"]
    branch = next(b for b in result["branches"] if b["method"] == "C")
    assert branch["iterations_total"] == 1


def test_save_failure_stops_before_next_model_call(tmp_path, monkeypatch):
    from benchmark import runner as module

    runner, client, _, task, manifest = setup(tmp_path)
    original = module._write_json

    def fail(path, value):
        if path.name == "feedback.json":
            raise OSError("synthetic disk failure")
        return original(path, value)

    monkeypatch.setattr(module, "_write_json", fail)
    with pytest.raises(OSError, match="disk failure"):
        runner.run([task], manifest)
    assert len(client.calls) == 1


def test_existing_run_is_never_overwritten(tmp_path):
    runner, client, _, task, manifest = setup(tmp_path, client=Client(initial="repaired"))
    runner.run([task], manifest)
    with pytest.raises(FileExistsError):
        runner.run([task], manifest)
    assert len(client.calls) == 1


@pytest.mark.skipif(not os.environ.get("TLA_TOOLS_JAR"), reason="Set TLA_TOOLS_JAR for real Java")
@pytest.mark.parametrize("defect", ["missing_constant", "missing_spec", "non_function"])
def test_runner_integrates_real_java_and_all_strategies_for_configuration_errors(tmp_path, defect):
    from benchmark.sft_data import sha
    from benchmark.verifier import JavaVerifier

    source = (
        "---- MODULE Sample ----\nEXTENDS Naturals\nCONSTANT N\nVARIABLE x\n"
        "Init == x=0\nNext == x'=1-x\nInv == x \\in {0,1}\n====\n"
    )
    repaired = source.replace("CONSTANT N\n", "")
    cfg = "INIT Init\nNEXT Next\nINVARIANT Inv\n"
    if defect == "missing_spec":
        source = repaired
        repaired = repaired.replace("====", "Spec == Init /\\ [][Next]_x\n====")
        cfg = "SPECIFICATION Spec\nINVARIANT Inv\n"
    if defect == "non_function":
        source = repaired.replace("Init == x=0", "Init == x = [1..2 -> {0,1}]")
        source = source.replace("Next == x'=1-x", "Next == x' = x[1]")
        source = source.replace("Inv == x \\in {0,1}", "Inv == TRUE")
    client = Client(initial=source, repair=repaired)
    runner, _, _, task, manifest = setup(tmp_path, client=client)
    jar = Path(os.environ["TLA_TOOLS_JAR"]).resolve()
    runner.verifier = JavaVerifier()
    runner.verifier_settings = VerifierSettings(jar, sha(jar), "java", 30, 60)
    task = TaskInput(task.task_id, task.module_name, task.description, cfg)
    result = runner.run([task], manifest)
    assert all(b["tlc_first_pass"] == 2 for b in result["branches"])
    category = "evaluation" if defect == "non_function" else "configuration"
    assert all(b["error_types"] == {category: 1} for b in result["branches"])
    assert result["completed_calls_total"]["calls"] == 5
