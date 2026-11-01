import json
from dataclasses import replace
from pathlib import Path

from benchmark import self_repair
from benchmark.sft_data import cases, initial_messages


def test_repairs_actual_responses_with_only_tool_feedback(tmp_path, monkeypatch):
    case = replace(cases()[-1], correct="SECRET_REFERENCE", broken="SECRET_FIXTURE")
    verified = []
    requests = []

    def verify(source, cfg, name, jar, output):
        verified.append(source)
        output.mkdir(parents=True)
        (output / "sany.stdout.log").write_text(f"diagnostic for {source}")
        success = source == "repaired"
        return {
            "sany": {"success": True, "timeout": False},
            "tlc": {"success": success, "timeout": False, "distinct_states": case.expected_states},
        }

    def generate(messages, directory):
        requests.append(json.loads(json.dumps(messages)))
        return "still wrong" if len(requests) == 1 else "repaired"

    monkeypatch.setattr(self_repair, "verify", verify)
    result = self_repair.run_case(
        case,
        "initial wrong",
        initial_messages(case),
        generate,
        Path("tools.jar"),
        tmp_path / "run",
        3,
    )
    assert verified == ["initial wrong", "still wrong", "repaired"]
    assert result["success"] and result["iterations_total"] == 3 and result["repair_calls"] == 2
    assert requests[0][-2]["content"] == "initial wrong"
    assert "diagnostic for initial wrong" in requests[0][-1]["content"]
    assert requests[1][-2]["content"] == "still wrong"
    assert "diagnostic for still wrong" in requests[1][-1]["content"]
    assert requests[1][:-2] == requests[0]
    assert "SECRET_REFERENCE" not in json.dumps(requests)
    assert "SECRET_FIXTURE" not in json.dumps(requests)


def test_stops_at_limit_and_keeps_initial_success_without_repair_calls(tmp_path, monkeypatch):
    case = cases()[-1]

    def verify(source, cfg, name, jar, output):
        output.mkdir(parents=True)
        (output / "sany.stdout.log").write_text("error")
        return {
            "sany": {"success": source == "good", "timeout": False},
            "tlc": {"success": source == "good", "timeout": False},
        }

    def unexpected_call(*args):
        raise AssertionError("Must not call the LLM after success or budget exhaustion")

    monkeypatch.setattr(self_repair, "verify", verify)
    for source, success in (("good", True), ("bad", False)):
        result = self_repair.run_case(
            case,
            source,
            initial_messages(case),
            unexpected_call,
            Path("tools.jar"),
            tmp_path / source,
            1,
        )
        assert result["success"] is success
        assert result["repair_calls"] == 0 and result["iterations_total"] == 1


def test_context_limit_is_recorded_as_incomplete_not_a_model_attempt(tmp_path, monkeypatch):
    case = cases()[-1]

    def verify(source, cfg, name, jar, output):
        output.mkdir(parents=True)
        (output / "sany.stdout.log").write_text("error")
        return {"sany": {"success": False, "timeout": False}}

    def generate(*args):
        raise self_repair.ContextLimit()

    monkeypatch.setattr(self_repair, "verify", verify)
    result = self_repair.run_case(
        case, "bad", initial_messages(case), generate, Path("tools.jar"), tmp_path / "run", 3
    )
    assert result["status"] == "context_limit" and not result["success"]
    assert result["iterations_total"] == 1 and result["repair_calls"] == 0
