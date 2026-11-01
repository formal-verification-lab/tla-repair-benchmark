from dataclasses import asdict

import pytest

from benchmark.cli import main
from benchmark.contracts import (
    AttemptRecord,
    CallRecord,
    CheckStatus,
    Method,
    ToolResult,
    Usage,
    VerificationResult,
)


def tool(name, status):
    return ToolResult(name, status, (), "", "", None, 0)


@pytest.mark.parametrize("status", list(CheckStatus))
def test_only_completed_tlc_can_succeed(status):
    result = VerificationResult(tool("sany", CheckStatus.PASSED), tool("tlc", status))
    assert result.success == (status == CheckStatus.PASSED)


def test_tlc_cannot_run_after_sany_failure():
    with pytest.raises(ValueError, match="TLC must not run"):
        VerificationResult(tool("sany", CheckStatus.SPEC_ERROR), tool("tlc", CheckStatus.PASSED))


@pytest.mark.parametrize("iteration", [0, 11])
def test_attempt_limit(iteration):
    with pytest.raises(ValueError):
        AttemptRecord("easy_01", 1, Method.FULL_LOG, iteration, "initial-1", "a.tla", "v.json")


def test_shared_initial_call_has_one_identity_across_branches():
    call = CallRecord(
        "initial-1", "initial", "easy_01", 1, None, "req.json", "res.json", Usage(9, 7), 1
    )
    attempts = [
        AttemptRecord("easy_01", 1, method, 1, call.call_id, "a.tla", "v.json") for method in Method
    ]
    assert {attempt.generation_call_id for attempt in attempts} == {call.call_id}
    assert asdict(call)["usage"] == {"input_tokens": 9, "output_tokens": 7}


def test_initial_call_cannot_belong_to_one_branch():
    with pytest.raises(ValueError, match="shared initial"):
        CallRecord("id", "initial", "easy_01", 1, Method.FULL_LOG, "r", "s", Usage(), 0)


def test_missing_usage_is_not_zero():
    assert Usage().input_tokens is None
    assert Usage(0, 0).input_tokens == 0
    with pytest.raises(ValueError):
        Usage(-1, 0)


def test_cli_exposes_contract_version(capsys):
    import json

    assert main(["schema-info"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema_version"] == "1.0.0"
    assert set(result["methods"]) == {"A", "B", "C"}


@pytest.mark.parametrize("status", [s for s in CheckStatus if s != CheckStatus.PASSED])
def test_skipped_tlc_does_not_turn_sany_failure_into_success(status):
    result = VerificationResult(tool("sany", status), tool("tlc", CheckStatus.NOT_RUN))
    assert not result.success
