import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmark import SCHEMA_VERSION
from benchmark.feedback import EXPLANATION_PROMPT, EXPLANATION_VERSION, FILTER_VERSION
from benchmark.task_dataset import DEFAULT_SELECTION
from benchmark.validation import PROFILE, main, validate_snapshot


def save(root, path, data):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def snapshot(tmp_path):
    # Synthetic metadata tests snapshot validation, not a real completed pilot.
    root = tmp_path / "snapshot"
    root.mkdir()
    selection = json.loads(DEFAULT_SELECTION.read_text())
    inputs = {}
    for task in selection["tasks"]:
        entry = {}
        for kind in ("description", "cfg"):
            path = f"tasks/{task['task_id']}/{kind}.txt"
            digest = save(root, path, f"synthetic {task['task_id']} {kind}".encode())
            task[kind + "_sha256"] = digest
            entry[kind + "_path"] = path
        inputs[task["task_id"]] = entry
    jar_hash = save(root, "tools/tools.jar", b"synthetic JAR fixture; never executed")
    selection["jar_sha256"] = jar_hash
    selection_hash = save(root, "tasks/selection.json", json.dumps(selection).encode())
    source_hash = save(root, "code/runner.py", b"# synthetic source; not a real runner\n")
    prompts = {}
    for name in ("system", "initial", "repair", "explanation"):
        prompt = EXPLANATION_PROMPT if name == "explanation" else f"synthetic {name} prompt"
        path = f"prompts/{name}.txt"
        prompts[name] = {"path": path, "sha256": save(root, path, prompt.encode())}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "validation_profile": PROFILE,
        "phase": "pilot",
        "experiment_id": "synthetic-snapshot",
        "created_at_utc": "2026-11-06T00:00:00+00:00",
        "code": {
            "revision": "a" * 40,
            "dirty": False,
            "runner_path": "code/runner.py",
            "source_files": {"code/runner.py": source_hash},
        },
        "model": {
            "provider": "test",
            "requested_id": "test-model",
            "resolved_id": "test-revision",
            "parameters": {"temperature": 0.1, "max_tokens": 512},
        },
        "environment": {
            "python_version": "test-python",
            "java_version": "test-java",
            "tools_version": "test-tools",
        },
        "verifier": {
            "jar_path": "tools/tools.jar",
            "jar_sha256": jar_hash,
            "java_executable": "java",
            "sany_timeout_seconds": 30,
            "tlc_timeout_seconds": 60,
            "tlc_arguments": [],
        },
        "protocol": {
            "methods": ["A", "B", "C"],
            "max_attempts": 10,
            "initial_attempt_number": 1,
            "shared_initial": True,
            "isolated_histories": True,
            "gold_in_prompts": False,
            "explanation_context": "fresh_specification_and_log_only",
            "explanation_model": "same_as_main",
            "explanation_counts_as_main_attempt": False,
            "code_response_policy": "forward_verbatim_and_flag",
            "empty_response_policy": "forward_verbatim_and_flag",
            "repetitions": 1,
            "api_failure_policy": "synthetic stop policy",
            "verification_failure_policy": "synthetic stop policy",
            "context_overflow_policy": "synthetic stop policy",
            "filter_version": FILTER_VERSION,
            "explanation_version": EXPLANATION_VERSION,
        },
        "prompts": prompts,
        "tasks": {
            "selection_path": "tasks/selection.json",
            "selection_sha256": selection_hash,
            "inputs": inputs,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, manifest


def test_valid_snapshot_is_not_main_readiness_certificate(snapshot):
    root, _ = snapshot
    report = validate_snapshot(root)
    assert report["snapshot_valid"]
    assert report["main_readiness"] == "not_assessed"
    assert report["errors"] == []
    assert "manifest.json" in report["source_sha256"]
    assert "prompts/explanation.txt" in report["source_sha256"]


@pytest.mark.parametrize(
    "group,key,value,message",
    [
        ("code", "dirty", True, "clean code"),
        ("code", "revision", "main", "full commit SHA"),
        ("code", "runner_path", None, "Runner implementation"),
        ("model", "resolved_id", None, "resolved_id"),
        ("environment", "java_version", None, "java_version"),
        ("protocol", "max_attempts", 11, "max_attempts"),
        ("protocol", "shared_initial", False, "shared_initial"),
        ("protocol", "gold_in_prompts", True, "gold_in_prompts"),
        ("protocol", "explanation_model", "different-model", "explanation_model"),
        ("protocol", "api_failure_policy", "", "api_failure_policy"),
        ("protocol", "filter_version", "unknown", "Filter version"),
        ("verifier", "tlc_arguments", ["-deadlock", "1"], "Unsupported TLC"),
        ("verifier", "tlc_timeout_seconds", float("nan"), "tlc_timeout_seconds"),
        ("verifier", "sany_timeout_seconds", True, "sany_timeout_seconds"),
    ],
)
def test_incomplete_or_changed_protocol_is_rejected(snapshot, group, key, value, message):
    root, manifest = snapshot
    manifest[group][key] = value
    (root / "manifest.json").write_text(json.dumps(manifest))
    report = validate_snapshot(root)
    assert not report["snapshot_valid"]
    assert any(message in error for error in report["errors"])


def test_changed_task_input_and_prompt_are_rejected(snapshot):
    root, manifest = snapshot
    (root / manifest["tasks"]["inputs"]["easy_02"]["cfg_path"]).write_text("changed")
    (root / "prompts/initial.txt").write_text("changed")
    report = validate_snapshot(root)
    assert not report["snapshot_valid"]
    assert sum("SHA-256 mismatch" in error for error in report["errors"]) == 2


def test_explanation_prompt_must_match_implementation_even_with_matching_hash(snapshot):
    root, manifest = snapshot
    digest = save(root, "prompts/explanation.txt", b"other explanation prompt")
    manifest["prompts"]["explanation"]["sha256"] = digest
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert "Explanation prompt differs from implementation" in validate_snapshot(root)["errors"]


def test_symlink_and_gold_paths_are_rejected_before_reading(snapshot, tmp_path, monkeypatch):
    root, manifest = snapshot
    outside = tmp_path / "outside.txt"
    outside.write_text("must not be read")
    (root / "external.txt").symlink_to(outside)
    manifest["prompts"]["initial"]["path"] = "external.txt"
    manifest["prompts"]["repair"]["path"] = "tasks/easy_02/reference/Gold.tla"
    (root / "manifest.json").write_text(json.dumps(manifest))
    original = Path.read_bytes

    def guard(path):
        assert path != outside and "reference" not in path.parts
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    report = validate_snapshot(root)
    assert any("external snapshot" in error for error in report["errors"])
    assert any("Forbidden snapshot path" in error for error in report["errors"])


def test_missing_manifest_and_draft_report_failure_without_overwriting(tmp_path):
    output = tmp_path / "report.json"
    assert main(["--snapshot", str(tmp_path), "--output", str(output)]) == 1
    with pytest.raises(FileExistsError):
        main(["--snapshot", str(tmp_path), "--output", str(output)])


def test_unknown_manifest_fields_are_not_silently_used_as_policies(snapshot):
    root, manifest = snapshot
    draft = copy.deepcopy(manifest)
    del draft["protocol"]["context_overflow_policy"]
    draft["overflow_policy"] = "some other field"
    (root / "manifest.json").write_text(json.dumps(draft))
    report = validate_snapshot(root)
    assert any("context_overflow_policy" in error for error in report["errors"])


def test_repository_example_is_intentionally_incomplete():
    root = Path(__file__).resolve().parents[1]
    example = json.loads((root / "configs/protocol-snapshot.example.json").read_text())
    assert example["code"]["runner_path"] is None
    assert example["model"]["resolved_id"] is None
    assert example["protocol"]["api_failure_policy"] is None
    assert example["code"]["source_files"] == {}


@pytest.mark.parametrize("stamp", [None, "2026-11-06", "2026-11-06T12:00:00+03:00"])
def test_snapshot_timestamp_must_be_utc(snapshot, stamp):
    root, manifest = snapshot
    manifest["created_at_utc"] = stamp
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert any("created_at_utc" in error for error in validate_snapshot(root)["errors"])
