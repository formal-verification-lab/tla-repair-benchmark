import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmark import task_dataset
from benchmark.sft_data import sha


def test_selection_has_original_gold_system_difficulties():
    selection = task_dataset.read_selection(task_dataset.DEFAULT_SELECTION)
    assert [
        (t["module_name"], t["source_complexity"], t["difficulty"]) for t in selection["tasks"]
    ] == [
        ("HourClock", "basic", "easy"),
        ("AsynchInterface", "basic", "easy"),
        ("Barrier", "intermediate", "medium"),
        ("CigaretteSmokers", "intermediate", "medium"),
    ]


@pytest.fixture
def bundle(tmp_path):
    selection = task_dataset.read_selection(task_dataset.DEFAULT_SELECTION)
    root = tmp_path / "tasks"
    for task in selection["tasks"]:
        base = root / task["task_id"]
        (base / "reference").mkdir(parents=True)
        (base / "reference" / f"{task['module_name']}.tla").write_text("SECRET_GOLD")
        (base / "description.md").write_text(f"Task {task['task_id']}")
        (base / "Spec.cfg").write_text("SPECIFICATION Spec\n")
        task["description_sha256"] = sha(base / "description.md")
        task["cfg_sha256"] = sha(base / "Spec.cfg")
    report = {
        "selection_sha256": hashlib.sha256(
            json.dumps(selection, sort_keys=True).encode()
        ).hexdigest(),
        "tasks": [{"task_id": t["task_id"], "sany": True, "tlc": True} for t in selection["tasks"]],
        "files": {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()},
    }
    (root / "prepared.json").write_text(json.dumps(report))
    return selection, root


def test_runner_inputs_never_read_gold_or_expose_its_path(bundle, monkeypatch):
    selection, root = bundle
    original = Path.open

    def guard(path, *args, **kwargs):
        if "reference" in path.parts:
            raise AssertionError("Gold read while forming runner inputs")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    tasks = task_dataset.load_inputs(selection, root)
    assert len(tasks) == 4
    payload = json.dumps([asdict(t) for t in tasks])
    assert "SECRET_GOLD" not in payload and "reference" not in payload


def test_changed_description_is_rejected_before_generation(bundle):
    selection, root = bundle
    (root / "easy_01/description.md").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        task_dataset.load_inputs(selection, root)


def test_offline_audit_detects_changed_gold_and_preserves_original_receipt(bundle):
    selection, root = bundle
    task_dataset.inspect_prepared(selection, root)
    gold = next((root / "easy_01/reference").iterdir())
    gold.write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        task_dataset.inspect_prepared(selection, root)


def test_missing_preparation_receipt_blocks_runner_input(bundle):
    selection, root = bundle
    (root / "prepared.json").unlink()
    with pytest.raises(FileNotFoundError):
        task_dataset.load_inputs(selection, root)


def test_failed_gold_receipt_blocks_runner_input(bundle):
    selection, root = bundle
    path = root / "prepared.json"
    receipt = json.loads(path.read_text())
    receipt["tasks"][0]["tlc"] = False
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="successful preparation receipt"):
        task_dataset.load_inputs(selection, root)


def test_failed_download_keeps_diagnostics_without_publishing_bundle(tmp_path, monkeypatch):
    selection = task_dataset.read_selection(task_dataset.DEFAULT_SELECTION)
    jar = tmp_path / "tools.jar"
    jar.write_bytes(b"test jar")
    selection["jar_sha256"] = sha(jar)
    config = tmp_path / "selection.json"
    config.write_text(json.dumps(selection))

    def fail(*args):
        raise OSError("Network unavailable")

    monkeypatch.setattr(task_dataset, "fetch", fail)
    output = tmp_path / "data/tasks"
    with pytest.raises(OSError, match="Network unavailable"):
        task_dataset.prepare(config, output, jar)
    assert not output.exists()
    failed = list(output.parent.glob("tasks-failed-*/failure.json"))
    assert len(failed) == 1
    assert json.loads(failed[0].read_text())["error_type"] == "OSError"


@pytest.fixture
def source_bundle(tmp_path, monkeypatch):
    """Synthetic download fixture; these tests do not claim to execute Java."""
    selection = task_dataset.read_selection(task_dataset.DEFAULT_SELECTION)
    sources, rows = {}, []
    for task in selection["tasks"]:
        gold = f"---- MODULE {task['module_name']} ----\n====\n".encode()
        cfg = b"SPECIFICATION Spec\n"
        description = f"Synthetic task {task['task_id']}"
        sources[task["gold_path"]] = gold
        sources[task["cfg_path"]] = cfg
        task["gold_sha256"] = hashlib.sha256(gold).hexdigest()
        task["cfg_sha256"] = hashlib.sha256(cfg).hexdigest()
        task["description_sha256"] = hashlib.sha256(description.encode()).hexdigest()
        rows.append(
            {
                **{
                    k: task[k]
                    for k in ("spec_id", "tier", "category", "source_path", "source_repo_url")
                },
                "complexity": task["source_complexity"],
                "spec_sha256": task["gold_sha256"],
                selection["description_field"]: description,
            }
        )
    sources[selection["manifest_path"]] = "\n".join(map(json.dumps, rows)).encode()
    sources[selection["readme_path"]] = b"Synthetic source README"
    jar = tmp_path / "tools.jar"
    jar.write_bytes(b"Synthetic JAR; not executable")
    selection["jar_sha256"] = sha(jar)
    config = tmp_path / "selection.json"
    config.write_text(json.dumps(selection))
    monkeypatch.setattr(task_dataset, "fetch", lambda selection, path, expected: sources[path])
    monkeypatch.setattr(task_dataset.subprocess, "check_output", lambda *a, **k: b"Test Java")
    return selection, config, jar


@pytest.mark.parametrize("failure", [None, "spec_error", "wrong_states"])
def test_preparation_uses_shared_verifier_and_requires_complete_gold_success(
    source_bundle, tmp_path, monkeypatch, failure
):
    from benchmark.contracts import CheckStatus, ToolResult, VerificationResult

    selection, config, jar = source_bundle
    calls = []

    def verify(self, spec_path, cfg_path, settings, work_dir):
        assert spec_path.parent.name == "reference"
        assert cfg_path.name == "Spec.cfg"
        assert settings.jar_sha256 == selection["jar_sha256"]
        task = next(t for t in selection["tasks"] if t["module_name"] == spec_path.stem)
        calls.append(task["task_id"])
        work_dir.mkdir()
        states = 999 if failure == "wrong_states" else task["expected_states"]
        status = CheckStatus.SPEC_ERROR if failure == "spec_error" else CheckStatus.PASSED
        result = VerificationResult(
            ToolResult("sany", CheckStatus.PASSED, (), "", "", 0, 0),
            ToolResult("tlc", status, (), f"{states} distinct states found", "", 0, 0),
        )
        (work_dir / "verification.json").write_text(json.dumps(asdict(result)))
        return result

    monkeypatch.setattr(task_dataset.JavaVerifier, "verify", verify)
    output = tmp_path / "prepared"
    if failure:
        with pytest.raises(ValueError, match="Gold/config validation failed"):
            task_dataset.prepare(config, output, jar)
        assert not output.exists()
        assert len(list(tmp_path.glob("tasks-failed-*/failure.json"))) == 1
        assert calls == ["easy_01"]
    else:
        receipt = task_dataset.prepare(config, output, jar)
        assert calls == [t["task_id"] for t in selection["tasks"]]
        assert receipt["checker"] == "benchmark.verifier.JavaVerifier"
        assert receipt["checker_source_sha256"] == sha(
            Path(task_dataset.__file__).with_name("verifier.py")
        )
        assert receipt["preparer_source_sha256"] == sha(Path(task_dataset.__file__))
        assert len(task_dataset.load_inputs(selection, output)) == 4
        assert task_dataset.prepare(config, output, jar) == receipt
        # Existing preparation is audited without repeating Java checks.
        assert len(calls) == 4
        (output / "medium_02/gold-check/verification.json").write_text("changed")
        with pytest.raises(ValueError, match="evidence changed"):
            task_dataset.inspect_prepared(selection, output)
