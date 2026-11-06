import json
from pathlib import Path

import pytest
from test_runner import Checker, Client

from benchmark import experiment
from benchmark.contracts import ModelSettings, TaskInput
from benchmark.runner import Runner
from benchmark.validation import validate_snapshot


def frozen(tmp_path, monkeypatch, phase):
    selection = json.loads(experiment.DEFAULT_SELECTION.read_text())
    tasks = []
    for t in selection["tasks"]:
        task = TaskInput(t["task_id"], t["module_name"], "synthetic task", "INIT Init\nNEXT Next\n")
        tasks.append(task)
        t["description_sha256"] = experiment.digest(task.description.encode())
        t["cfg_sha256"] = experiment.digest(task.cfg_text.encode())
    jar = tmp_path / "fixture.jar"
    jar.write_bytes(b"fixture jar, no Java called by this unit test")
    selection["jar_sha256"] = experiment.digest(jar.read_bytes())
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    monkeypatch.setattr(
        experiment.subprocess,
        "check_output",
        lambda cmd, **kwargs: "a" * 40 if cmd[1] == "rev-parse" else b"",
    )
    monkeypatch.setattr(experiment, "_version", lambda *args: "synthetic version")
    config = {
        "model": {
            "provider": "fixture",
            "requested_id": "requested-model",
            "resolved_id": "version-1",
            "parameters": {"max_tokens": 32},
        },
        "repetitions": 1,
        "verifier": {"sany_timeout_seconds": 30, "tlc_timeout_seconds": 60, "tlc_arguments": []},
        "api_timeout_seconds": 120,
    }
    manifest, files = experiment.snapshot(
        config, path, tasks, jar, phase, [t.task_id for t in tasks], 10
    )
    return tasks, manifest, files


def test_main_snapshot_generated_by_runner_passes_colleague_validator(tmp_path, monkeypatch):
    from benchmark.contracts import VerifierSettings

    tasks, manifest, files = frozen(tmp_path, monkeypatch, "main")
    root = tmp_path / "main"
    runner = Runner(
        root,
        Client(initial="repaired"),
        Checker(),
        ModelSettings("fixture", "requested-model", {"max_tokens": 32}),
        VerifierSettings(Path("unused.jar"), "unused", "java", 30, 60),
    )
    result = runner.run(tasks, manifest, snapshot_files=files)
    assert validate_snapshot(root)["snapshot_valid"]
    assert len(result["branches"]) == 12


def test_invalid_main_snapshot_stops_before_model_calls(tmp_path, monkeypatch):
    from benchmark.contracts import VerifierSettings

    tasks, manifest, files = frozen(tmp_path, monkeypatch, "main")
    files["prompts/explanation.txt"] = b"changed prompt"
    client = Client()
    runner = Runner(
        tmp_path / "main",
        client,
        Checker(),
        ModelSettings("fixture", "requested-model", {"max_tokens": 32}),
        VerifierSettings(Path("unused.jar"), "unused", "java", 30, 60),
    )
    with pytest.raises(ValueError, match="protocol validation failed"):
        runner.run(tasks, manifest, snapshot_files=files)
    assert not client.calls


def test_main_cli_requires_completed_pilot_before_network(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        experiment.main(["--phase", "main", "--dataset", str(tmp_path)])
    assert exc.value.code == 1
    assert "--pilot" in capsys.readouterr().err
