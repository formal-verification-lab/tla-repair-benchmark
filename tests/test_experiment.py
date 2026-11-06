import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from test_runner import Checker, Client

from benchmark import experiment
from benchmark.contracts import ModelSettings, TaskInput
from benchmark.runner import Runner
from benchmark.validation import validate_snapshot


def verifier_settings(manifest):
    from benchmark.contracts import VerifierSettings

    fields = {
        **manifest["verifier"],
        "jar_path": Path("unused.jar"),
        "tlc_arguments": tuple(manifest["verifier"]["tlc_arguments"]),
    }
    return VerifierSettings(**fields)


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

    tasks, manifest, files = frozen(tmp_path, monkeypatch, "main")
    root = tmp_path / "main"
    runner = Runner(
        root,
        Client(initial="repaired"),
        Checker(),
        ModelSettings("fixture", "requested-model", {"max_tokens": 32}),
        verifier_settings(manifest),
    )
    result = runner.run(tasks, manifest, snapshot_files=files)
    assert validate_snapshot(root)["snapshot_valid"]
    assert len(result["branches"]) == 12


def test_invalid_main_snapshot_stops_before_model_calls(tmp_path, monkeypatch):

    tasks, manifest, files = frozen(tmp_path, monkeypatch, "main")
    files["prompts/explanation.txt"] = b"changed prompt"
    client = Client()
    runner = Runner(
        tmp_path / "main",
        client,
        Checker(),
        ModelSettings("fixture", "requested-model", {"max_tokens": 32}),
        verifier_settings(manifest),
    )
    with pytest.raises(ValueError, match="protocol validation failed"):
        runner.run(tasks, manifest, snapshot_files=files)
    assert not client.calls


def test_main_cli_requires_completed_pilot_before_network(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        experiment.main(["--phase", "main", "--dataset", str(tmp_path)])
    assert exc.value.code == 1
    assert "--pilot" in capsys.readouterr().err


@pytest.mark.parametrize("mismatch", ["input", "verifier"])
def test_main_rejects_actual_values_different_from_valid_snapshot(tmp_path, monkeypatch, mismatch):
    tasks, manifest, files = frozen(tmp_path, monkeypatch, "main")
    client = Client()
    settings = verifier_settings(manifest)
    if mismatch == "input":
        tasks[0] = replace(tasks[0], description="different actual prompt")
    else:
        settings = replace(settings, sany_timeout_seconds=1)
    runner = Runner(
        tmp_path / "main",
        client,
        Checker(),
        ModelSettings("fixture", "requested-model", {"max_tokens": 32}),
        settings,
    )
    with pytest.raises(ValueError, match="differs|differ"):
        runner.run(tasks, manifest, snapshot_files=files)
    assert not client.calls


def test_tlc_help_version_accepts_documented_exit_one_but_rejects_launch_errors(monkeypatch):
    result = subprocess.CompletedProcess(
        ["java"],
        1,
        b"TLC - provides model checking and simulation of TLA+ specifications - Version fixture\n",
        b"",
    )
    monkeypatch.setattr(experiment.subprocess, "run", lambda *args, **kwargs: result)
    assert "Version fixture" in experiment._version(["java", "tlc2.TLC", "-help"])
    with pytest.raises(subprocess.CalledProcessError):
        experiment._version(["java", "-version"])
    result.stdout = b"Error: Unable to access jarfile"
    with pytest.raises(subprocess.CalledProcessError):
        experiment._version(["java", "tlc2.TLC", "-help"])
