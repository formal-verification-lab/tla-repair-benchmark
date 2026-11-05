"""Materialize pinned TLA+-Bench tasks locally and validate Gold offline.

Only load_inputs is used by the experimental runner. It reads descriptions and
cfg files; it never opens a reference specification or derives hints from one.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from urllib.request import urlopen

from benchmark.contracts import TaskInput, TaskMetadata, VerifierSettings
from benchmark.sft_data import dump, sha
from benchmark.verifier import JavaVerifier

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTION = ROOT / "configs/tasks.json"


def read_selection(path: Path) -> dict:
    selection = json.loads(path.read_text())
    if selection["repository"] != "https://github.com/LUC-AI4FM/tla_benchmark":
        raise ValueError("Unexpected dataset repository")
    if not re.fullmatch(r"[a-f0-9]{40}", selection["commit"]):
        raise ValueError("Pin a full dataset commit SHA")
    tasks = selection["tasks"]
    if Counter(t["difficulty"] for t in tasks) != {"easy": 2, "medium": 2}:
        raise ValueError("Select two easy and two medium tasks")
    if len({t["task_id"] for t in tasks}) != 4 or len({t["spec_id"] for t in tasks}) != 4:
        raise ValueError("Task IDs and source IDs must be unique")
    for task in tasks:
        if task["tier"] != "gold" or task["category"] != "system":
            raise ValueError("Use model-checkable system tasks")
        if selection["difficulty_mapping"][task["source_complexity"]] != task["difficulty"]:
            raise ValueError("Difficulty must match the original label mapping")
        if not re.fullmatch(r"[a-z0-9_]+", task["task_id"]) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z_0-9]*", task["module_name"]
        ):
            raise ValueError("Unsafe task or module name")
    return selection


def _checked(data: bytes, expected: str) -> bytes:
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Source or local task checksum mismatch")
    return data


def fetch(selection: dict, path: str, expected: str) -> bytes:
    if not path.startswith("dataset_release/") or ".." in Path(path).parts:
        raise ValueError("Unexpected dataset path")
    url = f"https://raw.githubusercontent.com/LUC-AI4FM/tla_benchmark/{selection['commit']}/{path}"
    with urlopen(url, timeout=60) as response:
        return _checked(response.read(), expected)


def load_inputs(selection: dict, directory: Path) -> tuple[TaskInput, ...]:
    """Allowlisted input boundary: no traversal of reference/ or Gold reads."""
    receipt = json.loads((directory / "prepared.json").read_text())
    expected = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    if (
        receipt["selection_sha256"] != expected
        or not all(task["sany"] and task["tlc"] for task in receipt["tasks"])
        or len(receipt["tasks"]) != 4
        or [task["task_id"] for task in receipt["tasks"]]
        != [task["task_id"] for task in selection["tasks"]]
    ):
        raise ValueError("A successful preparation receipt for this selection is required")
    return _read_inputs(selection, directory)


def _read_inputs(selection: dict, directory: Path) -> tuple[TaskInput, ...]:
    inputs = []
    for task in selection["tasks"]:
        base = directory / task["task_id"]
        description = _checked((base / "description.md").read_bytes(), task["description_sha256"])
        cfg = _checked((base / "Spec.cfg").read_bytes(), task["cfg_sha256"])
        inputs.append(
            TaskInput(task["task_id"], task["module_name"], description.decode(), cfg.decode())
        )
    return tuple(inputs)


def inspect_prepared(selection: dict, directory: Path) -> dict:
    """Offline audit including Gold; this function is not a runner input path."""
    prepared = json.loads((directory / "prepared.json").read_text())
    encoded = json.dumps(selection, sort_keys=True).encode()
    if prepared["selection_sha256"] != hashlib.sha256(encoded).hexdigest():
        raise ValueError("Task selection changed after preparation")
    files = {
        str(p.relative_to(directory)): sha(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p != directory / "prepared.json"
    }
    if files != prepared["files"]:
        raise ValueError("Prepared task data or evidence changed")
    load_inputs(selection, directory)
    return prepared


def prepare(selection_path: Path, directory: Path, jar: Path) -> dict:
    selection = read_selection(selection_path)
    if sha(jar) != selection["jar_sha256"]:
        raise ValueError("Use the pinned official TLA+ tools JAR")
    if directory.exists():
        return inspect_prepared(selection, directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Build in a sibling directory. Failed preparation never becomes a usable
    # task bundle; its logs are retained under a distinct failed directory.
    staging = Path(tempfile.mkdtemp(prefix=".tasks-staging-", dir=directory.parent))
    try:
        manifest = fetch(selection, selection["manifest_path"], selection["manifest_sha256"])
        readme = fetch(selection, selection["readme_path"], selection["readme_sha256"])
        (staging / "source-manifest.jsonl").write_bytes(manifest)
        (staging / "SOURCE-README.md").write_bytes(readme)
        rows = {row["spec_id"]: row for row in map(json.loads, manifest.decode().splitlines())}
        summaries = []
        for task in selection["tasks"]:
            row = rows[task["spec_id"]]
            for field in ("tier", "category", "source_path", "source_repo_url"):
                if row[field] != task[field]:
                    raise ValueError(f"Source provenance differs: {task['task_id']}:{field}")
            if row["complexity"] != task["source_complexity"]:
                raise ValueError("Source difficulty differs")
            description = _checked(
                row[selection["description_field"]].encode(), task["description_sha256"]
            )
            gold = fetch(selection, task["gold_path"], task["gold_sha256"])
            if not task["gold_sha256"].startswith(row["spec_sha256"]):
                raise ValueError("Gold differs from the source manifest")
            cfg = fetch(selection, task["cfg_path"], task["cfg_sha256"])
            base = staging / task["task_id"]
            reference = base / "reference"
            reference.mkdir(parents=True)
            (reference / f"{task['module_name']}.tla").write_bytes(gold)
            (base / "description.md").write_bytes(description)
            (base / "Spec.cfg").write_bytes(cfg)
            metadata = TaskMetadata(
                task["task_id"],
                task["difficulty"],
                f"{selection['repository']}/blob/{selection['commit']}/{task['gold_path']}",
                selection["commit"],
                task["gold_path"],
                task["description_sha256"],
                task["cfg_sha256"],
                task["gold_sha256"],
            )
            dump(base / "metadata.json", asdict(metadata))
            verification = JavaVerifier().verify(
                reference / f"{task['module_name']}.tla",
                base / "Spec.cfg",
                VerifierSettings(jar, selection["jar_sha256"], "java", 30, 60),
                base / "gold-check",
            )
            states_match = re.search(r"([\d,]+) distinct states found", verification.tlc.stdout)
            states = int(states_match[1].replace(",", "")) if states_match else None
            if not verification.success or states != task["expected_states"]:
                raise ValueError(f"Gold/config validation failed: {task['task_id']}")
            summary = {
                "task_id": task["task_id"],
                "module": task["module_name"],
                "difficulty": task["difficulty"],
                "states": states,
                "sany": True,
                "tlc": True,
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        _read_inputs(selection, staging)
        (staging / "selection.json").write_bytes(selection_path.read_bytes())
        (staging / "java-version.txt").write_bytes(
            subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)
        )
        report = {
            "selection_sha256": hashlib.sha256(
                json.dumps(selection, sort_keys=True).encode()
            ).hexdigest(),
            "jar_sha256": sha(jar),
            "tasks": summaries,
            "checker": "benchmark.verifier.JavaVerifier",
            "checker_source_sha256": sha(Path(__file__).with_name("verifier.py")),
            "preparer_source_sha256": sha(Path(__file__)),
            "files": {
                str(p.relative_to(staging)): sha(p)
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            },
            "redistribution": (
                "Source dataset README restricts raw redistribution; this bundle stays local."
            ),
        }
        dump(staging / "prepared.json", report)
        load_inputs(selection, staging)
        os.rename(staging, directory)
        return report
    except BaseException as exc:
        dump(staging / "failure.json", {"error_type": type(exc).__name__, "error": str(exc)})
        failed = directory.parent / staging.name.replace(".tasks-staging-", "tasks-failed-")
        staging.rename(failed)
        print(f"Incomplete preparation evidence retained at {failed}", flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=ROOT / "data/tasks")
    parser.add_argument("--jar", type=Path, required=True)
    args = parser.parse_args()
    # Both operations use local paths; no LLM or credential access is performed.
    result = prepare(args.selection.resolve(), args.output.resolve(), args.jar.resolve())
    print(json.dumps({"output": str(args.output), "tasks": result["tasks"]}), flush=True)


if __name__ == "__main__":
    main()
