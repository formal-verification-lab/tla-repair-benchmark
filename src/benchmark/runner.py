"""Run isolated repair branches using the shared client, verifier and strategies."""

import copy
import json
import re
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from benchmark import SCHEMA_VERSION
from benchmark.contracts import AttemptRecord, CallRecord, CheckStatus, Message, Method
from benchmark.feedback import ExplanationStrategy, FilteredLogStrategy, FullLogStrategy
from benchmark.llm import _write_json
from benchmark.metrics import summarize

SYSTEM_PROMPT = (
    "Write an executable TLA+ module for the supplied task. Return complete TLA+ "
    "source only. The TLC configuration is a separate, fixed file; do not rewrite "
    "or include it in the module."
)
INITIAL_PROMPT = (
    "Module name: {module_name}\nTask:\n{description}\n"
    "Fixed TLC configuration (provided separately):\n{cfg_text}"
)
REPAIR_PROMPT = "Return the corrected complete TLA+ module.\nVerification feedback:\n{text}"
EXTRACTION_VERSION = "single-markdown-fence-v1"


def initial_messages(task):
    return (
        Message("system", SYSTEM_PROMPT),
        Message("user", INITIAL_PROMPT.format(**asdict(task))),
    )


def extract_source(text):
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S)
    return blocks[0] if len(blocks) == 1 else text


def _safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Unsafe artifact identifier")
    return value


class RecordedClient:
    """Add task context to actual calls, including calls made by the explainer.

    Native provider artifacts are preserved. For a contract-only client, request
    and response JSON are reconstructed and explicitly labelled in context.json.
    """

    def __init__(self, root, client, expected_model):
        self.root = Path(root)
        self.client = client
        self.expected_model = expected_model
        self.active = None

    @contextmanager
    def context(self, task_id, repetition, method, purpose):
        if self.active is not None:
            raise RuntimeError("Nested call context is not supported")
        self.active = dict(task_id=task_id, repetition=repetition, method=method, purpose=purpose)
        try:
            yield
        finally:
            self.active = None

    def generate(self, messages, settings):
        if self.active is None:
            raise RuntimeError("Every call requires explicit experimental context")
        before = set(self.root.glob("calls/*"))
        started = time.monotonic()
        try:
            response = self.client.generate(tuple(messages), copy.deepcopy(settings))
        except BaseException as exc:
            new = set(self.root.glob("calls/*")) - before
            if not new:
                directory = self.root / "calls" / uuid4().hex
                directory.mkdir(parents=True, exist_ok=False)
                _write_json(
                    directory / "error.json",
                    {
                        "call_id": directory.name,
                        "error_type": type(exc).__name__,
                        "duration_seconds": time.monotonic() - started,
                        "usage": {"input_tokens": None, "output_tokens": None},
                    },
                )
                new = {directory}
            for directory in new:
                _write_json(directory / "context.json", self.active)
            raise
        directory = self.root / "calls" / _safe_id(response.call_id)
        if directory in before:
            raise ValueError("Client reused a call ID")
        directory.mkdir(parents=True, exist_ok=True)
        native = (directory / "request.json").is_file() and (directory / "response.json").is_file()
        if not (directory / "request.json").exists():
            _write_json(
                directory / "request.json",
                {
                    "messages": [asdict(m) for m in messages],
                    "settings": asdict(settings),
                },
            )
        if not (directory / "response.json").exists():
            _write_json(directory / "response.json", response.raw_response)
        _write_json(
            directory / "context.json",
            {
                **self.active,
                "artifact_source": "provider" if native else "returned_object",
            },
        )
        record = CallRecord(
            response.call_id,
            self.active["purpose"],
            self.active["task_id"],
            self.active["repetition"],
            self.active["method"],
            str((directory / "request.json").relative_to(self.root)),
            str((directory / "response.json").relative_to(self.root)),
            response.usage,
            response.duration_seconds,
        )
        _write_json(directory / "record.json", asdict(record))
        # The returned call is still counted and retained if model drift is detected.
        if response.model != self.expected_model:
            raise ValueError("Model version changed during the experiment")
        return response


class Runner:
    def __init__(
        self, root, client, verifier, settings, verifier_settings, *, max_attempts=10, repetitions=1
    ):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
            raise ValueError("Total attempts must be 1..10, including the initial response")
        if type(repetitions) is not int or repetitions not in {1, 3}:
            raise ValueError("Use one required repetition or three extension repetitions")
        self.root = Path(root).absolute()
        self.client = client
        self.verifier = verifier
        self.settings = copy.deepcopy(settings)
        self.verifier_settings = verifier_settings
        self.max_attempts = max_attempts
        self.repetitions = repetitions

    def _write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, value)

    def _verify(self, task, response, directory, cfg):
        directory.mkdir(parents=True, exist_ok=False)
        source = extract_source(response.text)
        spec = directory / f"{task.module_name}.tla"
        with spec.open("x", encoding="utf-8", newline="") as output:
            output.write(source)
        _write_json(
            directory / "extraction.json",
            {
                "version": EXTRACTION_VERSION,
                "changed": source != response.text,
                "empty": not source.strip(),
                "generation_call_id": response.call_id,
            },
        )
        check = self.verifier.verify(spec, cfg, self.verifier_settings, directory / "check")
        _write_json(directory / "verification.json", asdict(check))
        return (
            source,
            check,
            str(spec.relative_to(self.root)),
            str((directory / "verification.json").relative_to(self.root)),
        )

    @staticmethod
    def _technical(check):
        return any(
            r.status in {CheckStatus.TIMEOUT, CheckStatus.TOOL_ERROR}
            for r in (check.sany, check.tlc)
        )

    def run(self, tasks, manifest, *, snapshot_files=None):
        tasks = tuple(tasks)
        if not tasks or len({_safe_id(t.task_id) for t in tasks}) != len(tasks):
            raise ValueError("Provide nonempty tasks with unique safe IDs")
        for task in tasks:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", task.module_name):
                raise ValueError("Invalid TLA+ module name")
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("phase") not in {
            "pilot",
            "main",
        }:
            raise ValueError("Provide a versioned pilot or main manifest")
        if (
            manifest["protocol"]["max_attempts"] != self.max_attempts
            or manifest["protocol"]["repetitions"] != self.repetitions
        ):
            raise ValueError("Manifest and actual attempt/repetition limits differ")
        if manifest["model"]["provider"] != self.settings.provider:
            raise ValueError("Manifest and actual provider differ")
        if "verifier" in manifest:
            actual = asdict(self.verifier_settings)
            actual["tlc_arguments"] = list(actual["tlc_arguments"])
            if any(
                actual[key] != value
                for key, value in manifest["verifier"].items()
                if key != "jar_path"
            ):
                raise ValueError("Manifest and actual verifier settings differ")
        if manifest["phase"] == "main" and {t.task_id for t in tasks} != set(
            manifest["tasks"]["inputs"]
        ):
            raise ValueError("Main series must execute all four selected tasks")
        if manifest["model"]["requested_id"] != self.settings.model or manifest["model"][
            "parameters"
        ] != dict(self.settings.parameters):
            raise ValueError("Manifest and actual model settings differ")
        self.root.mkdir(parents=True, exist_ok=False)
        _write_json(self.root / "manifest.json", manifest)
        for relative, data in (snapshot_files or {}).items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Snapshot paths must stay within the experiment")
            destination = self.root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                output.write(data)
        if manifest["phase"] == "main":
            from benchmark.validation import validate_snapshot

            validation = validate_snapshot(self.root)
            _write_json(self.root / "protocol-validation.json", validation)
            if not validation["snapshot_valid"]:
                raise ValueError("Frozen protocol validation failed before any model call")
        for task in tasks:
            entry = manifest.get("tasks", {}).get("inputs", {}).get(task.task_id)
            if entry:
                for field, text in (("description", task.description), ("cfg", task.cfg_text)):
                    relative = Path(entry[field + "_path"])
                    if (
                        relative.is_absolute()
                        or ".." in relative.parts
                        or any(p in {"reference", "gold-check"} for p in relative.parts)
                    ):
                        raise ValueError("Forbidden task input path in snapshot")
                    path = (self.root / relative).resolve(strict=True)
                    if (
                        not path.is_relative_to(self.root.resolve())
                        or path.read_bytes() != text.encode()
                    ):
                        raise ValueError("Actual task input differs from archived protocol")
        recorded = RecordedClient(self.root, self.client, manifest["model"]["resolved_id"])
        strategies = {
            Method.FULL_LOG: FullLogStrategy(self.root),
            Method.FILTERED_LOG: FilteredLogStrategy(self.root),
            Method.LLM_EXPLANATION: ExplanationStrategy(self.root, recorded, self.settings),
        }
        outcomes = []
        started = time.monotonic()
        try:
            for task in tasks:
                for repetition in range(1, self.repetitions + 1):
                    group = Path("tasks") / task.task_id / f"repeat-{repetition:02d}"
                    cfg = self.root / group / "Spec.cfg"
                    cfg.parent.mkdir(parents=True)
                    with cfg.open("x", encoding="utf-8", newline="") as output:
                        output.write(task.cfg_text)
                    initial = initial_messages(task)
                    with recorded.context(task.task_id, repetition, None, "initial"):
                        first = recorded.generate(initial, self.settings)
                    shared = self._verify(task, first, self.root / group / "initial", cfg)
                    # Materialize all three initial references before any repair.
                    for method in Method:
                        self._attempt(group, task, repetition, method, 1, first, shared, None)
                    for method in Method:
                        source, check, _, _ = shared
                        history = initial + (Message("assistant", first.text),)
                        iteration = 1
                        while (
                            not check.success
                            and not self._technical(check)
                            and iteration < self.max_attempts
                        ):
                            iteration += 1
                            directory = (
                                self.root / group / method.value / f"attempt-{iteration:02d}"
                            )
                            directory.mkdir(parents=True, exist_ok=False)
                            with recorded.context(task.task_id, repetition, method, "explanation"):
                                feedback = strategies[method].build(source, check)
                            feedback_path = str(
                                (directory / "feedback.json").relative_to(self.root)
                            )
                            _write_json(directory / "feedback.json", asdict(feedback))
                            history += (Message("user", REPAIR_PROMPT.format(text=feedback.text)),)
                            with recorded.context(task.task_id, repetition, method, "repair"):
                                response = recorded.generate(history, self.settings)
                            version = self._verify(task, response, directory / "generated", cfg)
                            self._attempt(
                                group,
                                task,
                                repetition,
                                method,
                                iteration,
                                response,
                                version,
                                feedback_path,
                            )
                            source, check, _, _ = version
                            history += (Message("assistant", response.text),)
                        status = (
                            "success"
                            if check.success
                            else "verification_interruption"
                            if self._technical(check)
                            else "attempt_limit"
                        )
                        outcome = {
                            "task_id": task.task_id,
                            "repetition": repetition,
                            "method": method.value,
                            "iterations_total": iteration,
                            "status": status,
                        }
                        self._write(group / method.value / "completion.json", outcome)
                        outcomes.append(outcome)
                        print(json.dumps(outcome), flush=True)
            metrics = summarize(self.root)
            self._write("metrics.json", metrics)
            self._write(
                "complete.json",
                {"branches": outcomes, "duration_seconds": time.monotonic() - started},
            )
            return metrics
        except BaseException as exc:
            self._write(
                "failure.json",
                {"error_type": type(exc).__name__, "duration_seconds": time.monotonic() - started},
            )
            raise

    def _attempt(self, group, task, repetition, method, iteration, response, version, feedback):
        record = AttemptRecord(
            task.task_id,
            repetition,
            method,
            iteration,
            response.call_id,
            version[2],
            version[3],
            feedback,
        )
        self._write(
            group / method.value / f"attempt-{iteration:02d}" / "record.json", asdict(record)
        )
