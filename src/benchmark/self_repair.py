"""Bounded full-log repair of the adapter's own saved synthetic generations."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from benchmark.finetune import ROOT, config_hash, read, resolve, validate
from benchmark.sft_data import Case, dump, full_log, initial_messages, sha, verify

REPAIR_REQUEST = "Return the corrected complete module.\n"


class ContextLimit(Exception):
    pass


def source_from_response(text: str) -> str:
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S)
    return blocks[0] if len(blocks) == 1 else text


def run_case(case, response, messages, generate, jar, output, max_attempts):
    """All feedback comes from tools; reference answers never enter the history."""
    output.mkdir(parents=True, exist_ok=False)
    history = [dict(message) for message in messages]
    attempts = []
    status = "attempt_limit"
    for number in range(1, max_attempts + 1):
        directory = output / f"attempt-{number:02d}"
        directory.mkdir()
        dump(directory / "messages.json", history)
        if number > 1:
            try:
                response = generate(history, directory)
            except ContextLimit:
                status = "context_limit"
                break
        else:
            dump(directory / "generation.json", {"reused_initial_response": True})
        (directory / "response.txt").write_text(response)
        checks = verify(
            source_from_response(response), case.cfg, case.name, jar, directory / "checks"
        )
        row = {
            "attempt": number,
            "sany": checks["sany"]["success"],
            "tlc": checks.get("tlc", {}).get("success", False),
            "expected_state_count": checks.get("tlc", {}).get("distinct_states")
            == case.expected_states,
        }
        attempts.append(row)
        print(json.dumps({"task": case.name, **row}), flush=True)
        if any(result["timeout"] for result in checks.values()):
            status = "verifier_timeout"
            break
        if row["tlc"]:
            status = "tool_success"
            break
        if number < max_attempts:
            feedback = REPAIR_REQUEST + full_log(directory / "checks")
            (directory / "feedback.txt").write_text(feedback)
            history.extend(
                [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": feedback},
                ]
            )
    result = {
        "task": case.name,
        "status": status,
        "attempts": attempts,
        "iterations_total": len(attempts),
        "repair_calls": len(attempts) - 1,
        "success": bool(attempts and attempts[-1]["tlc"]),
    }
    dump(output / "summary.json", result)
    return result


def start(training_run: Path, max_attempts: int) -> Path:
    if not 1 <= max_attempts <= 10:
        raise ValueError("Total attempts, including the saved initial generation, must be 1..10")
    training_run = training_run.resolve()
    manifest = read(training_run / "manifest.json")
    config = manifest["config"]
    validate(config)
    if config_hash(config) != config_hash(read(ROOT / "configs/lora-pilot.json")):
        raise ValueError("Use the original training configuration")
    complete = read(training_run / "complete.json")
    adapter = training_run / "adapter"
    if complete["smoke"] or sha(adapter / "adapters.safetensors") != complete["adapter_sha256"]:
        raise ValueError("A complete, unchanged training adapter is required")
    if sha(resolve(config["dataset_dir"]) / "manifest.json") != manifest["dataset_manifest_sha256"]:
        raise ValueError("Training and evaluation datasets differ")
    examples = [
        Case(**c)
        for c in read(resolve(config["dataset_dir"]) / "cases.json")
        if c["split"] == "test"
    ]
    inputs = []
    for case in examples:
        initial = training_run / "adapted" / case.name / "generation"
        messages = read(initial / "messages.json")
        if messages != initial_messages(case):
            raise ValueError("Saved initial prompt differs from the sealed test task")
        inputs.append((case, initial, messages, (initial / "response.txt").read_text()))

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "runs" / f"self-repair-{stamp}-{uuid4().hex[:6]}"
    output.mkdir()
    dump(
        output / "manifest.json",
        {
            "phase": "synthetic_self_repair_pilot",
            "method": "full_log",
            "training_run": str(training_run),
            "training_manifest_sha256": sha(training_run / "manifest.json"),
            "adapter_sha256": complete["adapter_sha256"],
            "config": config,
            "max_attempts": max_attempts,
            "initial_generation_counts_as_attempt": True,
            "max_output_tokens": 512,
            "context_window": 8192,
            "decoding": "greedy",
            "inputs": {
                case.name: {
                    "response_sha256": sha(initial / "response.txt"),
                    "messages_sha256": sha(initial / "messages.json"),
                }
                for case, initial, _, _ in inputs
            },
            "source_files": {
                name: sha(ROOT / name)
                for name in (
                    "src/benchmark/self_repair.py",
                    "src/benchmark/sft_data.py",
                    "src/benchmark/finetune.py",
                )
            },
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "limitations": (
                "Four parameterized synthetic cases in two families; tool success and "
                "state counts do not prove semantic equivalence. "
                "This is not the coursework benchmark."
            ),
        },
    )
    (output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    print(f"Run directory: {output}", flush=True)
    started = time.monotonic()
    try:
        model, tokenizer = load(str(resolve(config["model_dir"])), adapter_path=str(adapter))

        def generate(messages, directory):
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False
            )
            if len(tokens) + 512 > 8192:
                dump(
                    directory / "context-limit.json",
                    {"input_tokens": len(tokens), "max_output_tokens": 512, "context_window": 8192},
                )
                raise ContextLimit()
            mx.random.seed(config["seed"])
            parts = []
            last = None
            call_started = time.monotonic()
            for last in stream_generate(
                model, tokenizer, prompt=tokens, sampler=make_sampler(temp=0), max_tokens=512
            ):
                parts.append(last.text)
            if last is None:
                raise RuntimeError("MLX returned no generation events")
            dump(
                directory / "generation.json",
                {
                    "input_tokens": last.prompt_tokens,
                    "output_tokens": last.generation_tokens,
                    "finish_reason": last.finish_reason,
                    "peak_memory_gb": last.peak_memory,
                    "duration_seconds": time.monotonic() - call_started,
                },
            )
            return "".join(parts)

        results = [
            run_case(
                case,
                response,
                messages,
                generate,
                resolve(config["jar_path"]),
                output / case.name,
                max_attempts,
            )
            for case, _, messages, response in inputs
        ]
        dump(output / "summary.json", results)
        dump(
            output / "complete.json",
            {
                "cases": len(results),
                "successes": sum(row["success"] for row in results),
                "duration_seconds": time.monotonic() - started,
            },
        )
    except BaseException as exc:
        dump(output / "failed.json", {"error_type": type(exc).__name__, "error": str(exc)})
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    print(start(args.training_run, args.max_attempts), flush=True)


if __name__ == "__main__":
    main()
