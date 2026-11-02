"""Prepare, smoke-test and run the bounded synthetic MLX LoRA pilot."""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

from benchmark.sft_data import (
    EXPLAIN_SYSTEM,
    Case,
    dump,
    full_log,
    initial_messages,
    prepare,
    sha,
    verify,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/lora-pilot.json"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def resolve(path: str) -> Path:
    result = Path(path)
    return result if result.is_absolute() else ROOT / result


def read(path: Path):
    return json.loads(path.read_text())


def model_files(path: Path) -> dict:
    return {
        str(file.relative_to(path)): sha(file)
        for file in sorted(path.rglob("*"))
        if file.is_file() and ".cache" not in file.relative_to(path).parts
    }


def tokenizer_audit(config: dict, output: Path) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(resolve(config["model_dir"]), trust_remote_code=False)
    lengths = {}
    maximum = config["training"]["max_seq_length"]
    for split in ("train", "valid", "test"):
        rows = [
            json.loads(line)
            for line in (resolve(config["dataset_dir"]) / f"{split}.jsonl").read_text().splitlines()
        ]
        sizes = []
        for index, row in enumerate(rows):
            messages = row["messages"]
            if messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip():
                raise ValueError(f"Missing supervised answer: {split}:{index}")
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_dict=False
            )
            if isinstance(tokens, dict):
                tokens = tokens["input_ids"]
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(type(x) is not int for x in tokens)
            ):
                raise ValueError("Tokenizer did not return a flat token ID sequence")
            size = len(tokens)
            if size > maximum:
                raise ValueError(
                    f"Refusing token truncation: {split}:{index} has {size} > {maximum}"
                )
            sizes.append(size)
        lengths[split] = {
            "count": len(sizes),
            "min": min(sizes),
            "max": max(sizes),
            "longest_row": sizes.index(max(sizes)),
        }
    report = {"max_seq_length": maximum, "splits": lengths, "truncation": False}
    dump(output, report)
    return report


def dataset_files(path: Path) -> dict:
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def config_hash(config: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def seal_preparation(config: dict) -> None:
    dataset = resolve(config["dataset_dir"])
    dump(
        dataset.parent / "preparation.json",
        {
            "config_sha256": config_hash(config),
            "dataset_files": dataset_files(dataset),
        },
    )


def validate(config: dict, require_preflight: bool = True) -> dict:
    model_manifest = read(resolve(config["model_dir"]).parent / "model-manifest.json")
    if (
        model_manifest["revision"] != config["model_revision"]
        or model_manifest["model_id"] != config["model_id"]
    ):
        raise ValueError("Model identity differs from preparation manifest")
    if model_files(resolve(config["model_dir"])) != model_manifest["files"]:
        raise ValueError("Prepared model files changed")
    dataset = resolve(config["dataset_dir"])
    manifest = read(dataset / "manifest.json")
    for split in ("train", "valid", "test"):
        if sha(dataset / f"{split}.jsonl") != manifest["splits"][split]["sha256"]:
            raise ValueError(f"Prepared dataset changed: {split}")
    families = [set(manifest["splits"][split]["families"]) for split in ("train", "valid", "test")]
    if any(families[i] & families[j] for i in range(3) for j in range(i)):
        raise ValueError("Synthetic task family leaks across splits")
    if sha(resolve(config["jar_path"])) != config["jar_sha256"]:
        raise ValueError("Unexpected TLA+ tools JAR")
    if require_preflight:
        sealed = read(dataset.parent / "preparation.json")
        if sealed["config_sha256"] != config_hash(config):
            raise ValueError("Configuration changed after preparation")
        if sealed["dataset_files"] != dataset_files(dataset):
            raise ValueError("Prepared evidence or evaluation fixtures changed")
    return manifest


def prepare_all(config: dict) -> None:
    from huggingface_hub import snapshot_download

    model = resolve(config["model_dir"])
    model_manifest = model.parent / "model-manifest.json"
    if not model_manifest.exists():
        proxy_keys = (
            "ALL_PROXY",
            "all_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
        )
        saved = {key: os.environ.pop(key) for key in proxy_keys if key in os.environ}
        try:
            snapshot_download(
                config["model_id"],
                revision=config["model_revision"],
                local_dir=model,
                max_workers=3,
            )
        finally:
            os.environ.update(saved)
    if not model_manifest.exists():
        dump(
            model_manifest,
            {
                "model_id": config["model_id"],
                "revision": config["model_revision"],
                "files": model_files(model),
            },
        )
    jar = resolve(config["jar_path"])
    if not jar.exists():
        with build_opener(ProxyHandler({})).open(config["jar_url"], timeout=60) as response:
            data = response.read()
        jar.parent.mkdir(parents=True, exist_ok=True)
        with jar.open("xb") as file:
            file.write(data)
    if sha(jar) != config["jar_sha256"]:
        raise ValueError("JAR checksum mismatch")
    dataset = resolve(config["dataset_dir"])
    if not (dataset / "manifest.json").exists():
        prepare(dataset, jar)
    validate(config, require_preflight=False)
    audit = dataset / "token-audit.json"
    if not audit.exists():
        tokenizer_audit(config, audit)
    if not (dataset.parent / "preparation.json").exists():
        seal_preparation(config)
    validate(config)
    print("Prepared and verified. Use smoke before the long train command.", flush=True)


def run_process(command: list[str], directory: Path, label: str) -> None:
    with (directory / f"{label}.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                clean = ANSI.sub("", line)
                if re.search(
                    r"(?i)(?:loss\s*[:=]?\s*|^\s*\d+\s+(?:val\s+)?)[-+]?(?:nan|inf)\b",
                    clean,
                ):
                    process.terminate()
                    raise ValueError("Nonfinite training loss; stopping")
            code = process.wait()
            if code:
                raise RuntimeError(f"{label} exited with code {code}; see {directory}")
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def evaluate(
    config: dict,
    directory: Path,
    adapter: Path | None,
    limit: int | None = None,
    max_tokens: int = 512,
) -> dict:
    """Disjoint synthetic test families, evaluated before and after training.

    Only task text/cfg enters generation. Repair sees deliberately broken test
    fixtures. These are separate from the four coursework tasks and their Gold.
    """
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    directory.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load(
        str(resolve(config["model_dir"])), adapter_path=str(adapter) if adapter else None
    )
    test_cases = [
        Case(**case)
        for case in read(resolve(config["dataset_dir"]) / "cases.json")
        if case["split"] == "test"
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError("Evaluation case limit must be positive")
        test_cases = test_cases[:limit]
    rows = []
    for index, case in enumerate(test_cases):
        evidence = resolve(config["dataset_dir"]) / "evidence" / case.name / "broken"
        log = full_log(evidence)
        repair = initial_messages(case) + [
            {"role": "assistant", "content": case.broken},
            {"role": "user", "content": "Return the corrected complete module.\n" + log},
        ]
        explanation = [
            {"role": "system", "content": EXPLAIN_SYSTEM},
            {"role": "user", "content": case.broken + "\nVerification log:\n" + log},
        ]
        for kind, messages in (
            ("generation", initial_messages(case)),
            ("repair", repair),
            ("explanation", explanation),
        ):
            out = directory / case.name / kind
            out.mkdir(parents=True, exist_ok=False)
            dump(out / "messages.json", messages)
            mx.random.seed(config["seed"] + index)
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            parts = []
            last = None
            for last in stream_generate(
                model, tokenizer, prompt=prompt, sampler=make_sampler(temp=0), max_tokens=max_tokens
            ):
                parts.append(last.text)
            text = "".join(parts)
            if last is None:
                raise RuntimeError("MLX returned no generation events")
            dump(
                out / "generation.json",
                {
                    "input_tokens": last.prompt_tokens,
                    "output_tokens": last.generation_tokens,
                    "finish_reason": last.finish_reason,
                    "peak_memory_gb": last.peak_memory,
                },
            )
            (out / "response.txt").write_text(text)
            row = {"task": case.name, "kind": kind}
            if kind == "explanation":
                row["format_ok"] = bool(text.strip()) and not re.search(
                    r"```|----\s*MODULE|====|(?m:^\s*\w+\s*==)", text
                )
                row["semantic_accuracy"] = "not_automatically_graded"
            else:
                blocks = re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S)
                source = blocks[0] if len(blocks) == 1 else text
                # Model stays loaded for evaluation; bound Java memory and use one worker.
                checks = verify(
                    source, case.cfg, case.name, resolve(config["jar_path"]), out / "checks"
                )
                row["sany"] = checks["sany"]["success"]
                row["tlc"] = checks.get("tlc", {}).get("success", False)
                row["expected_state_count"] = (
                    checks.get("tlc", {}).get("distinct_states") == case.expected_states
                )
            rows.append(row)
            print(json.dumps(row), flush=True)
    dump(directory / "summary.json", rows)
    return {
        "records": len(rows),
        "generation_tlc": sum(r.get("tlc", False) for r in rows if r["kind"] == "generation"),
        "repair_tlc": sum(r.get("tlc", False) for r in rows if r["kind"] == "repair"),
        "explanation_format": sum(
            r.get("format_ok", False) for r in rows if r["kind"] == "explanation"
        ),
        "cases": len(test_cases),
    }


def start(config: dict, smoke: bool) -> Path:
    import importlib.metadata

    if importlib.metadata.version("mlx-lm") != "0.32.2":
        raise ValueError("Use the pinned MLX environment from requirements-mlx.txt")
    validate(config)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "runs" / f"lora-{'smoke' if smoke else 'pilot'}-{stamp}-{uuid4().hex[:6]}"
    directory.mkdir(parents=True, exist_ok=False)
    audit = tokenizer_audit(config, directory / "token-audit.json")
    training = {
        **config["training"],
        "model": str(resolve(config["model_dir"])),
        "data": str(resolve(config["dataset_dir"])),
        "adapter_path": str(directory / "adapter"),
        "seed": config["seed"],
    }
    if smoke:
        data = directory / "smoke-data"
        data.mkdir()
        for split in ("train", "valid"):
            lines = (resolve(config["dataset_dir"]) / f"{split}.jsonl").read_text().splitlines()
            (data / f"{split}.jsonl").write_text(
                lines[audit["splits"][split]["longest_row"]] + "\n"
            )
        training.update(iters=1, val_batches=1, steps_per_report=1, save_every=1, data=str(data))
    dump(directory / "training.json", training)
    versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (directory / "environment.txt").write_text(versions)
    dump(
        directory / "manifest.json",
        {
            "phase": "smoke" if smoke else "synthetic_lora_pilot",
            "config": config,
            "source_files": {
                name: sha(ROOT / name)
                for name in ("src/benchmark/finetune.py", "src/benchmark/sft_data.py")
            },
            "model_manifest_sha256": sha(
                resolve(config["model_dir"]).parent / "model-manifest.json"
            ),
            "dataset_manifest_sha256": sha(resolve(config["dataset_dir"]) / "manifest.json"),
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
            "evaluation": (
                "greedy; 512 token cap; exact raw outputs retained; "
                "state counts are a sanity check, not a semantic equivalence proof"
            ),
        },
    )
    print(f"Run directory: {directory}", flush=True)
    try:
        if not smoke:
            run_process(
                [
                    sys.executable,
                    "-m",
                    "benchmark.finetune",
                    "evaluate",
                    "--config-json",
                    json.dumps(config),
                    "--output",
                    str(directory / "baseline"),
                ],
                directory,
                "baseline",
            )
        run_process(
            [
                sys.executable,
                "-m",
                "benchmark.finetune",
                "fit",
                "--output",
                str(directory),
            ],
            directory,
            "training",
        )
        import mlx.core as mx

        weights = mx.load(str(directory / "adapter" / "adapters.safetensors"))
        if not weights or not all(bool(mx.all(mx.isfinite(value))) for value in weights.values()):
            raise ValueError("Adapter contains nonfinite weights")
        if not smoke:
            for name, expected in read(directory / "manifest.json")["source_files"].items():
                if sha(ROOT / name) != expected:
                    raise ValueError(
                        "Source changed during training; refusing mixed-code evaluation"
                    )
            run_process(
                [
                    sys.executable,
                    "-m",
                    "benchmark.finetune",
                    "evaluate",
                    "--config-json",
                    json.dumps(config),
                    "--output",
                    str(directory / "adapted"),
                    "--adapter",
                    str(directory / "adapter"),
                ],
                directory,
                "adapted",
            )
        dump(
            directory / "complete.json",
            {"smoke": smoke, "adapter_sha256": sha(directory / "adapter" / "adapters.safetensors")},
        )
    except BaseException as exc:
        dump(directory / "failed.json", {"error_type": type(exc).__name__, "error": str(exc)})
        raise
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "smoke", "train", "evaluate", "fit"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--config-json", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, default=512, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == "fit":
        if not args.output:
            parser.error("fit requires --output")
        import time

        import mlx.core as mx
        from mlx_lm.lora import main as mlx_train

        started = time.monotonic()
        sys.argv = ["mlx_lm.lora", "-c", str(args.output / "training.json"), "--train"]
        try:
            mlx_train()
        finally:
            peak = mx.get_peak_memory()
            dump(
                args.output / "memory.json",
                {
                    "mlx_peak_bytes": peak,
                    "duration_seconds": time.monotonic() - started,
                },
            )
            print(f"MLX peak allocated memory: {peak / 1024**3:.2f} GiB", flush=True)
        return
    config = json.loads(args.config_json) if args.config_json else read(args.config)
    if args.command == "prepare":
        prepare_all(config)
    elif args.command == "evaluate":
        if not args.output:
            parser.error("evaluate requires --output")
        validate(config)
        print(
            json.dumps(evaluate(config, args.output, args.adapter, args.limit, args.max_tokens)),
            flush=True,
        )
    else:
        print(start(config, args.command == "smoke"), flush=True)


if __name__ == "__main__":
    main()
